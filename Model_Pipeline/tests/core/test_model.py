from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.core.data import WindowSpec
from scripts.core.errors import ConfigError, ModelError, NotFittedError, SchemaError
from scripts.core.model import FORMAT_VERSION, MANIFEST, Model, Predictions, WindowPredictions
from scripts.core.targets import get_target, parse_label
from scripts.utils.file_io import read_json, write_json
from tests.helpers import make_dataset

SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)


class Prior(Model):
    """A test family: every hour gets each label's class frequencies from training."""

    family = "prior_test"
    needs = ("features",)

    @classmethod
    def default_params(cls):
        return {"smoothing": 1.0}

    def _fit(self, train, val, sample_weight, seed):
        self.freqs = {}
        for label in self.labels:
            k = len(get_target(parse_label(label)[0]).classes)
            y = train.labels[label].to_numpy()
            counts = np.bincount(y[~np.isnan(y)].astype(int), minlength=k) + self.params["smoothing"]
            self.freqs[label] = (counts / counts.sum()).tolist()

    def _predict(self, data):
        n = len(data.hours)
        return Predictions(data.hours, {label: np.tile(f, (n, 1)) for label, f in self.freqs.items()})

    def _save_weights(self, folder):
        write_json(self.freqs, folder / "freqs.json")

    def _load_weights(self, folder):
        self.freqs = read_json(folder / "freqs.json")


@pytest.fixture
def data():
    return make_dataset(SPEC, n_participants=3)


@pytest.fixture
def split(data):
    return data.subset(["COPS-1", "COPS-2"]), data.subset(["COPS-3"])


def hours(n=2):
    return pd.DataFrame({"participant_id": ["P1"] * n, "hour_index": list(range(n))})


def state_probs(*rows):
    return np.array(rows, dtype=float)


# ---- predictions ------------------------------------------------------------------------------

def test_p_off_p_asleep_and_expected_score_come_from_the_probabilities():
    # Hour 0: all slight akinesia (-1). Hour 1: half good (0), half slight dyskinesia (+1).
    p = state_probs([0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0.5, 0.5, 0, 0])
    preds = Predictions(hours(), {"state_now": p, "sleep_now": [[1, 0], [0.4, 0.6]]})
    np.testing.assert_allclose(preds.p_state("state_now", "off"), [1, 0])
    np.testing.assert_allclose(preds.p_state("state_now", "dyskinetic"), [0, 0.5])
    np.testing.assert_allclose(preds.p_asleep("sleep_now"), [0, 0.6])
    np.testing.assert_allclose(preds.expected_score("state_now"), [-1, 0.5])
    assert preds.predicted("state_now").tolist() == [2, 3]


def test_expected_tremor_and_no_score_for_sleep():
    preds = Predictions(hours(1), {"tremor_now": [[0.5, 0.25, 0.25]], "sleep_now": [[0, 1]]})
    np.testing.assert_allclose(preds.expected_score("tremor_now"), [0.75])
    assert np.isnan(preds.expected_score("sleep_now")[0])   # sleep has no diary score


def test_p_asleep_and_p_state_refuse_the_wrong_label():
    preds = Predictions(hours(1), {"sleep_now": [[0.5, 0.5]], "state_now": [[1] + [0] * 6]})
    with pytest.raises(ConfigError, match="sleep label"):
        preds.p_asleep("state_now")
    with pytest.raises(ConfigError):
        preds.p_state("state_now", "asleep")   # asleep is the sleep output's now


def test_frame_has_a_column_per_class():
    frame = Predictions(hours(1), {"tremor_now": [[0.2, 0.3, 0.5]]}).frame("tremor_now")
    assert list(frame.columns) == ["participant_id", "hour_index", "none", "weak_or_slight", "strong_or_severe"]


def test_predictions_cant_be_changed_afterwards():
    preds = Predictions(hours(1), {"tremor_now": [[0.2, 0.3, 0.5]]})
    with pytest.raises(ValueError):
        preds.probs["tremor_now"][0, 0] = 1


@pytest.mark.parametrize("probs, match", [
    ({"tremor_now": [[0.2, 0.3, 0.5]]}, r"\(1, 3\), need \(2, 3\)"),
    ({"tremor_now": [[0.2, 0.3, 0.4], [0.2, 0.3, 0.5]]}, "add up to 1"),
    ({"tremor_now": [[1.5, -0.5, 0], [0.2, 0.3, 0.5]]}, "non-negative"),
    ({"tremor_now": [[np.nan, 0.5, 0.5], [0.2, 0.3, 0.5]]}, "finite"),
    ({"freezing_now": [[1.0], [1.0]]}, "unknown label"),
    ({}, "no labels"),
])
def test_bad_predictions_are_refused(probs, match):
    with pytest.raises(ModelError, match=match):
        Predictions(hours(), probs)


def test_window_predictions_are_checked_too():
    hourly = {"tremor_now": [[0.2, 0.3, 0.5]]}
    good = WindowPredictions(np.array([0, 10]), {"tremor_now": [[1, 0, 0], [0, 1, 0]]})
    assert Predictions(hours(1), hourly, windows={"P1": good}).windows["P1"].time_s.tolist() == [0, 10]
    with pytest.raises(ModelError, match=r"window tremor_now of P1"):
        Predictions(hours(1), hourly, windows={"P1": WindowPredictions(np.array([0]), good.probs)})
    with pytest.raises(ModelError, match="labels the hourly ones don't"):
        Predictions(hours(1), hourly, windows={"P1": WindowPredictions(np.array([0]), {"state_now": [[1] + [0] * 7]})})


# ---- building a model ---------------------------------------------------------------------------

def test_labels_are_every_target_at_every_horizon():
    assert Prior(["state", "tremor"]).labels == ("state_now", "state_next_hour", "tremor_now", "tremor_next_hour")
    assert Prior(["tremor"], horizons=["now"]).labels == ("tremor_now",)


def test_settings_merge_with_defaults_and_typos_are_refused():
    assert Prior(["state"]).params == {"smoothing": 1.0}
    assert Prior(["state"], params={"smoothing": 0.5}).params == {"smoothing": 0.5}
    with pytest.raises(ConfigError, match="no setting"):
        Prior(["state"], params={"smoothin": 0.5})
    with pytest.raises(ConfigError, match="plain|numbers"):
        Prior(["state"], params={"smoothing": object()})


@pytest.mark.parametrize("kwargs, match", [
    ({"targets": ["freezing"]}, "unknown target"),
    ({"targets": ["state"], "horizons": ["tomorrow"]}, "unknown horizon"),
    ({"targets": []}, "at least one target"),
])
def test_bad_model_settings_are_refused(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        Prior(**kwargs)


def test_a_family_must_name_itself_and_its_groups():
    class Nameless(Prior):
        family = ""

    class Reads(Prior):
        family, needs = "reads_test", ("medication",)

    class WindowReader(Prior):
        family, needs = "windows_test", ("windows",)

    with pytest.raises(ConfigError, match="family name"):
        Nameless(["state"])
    with pytest.raises(ConfigError, match="unknown group"):
        Reads(["state"])
    with pytest.raises(ConfigError, match="window spec"):
        WindowReader(["state"])
    assert WindowReader(["state"], spec=SPEC).spec == SPEC


# ---- fit and predict ------------------------------------------------------------------------------

def test_fit_then_predict_one_row_per_hour(split):
    train, test = split
    model = Prior(["state", "tremor"]).fit(train, seed=3)
    preds = model.predict(test)
    assert model.is_fitted and set(preds.labels) == set(model.labels)
    assert preds.hours.equals(test.hours[["participant_id", "hour_index"]].reset_index(drop=True))
    assert model.manifest()["training"]["seed"] == 3


def test_predict_or_save_before_fit_is_refused(split, tmp_path):
    with pytest.raises(NotFittedError):
        Prior(["state"]).predict(split[1])
    with pytest.raises(NotFittedError):
        Prior(["state"]).save(tmp_path / "m")


def test_train_and_validation_must_be_different_participants(data):
    with pytest.raises(SchemaError, match="both training and validation"):
        Prior(["state"]).fit(data.subset(["COPS-1", "COPS-2"]), val=data.subset(["COPS-2"]))


def test_the_groups_and_labels_the_model_needs_must_be_there(split):
    train, _ = split
    with pytest.raises(SchemaError, match="features group"):
        Prior(["state"]).fit(replace(train, features=None))
    with pytest.raises(SchemaError, match="no \\['tremor_now'"):
        Prior(["tremor"], horizons=["now"]).fit(replace(train, labels=train.labels.drop(columns="tremor_now")))


@pytest.mark.parametrize("seed", [-1, 1.5, True, "0"])
def test_bad_seeds_are_refused(split, seed):
    with pytest.raises(ConfigError, match="seed"):
        Prior(["state"]).fit(split[0], seed=seed)


@pytest.mark.parametrize("weights", ["wrong_length", "all_zero", "negative", "nan"])
def test_bad_sample_weights_are_refused(split, weights):
    train, _ = split
    n = len(train.hours)
    w = {"wrong_length": np.ones(n + 1), "all_zero": np.zeros(n), "negative": -np.ones(n),
         "nan": np.full(n, np.nan)}[weights]
    with pytest.raises(ConfigError, match="sample_weight"):
        Prior(["state"]).fit(train, sample_weight=w)


@pytest.mark.parametrize("bad_output, match", [
    (lambda self, data: {"state_now": None}, "must return Predictions"),
    (lambda self, data: Predictions(data.hours, {"tremor_now": np.full((len(data.hours), 3), 1 / 3)}),
     "but its labels are"),
    (lambda self, data: Predictions(data.hours.iloc[::-1],
                                    {"state_now": np.full((len(data.hours), 7), 1 / 7)}), "same order"),
])
def test_a_family_that_breaks_the_contract_is_caught(split, bad_output, match):
    class Broken(Prior):
        family = "broken_test"
        _predict = bad_output

    model = Broken(["state"], horizons=["now"]).fit(split[0])
    with pytest.raises(ModelError, match=match):
        model.predict(split[1])


# ---- save and load --------------------------------------------------------------------------------

def test_save_then_load_gives_the_same_predictions(split, tmp_path):
    train, test = split
    model = Prior(["state", "tremor"], params={"smoothing": 0.5}).fit(train, seed=7)
    folder = model.save(tmp_path / "prior")
    loaded = Prior.load(folder)
    for label in model.labels:
        np.testing.assert_array_equal(loaded.predict(test).probs[label], model.predict(test).probs[label])
    manifest = read_json(folder / MANIFEST)
    assert manifest["format_version"] == FORMAT_VERSION and manifest["family"] == "prior_test"
    assert manifest["params"] == {"smoothing": 0.5} and manifest["training"]["train_participants"] == 2
    assert {"git_commit", "python", "packages"} <= set(manifest["environment"])
    assert loaded.params == model.params and loaded.is_fitted


def test_a_model_is_never_overwritten(split, tmp_path):
    model = Prior(["state"]).fit(split[0])
    model.save(tmp_path / "m")
    with pytest.raises(ModelError, match="never overwritten"):
        model.save(tmp_path / "m")
    (tmp_path / "empty").mkdir()
    assert (model.save(tmp_path / "empty") / MANIFEST).exists()   # an empty folder is fine


def test_a_failed_save_leaves_nothing_behind(split, tmp_path):
    class FailsToSave(Prior):
        family = "fails_test"

        def _save_weights(self, folder):
            raise OSError("disk full")

    with pytest.raises(OSError):
        FailsToSave(["state"]).fit(split[0]).save(tmp_path / "m")
    assert list(tmp_path.iterdir()) == []


def test_load_refuses_another_family_or_format(split, tmp_path):
    folder = Prior(["state"]).fit(split[0]).save(tmp_path / "m")

    class Other(Prior):
        family = "other_test"

    with pytest.raises(ModelError, match="holds a 'prior_test' model"):
        Other.load(folder)
    manifest = read_json(folder / MANIFEST)
    write_json({**manifest, "format_version": FORMAT_VERSION + 1}, folder / MANIFEST)
    with pytest.raises(ModelError, match="manifest format"):
        Prior.load(folder)
    with pytest.raises(ModelError, match="no manifest.json"):
        Prior.load(tmp_path / "nothing")


# ---- window models ----------------------------------------------------------------------------------

class WindowPrior(Prior):
    """A test window model: the same frequencies on every worn window, and per hour."""

    family = "window_prior_test"
    needs = ("windows",)
    scores_windows = True
    skip = ()   # participants to leave out, to test the checks

    def _predict(self, data):
        hourly = super()._predict(data)
        windows = {}
        for pid in data.participants:
            if pid in self.skip:
                continue
            times = data.windows[pid].time_s[data.windows[pid].worn]
            windows[pid] = WindowPredictions(times, {label: np.tile(f, (len(times), 1))
                                                     for label, f in self.freqs.items()})
        return Predictions(hourly.hours, hourly.probs, windows=windows)


def with_unworn_windows(data, pid, every=5):
    w = data.windows[pid]
    worn = w.worn.copy()
    worn[::every] = False
    return replace(data, windows={**data.windows, pid: replace(w, worn=worn)})


def test_a_window_model_scores_every_worn_window(split):
    train, test = split
    test = with_unworn_windows(test, "COPS-3")
    preds = WindowPrior(["state"], spec=SPEC).fit(train).predict(test)
    w = test.windows["COPS-3"]
    np.testing.assert_array_equal(preds.windows["COPS-3"].time_s, w.time_s[w.worn])
    assert set(preds.windows["COPS-3"].probs) == {"state_now", "state_next_hour"}
    assert WindowPrior(["state"], spec=SPEC).manifest()["scores_windows"] is True


def test_a_window_model_must_cover_every_participant(data):
    model = WindowPrior(["state"], spec=SPEC).fit(data.subset(["COPS-1"]))
    model.skip = ("COPS-3",)
    with pytest.raises(ModelError, match="no windows for 1 participant"):
        model.predict(data.subset(["COPS-2", "COPS-3"]))


def test_a_window_model_must_score_the_worn_windows_exactly(split):
    class ScoresAll(WindowPrior):
        family = "scores_all_test"

        def _predict(self, data):   # scores unworn windows too: wrong
            out = super()._predict(replace(data, windows={
                pid: replace(w, worn=np.ones(len(w), dtype=bool)) for pid, w in data.windows.items()}))
            return out

    train, test = split
    with pytest.raises(ModelError, match="every worn window"):
        ScoresAll(["state"], spec=SPEC).fit(train).predict(with_unworn_windows(test, "COPS-3"))


def test_window_predictions_need_every_label(split):
    class OneLabel(WindowPrior):
        family = "one_label_test"

        def _predict(self, data):
            out = super()._predict(data)
            return Predictions(out.hours, out.probs, windows={
                pid: WindowPredictions(w.time_s, {"state_now": w.probs["state_now"]})
                for pid, w in out.windows.items()})

    with pytest.raises(ModelError, match="need every label"):
        OneLabel(["state"], spec=SPEC).fit(split[0]).predict(split[1])


def test_an_hourly_model_must_not_return_windows(split):
    class Leaky(Prior):
        family = "leaky_test"

        def _predict(self, data):
            out = super()._predict(data)
            pid = data.participants[0]
            return Predictions(out.hours, out.probs, windows={pid: WindowPredictions(np.array([0]), {})})

    with pytest.raises(ModelError, match="doesn't score windows"):
        Leaky(["state"]).fit(split[0]).predict(split[1])


def test_a_window_model_must_read_windows():
    class NoWindows(Prior):
        family, scores_windows = "no_windows_test", True

    with pytest.raises(ConfigError, match="must need the windows group"):
        NoWindows(["state"])
