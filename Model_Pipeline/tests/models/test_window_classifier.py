import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from model_pipeline.core.data import Dataset, Windows, WindowSpec
from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.core.model import Predictions
from model_pipeline.models.motor_causal_transformer import harnet as harnet_module
from model_pipeline.models.window_classifier import WindowBoosting, WindowLogistic

H = 3600
SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)
FAMILIES = [(WindowLogistic, {}), (WindowBoosting, {"pca_components": 8, "iterations": 30})]
UNWORN_HOUR = 4


class Fake(nn.Module):
    """Stands in for harnet10: the window's mean per axis, spread over 1,024 features."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, 1024)

    def forward(self, x):
        return self.proj(x.mean(-1)).unsqueeze(-1)


@pytest.fixture(autouse=True)
def environment(tmp_path, monkeypatch):
    """CPU, a cache in tmp_path, and the fake harnet10 (named "fake-weights")."""
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("MODEL_FEATURE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(harnet_module.Harnet10, "load",
                        classmethod(lambda cls, hub, device="cpu", batch_size=2048: cls(Fake(), "fake-weights")))


def dataset(pids=("A", "B", "C", "D"), n_hours=10, seed=0):
    """Windows every 2 min (30 an hour), none worn in hour 4; the movement follows the state (OFF or good)."""
    rng = np.random.default_rng(seed)
    windows, hours, labels = {}, [], []
    for pid in pids:
        t = np.arange(0, n_hours * H, 120.0)
        states = np.tile([0.0, 3.0], n_hours // 2)
        rng.shuffle(states)
        signal = rng.normal(size=(len(t), 3, 300)).astype(np.float32) * 0.1
        signal += states[(t // H).astype(int)][:, None, None] * 0.2
        windows[pid] = Windows(t, (t // H) != UNWORN_HOUR, signal)
        for h in range(n_hours):
            hours.append({"participant_id": pid, "hour_index": h})
            labels.append({"sleep_now": float(h == 0), "state_now": states[h], "tremor_now": float(states[h] == 0),
                           "sleep_next_hour": float(h == 0),
                           "state_next_hour": states[h + 1] if h + 1 < n_hours else np.nan,
                           "tremor_next_hour": np.nan})
    return Dataset(SPEC, pd.DataFrame(hours), pd.DataFrame(labels), windows=windows)


def fitted(family, params, data=None, **fit):
    data = data or dataset()
    model = family(["sleep", "state", "tremor"], params=params, spec=SPEC)
    return model.fit(data.subset(["A", "B", "C"]), seed=0, **fit), data


@pytest.mark.parametrize("family, params", FAMILIES)
def test_learns_the_state_and_follows_the_contract_on_new_patients(family, params):
    model, data = fitted(family, params)
    out = model.predict(data.subset(["D"]))         # the base class checks hours, labels, sums and windows
    assert isinstance(out, Predictions) and set(out.labels) == set(model.labels) and len(out.hours) == 10
    w = data.windows["D"]
    assert np.array_equal(out.windows["D"].time_s, w.time_s[w.worn])
    p_off = out.p_state("state_now", "off")
    state = data.subset(["D"]).labels["state_now"].to_numpy()
    worn = out.hours.hour_index.to_numpy() != UNWORN_HOUR
    assert p_off[worn & (state == 0)].min() > p_off[worn & (state == 3)].max()


@pytest.mark.parametrize("family, params", FAMILIES)
def test_an_hour_is_the_mean_of_its_windows_and_an_unworn_hour_gets_the_class_mix(family, params):
    model, data = fitted(family, params)
    out = model.predict(data)
    for pid in ("A", "D"):
        w = out.windows[pid]
        in_hour = (w.time_s // H) == 2
        row = out.hours.index[(out.hours.participant_id == pid) & (out.hours.hour_index == 2)][0]
        np.testing.assert_allclose(out.probs["state_now"][row], w.probs["state_now"][in_hour].mean(0), atol=1e-12)
    unworn = out.hours.index[out.hours.hour_index == UNWORN_HOUR]
    p = out.probs["state_now"][unworn]
    assert np.allclose(p, p[0]) and np.allclose(p[0], model._prior["state_now"])


def test_training_reads_at_most_windows_per_hour_from_each_answered_hour():
    model, _ = fitted(WindowLogistic, {"windows_per_hour": 7})
    r = model.training_result
    # 3 patients x 9 worn hours, every one answered (sleep_now), 7 windows each.
    assert r["windows"] == 3 * 9 * 7
    assert r["labels"]["state_now"]["windows"] == 3 * 9 * 7 and r["labels"]["state_next_hour"]["windows"] == 3 * 8 * 7
    same, _ = fitted(WindowLogistic, {"windows_per_hour": 7})
    data = dataset()
    np.testing.assert_array_equal(model.predict(data).probs["state_now"], same.predict(data).probs["state_now"])


def test_a_sample_weight_of_zero_is_the_same_as_leaving_the_hours_out():
    data = dataset()
    train = data.subset(["A", "B", "C"])
    weight = np.where(train.hours.participant_id.to_numpy() == "C", 0.0, 1.0)
    weighted = WindowLogistic(["state"], ["now"], spec=SPEC).fit(train, sample_weight=weight, seed=0)
    without = WindowLogistic(["state"], ["now"], spec=SPEC).fit(data.subset(["A", "B"]), seed=0)
    np.testing.assert_allclose(weighted.predict(data).probs["state_now"], without.predict(data).probs["state_now"],
                               atol=1e-9)
    assert weighted._prior == without._prior


def test_a_label_with_one_class_in_training_predicts_the_class_mix():
    data = dataset()
    labels = data.labels.assign(sleep_now=0.0)       # never asleep
    data = Dataset(SPEC, data.hours, labels, windows=data.windows)
    model = WindowLogistic(["sleep"], ["now"], spec=SPEC).fit(data.subset(["A", "B"]), seed=0)
    out = model.predict(data)
    assert np.allclose(out.probs["sleep_now"], model._prior["sleep_now"])
    assert model.training_result["labels"]["sleep_now"]["classes_seen"] == 1


@pytest.mark.parametrize("family, params", FAMILIES)
def test_save_and_load_give_the_same_predictions(tmp_path, family, params):
    model, data = fitted(family, params)
    model.save(tmp_path / "m")
    info = json.loads((tmp_path / "m" / "weights" / "training.json").read_text(encoding="utf-8"))
    assert info["harnet10"] == "fake-weights"
    loaded = family.load(tmp_path / "m")
    a, b = model.predict(data), loaded.predict(data)
    for label in model.labels:
        np.testing.assert_array_equal(a.probs[label], b.probs[label])
    assert loaded.training_result == model.training_result


def test_predict_refuses_other_harnet10_weights(tmp_path, monkeypatch):
    model, data = fitted(WindowLogistic, {})
    model.save(tmp_path / "m")
    monkeypatch.setattr(harnet_module.Harnet10, "load",
                        classmethod(lambda cls, hub, device="cpu", batch_size=2048: cls(Fake(), "other-weights")))
    with pytest.raises(ModelError, match="harnet10 weights"):
        WindowLogistic.load(tmp_path / "m").predict(data)


def test_bad_settings_and_a_spec_harnet10_cant_read_are_refused():
    for family, params, match in ((WindowLogistic, {"c": 0}, "c must"), (WindowLogistic, {"max_iter": 1.5}, "max_iter"),
                                  (WindowLogistic, {"windows_per_hour": 0}, "windows_per_hour"),
                                  (WindowLogistic, {"class_weight_cap": 0.5}, "class_weight_cap"),
                                  (WindowBoosting, {"leaves": 1}, "leaves"),
                                  (WindowBoosting, {"learning_rate": -1}, "learning_rate"),
                                  (WindowBoosting, {"c": 1}, "no setting")):
        with pytest.raises(ConfigError, match=match):
            family(["state"], params=params, spec=SPEC)
    with pytest.raises(ConfigError, match="window spec"):
        WindowLogistic(["state"])
    other = WindowSpec(version=1, seconds=5, rate_hz=30, axes=3)
    with pytest.raises(ConfigError, match="harnet10 reads"):
        WindowLogistic(["state"], spec=other)._fit(dataset(), None, None, 0)


def test_the_families_are_named_and_read_only_windows():
    assert (WindowLogistic.family, WindowBoosting.family) == ("window_logistic", "window_boosting")
    assert WindowLogistic.needs == WindowBoosting.needs == ("windows",)
    assert WindowLogistic.scores_windows and WindowBoosting.scores_windows


def test_scaling_every_sample_weight_changes_nothing():
    data = dataset()
    train = data.subset(["A", "B", "C"])
    weight = np.where(train.hours.participant_id.to_numpy() == "A", 2.0, 1.0)
    one = WindowLogistic(["state"], ["now"], spec=SPEC).fit(train, sample_weight=weight, seed=0)
    tiny = WindowLogistic(["state"], ["now"], spec=SPEC).fit(train, sample_weight=weight * 0.01, seed=0)
    np.testing.assert_allclose(one.predict(data).probs["state_now"], tiny.predict(data).probs["state_now"], atol=1e-9)
    assert one._prior == tiny._prior


def test_a_tampered_pickle_is_refused_before_it_is_unpickled(tmp_path):
    model, _ = fitted(WindowLogistic, {})
    model.save(tmp_path / "m")
    with open(tmp_path / "m" / "weights" / "classifiers.joblib", "ab") as f:
        f.write(b"extra")
    with pytest.raises(ModelError, match="SHA-256"):
        WindowLogistic.load(tmp_path / "m")


@pytest.mark.parametrize("family, params", FAMILIES)
def test_the_patient_baseline_trains_saves_and_predicts_the_same(tmp_path, family, params):
    data = dataset()
    model, _ = fitted(family, {**params, "baseline": {"mode": "rolling", "window_h": 4, "warmup_h": 1}}, data)
    assert model._cohort is not None
    out = model.predict(data)
    model.save(tmp_path / "m")
    loaded = family.load(tmp_path / "m")
    for label in model.labels:
        np.testing.assert_array_equal(out.probs[label], loaded.predict(data).probs[label])
    plain, _ = fitted(family, params, data)
    assert not np.allclose(out.probs["state_now"], plain.predict(data).probs["state_now"])
