import numpy as np
import pandas as pd
import pytest

from scripts.core.data import Dataset, WindowSpec
from scripts.core.errors import ConfigError
from scripts.core.targets import build_labels
from scripts.data.baseline_loader import FEATURE_NAMES
from scripts.models.hourly_logistic import HourlyLogistic
from scripts.models.registry import build

SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)


def dataset(people=("COPS-1", "COPS-2", "COPS-3", "COPS-4"), hours=48, seed=0):
    """OFF hours move less and come long after a dose; tremor hours have more 4-8 Hz power."""
    rng = np.random.default_rng(seed)
    rows, feats, events = [], [], []
    for p in people:
        for h in range(hours):
            score = int(rng.choice([-2, -1, 0, 0, 1]))
            tremor = int(rng.integers(0, 3))
            move = 0.02 * (1 + 0.6 * np.sign(score)) * rng.lognormal(0, 0.2)
            trem = 0.01 * (1 + tremor) * rng.lognormal(0, 0.2)
            rows.append({"participant_id": p, "hour_index": h, "hour_start": h % 24, "kinesia_text": "Good Kinesia",
                         "kinesia_score": float(score), "tremor_score": float(tremor)})
            feats.append({"move_mean": move, "move_std": move, "move_p95": 3 * move,
                          "still_fraction": float(np.clip(0.6 - 10 * move, 0, 1)), "tremor_mean": trem,
                          "tremor_p95": 3 * trem, "tremor_ratio": trem / move})
        events += [{"participant_id": p, "time_s": t * 3600 + 1800, "kind": "dose", "ledd": 100.0,
                    "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan}
                   for t in range(0, hours, 8)]
    table = pd.DataFrame(rows)
    return Dataset(spec=SPEC, hours=table, labels=build_labels(table, ["state", "tremor"]),
                   features=pd.DataFrame(feats)[FEATURE_NAMES], events=pd.DataFrame(events))


def model(**params):
    return HourlyLogistic(["state", "tremor"], ["now", "next_hour"], params)


def test_fits_and_predicts_every_hour_with_the_targets_classes():
    data = dataset()
    out = model().fit(data.subset(["COPS-1", "COPS-2", "COPS-3"]), seed=0).predict(data.subset(["COPS-4"]))
    assert out.probs["state_now"].shape == (48, 7) and out.probs["tremor_now"].shape == (48, 3)
    assert set(out.labels) == {"state_now", "state_next_hour", "tremor_now", "tremor_next_hour"}


def test_it_learns_the_signal():
    from sklearn.metrics import roc_auc_score
    data = dataset(people=[f"COPS-{i}" for i in range(1, 9)])
    test = data.subset(["COPS-7", "COPS-8"])
    out = model().fit(data.subset([f"COPS-{i}" for i in range(1, 7)]), seed=0).predict(test)
    y = test.labels["state_now"].to_numpy()
    assert roc_auc_score(y < 3, out.p_state("state_now", "off")) > 0.8
    assert roc_auc_score(test.labels["tremor_now"] > 0, 1 - out.probs["tremor_now"][:, 0]) > 0.8


def test_state_groups_are_spread_over_their_scores():
    data = dataset()
    out = model().fit(data, seed=0).predict(data)
    p = out.probs["state_now"]
    assert (p[:, 0] == 0).all() or p[:, 0].max() < p[:, 1].max()   # -3 never seen: only the +1 prior
    assert np.allclose(p.sum(axis=1), 1)


def test_save_and_load_give_the_same_predictions(tmp_path):
    data = dataset()
    fitted = model().fit(data, seed=0)
    loaded = HourlyLogistic.load(fitted.save(tmp_path / "m"))
    np.testing.assert_allclose(fitted.predict(data).probs["state_now"], loaded.predict(data).probs["state_now"])


def test_the_registry_builds_it_and_refuses_bad_settings():
    name, m = build({"name": "hourly_logistic", "family": "hourly_logistic", "params": {"c": 0.5}}, SPEC,
                    ["state"], ["now"])
    assert isinstance(m, HourlyLogistic) and m.params["c"] == 0.5
    with pytest.raises(ConfigError):
        model(class_weight="sometimes")
    with pytest.raises(ConfigError):
        model(features={"warmup": 3})
