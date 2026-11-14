import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts.core.data import Dataset, Windows, WindowSpec
from scripts.core.errors import ConfigError, ModelError
from scripts.core.model import Predictions
from scripts.models.motor_causal_transformer import harnet as harnet_module
from scripts.models.motor_causal_transformer.model import MotorCausalTransformer

H = 3600
SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)
TINY = {"network": {"movement_params": {"hidden_dim": 16, "output_dim": 16}, "d_model": 16, "dropout": 0.1,
                    "decoder": {"heads": 2, "query_block": 32}, "heads": {"hidden": 8}},
        "pretrain": {"epochs": 1, "chunk_h": 4},
        "diary": {"max_epochs": 2, "patience": 1, "chunk_h": 4, "eval_past_h": [1]},
        "score_past_h": 2}


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


def dataset(pids=("A", "B", "C"), n_hours=10, seed=0):
    """Windows every 2 min, except hour 4 of every participant (unworn); its movement follows the state."""
    rng = np.random.default_rng(seed)
    windows, hours, labels = {}, [], []
    for pid in pids:
        t = np.arange(0, n_hours * H, 120.0)
        states = rng.choice([0.0, 3.0], n_hours)
        signal = rng.normal(size=(len(t), 3, 300)).astype(np.float32) * 0.1
        signal += states[(t // H).astype(int)][:, None, None] * 0.2
        windows[pid] = Windows(t, (t // H) != 4, signal)
        for h in range(n_hours):
            hours.append({"participant_id": pid, "hour_index": h})
            labels.append({"sleep_now": 0.0, "state_now": states[h], "tremor_now": float(states[h] == 0),
                           "sleep_next_hour": 0.0, "state_next_hour": states[h + 1] if h + 1 < n_hours else np.nan,
                           "tremor_next_hour": np.nan})
    events = pd.DataFrame([{"participant_id": pids[0], "time_s": 2 * H + 30.0, "kind": "dose", "ledd": 100.0,
                            "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan}])
    patients = pd.DataFrame({"participant_id": list(pids), "more_affected_side": [True] * len(pids),
                             "dbs": [False] * len(pids)})
    return Dataset(SPEC, pd.DataFrame(hours), pd.DataFrame(labels), windows=windows, events=events,
                   patients=patients)


def fitted(**kw):
    data = dataset()
    model = MotorCausalTransformer(["sleep", "state", "tremor"], params={**TINY, **kw}, spec=SPEC)
    return model.fit(data.subset(["A", "B"]), val=data.subset(["C"]), seed=0), data


def test_fit_runs_both_steps_and_records_them():
    model, _ = fitted()
    r = model.training_result
    assert len(r["history"]["pretrain"]) == 1 and 1 <= len(r["history"]["diary"]) <= 2
    assert 1 <= r["best_epoch"] <= 2
    assert model.manifest()["family"] == "motor_causal_transformer"


def test_predict_follows_the_contract_every_hour_and_every_worn_window():
    model, data = fitted()
    out = model.predict(data)                    # the base class checks hours, labels, sums, windows
    assert isinstance(out, Predictions) and len(out.hours) == 30 and set(out.labels) == set(model.labels)
    for pid in data.participants:
        w = data.windows[pid]
        assert np.array_equal(out.windows[pid].time_s, w.time_s[w.worn])
    # Hour 4 had no worn windows: the training class mix, the same for every participant.
    rows = out.hours.index[out.hours.hour_index == 4]
    p = out.probs["state_now"][rows]
    assert np.allclose(p, p[0]) and not np.allclose(out.probs["state_now"][rows - 1], p[0])


def test_save_and_load_give_the_same_predictions(tmp_path):
    model, data = fitted()
    model.save(tmp_path / "m")
    info = json.loads((tmp_path / "m" / "weights" / "training.json").read_text(encoding="utf-8"))
    assert info["harnet10"] == "fake-weights"
    loaded = MotorCausalTransformer.load(tmp_path / "m")
    a, b = model.predict(data), loaded.predict(data)
    for label in model.labels:
        np.testing.assert_allclose(a.probs[label], b.probs[label], rtol=1e-6, atol=1e-7)
    assert json.dumps(loaded.training_result) == json.dumps(model.training_result)     # NaN != NaN


def test_predict_refuses_other_harnet10_weights(tmp_path, monkeypatch):
    model, data = fitted()
    model.save(tmp_path / "m")
    monkeypatch.setattr(harnet_module.Harnet10, "load",
                        classmethod(lambda cls, hub, device="cpu", batch_size=2048: cls(Fake(), "other-weights")))
    with pytest.raises(ModelError, match="harnet10 weights"):
        MotorCausalTransformer.load(tmp_path / "m").predict(data)


def test_a_cached_pretraining_gives_exactly_the_same_model(tmp_path, monkeypatch):
    data = dataset()
    store = tmp_path / "pretrained_cache"

    def fit(**params):
        model = MotorCausalTransformer(["sleep", "state", "tremor"], params={**TINY, **params}, spec=SPEC)
        model.fit(data.subset(["A", "B"]), val=data.subset(["C"]), seed=0)
        return model, model.predict(data)

    first, a = fit()
    assert len(list(store.glob("*.pt"))) == 1 and len(list(store.glob("*.json"))) == 1
    second, b = fit()                                                  # a hit: nothing new is cached
    assert len(list(store.glob("*.pt"))) == 1
    for label in first.labels:
        np.testing.assert_array_equal(a.probs[label], b.probs[label])
    assert json.dumps(first.training_result) == json.dumps(second.training_result)
    monkeypatch.setenv("MODEL_PRETRAIN_CACHE", "off")                  # and both equal a fit without the cache
    _, c = fit()
    for label in first.labels:
        np.testing.assert_array_equal(a.probs[label], c.probs[label])
    monkeypatch.setenv("MODEL_PRETRAIN_CACHE", str(store))
    fit(diary={**TINY["diary"], "lr": 1e-3})                           # another diary step: still a hit
    assert len(list(store.glob("*.pt"))) == 1
    fit(pretrain={**TINY["pretrain"], "lr": 1e-3})                     # another pretraining: a new entry
    assert len(list(store.glob("*.pt"))) == 2


def test_a_damaged_cache_entry_means_pretraining_again(tmp_path):
    data = dataset()
    model = MotorCausalTransformer(["state"], params=TINY, spec=SPEC).fit(data.subset(["A", "B"]), seed=0)
    [entry] = (tmp_path / "pretrained_cache").glob("*.pt")
    entry.write_bytes(b"not a checkpoint")
    again = MotorCausalTransformer(["state"], params=TINY, spec=SPEC).fit(data.subset(["A", "B"]), seed=0)
    np.testing.assert_array_equal(model.predict(data).probs["state_now"], again.predict(data).probs["state_now"])


def test_without_validation_it_trains_final_epochs():
    data = dataset()
    model = MotorCausalTransformer(["state"], ["now"], params={**TINY, "final_epochs": 3, "pretrain": {"epochs": 0}},
                                   spec=SPEC).fit(data, seed=0)
    r = model.training_result
    assert r["history"]["pretrain"] == [] and len(r["history"]["diary"]) == 3 and r["best_epoch"] == 3
    out = model.predict(data)
    assert out.labels == ("state_now",)


def test_params_blocks_merge_into_their_defaults_and_bad_ones_are_refused():
    m = MotorCausalTransformer(["state"], params={"diary": {"lr": 1e-3}}, spec=SPEC)
    assert m.params["diary"]["lr"] == 1e-3 and m.params["diary"]["max_epochs"] == 15
    assert m.params["pretrain"]["epochs"] == 20 and m.params["score_past_h"] == 8
    for params, match in (({"diary": {"lr2": 1}}, "diary settings"), ({"network": {"width": 3}}, "unexpected"),
                          ({"score_past_h": -1}, "score_past_h"), ({"final_epochs": 0}, "final_epochs"),
                          ({"network": {"heads": {"horizons": ["now"]}}}, "horizons"), ({"nope": 1}, "no setting")):
        with pytest.raises(ConfigError, match=match):
            MotorCausalTransformer(["state"], params=params, spec=SPEC)
    with pytest.raises(ConfigError, match="window spec"):
        MotorCausalTransformer(["state"])


def test_sample_weights_reweight_the_diary_step_only():
    data = dataset()
    train = data.subset(["A", "B"])

    def fit(weights):
        model = MotorCausalTransformer(["state"], params=TINY, spec=SPEC)
        model.fit(train, val=data.subset(["C"]), sample_weight=weights, seed=0)
        return model

    plain, ones = fit(None), fit(np.ones(len(train.hours)))
    np.testing.assert_array_equal(plain.predict(data).probs["state_now"], ones.predict(data).probs["state_now"])
    weights = np.where(train.hours.participant_id.to_numpy() == "B", 0.0, 1.0)
    reweighted = fit(weights)
    # The same pretrained body (pretraining reads the movement, not the weights), another diary step.
    pretraining = [json.dumps(m.training_result["history"]["pretrain"]) for m in (reweighted, plain)]
    assert pretraining[0] == pretraining[1]
    assert not np.array_equal(reweighted.predict(data).probs["state_now"], plain.predict(data).probs["state_now"])
    assert reweighted._prior != plain._prior                          # the class mix of A's hours only


def test_refuses_a_spec_harnet10_cant_read():
    data = dataset()
    other = WindowSpec(version=1, seconds=5, rate_hz=30, axes=3)
    with pytest.raises(ConfigError, match="harnet10 reads"):
        MotorCausalTransformer(["state"], params=TINY, spec=other)._fit(data, None, None, 0)


def test_a_cache_that_cant_be_written_costs_the_speed_up_not_the_fit(tmp_path, monkeypatch):
    from scripts.models.motor_causal_transformer import pretrained_cache

    def full_disk(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(pretrained_cache.torch, "save", full_disk)
    model = MotorCausalTransformer(["state"], params=TINY, spec=SPEC).fit(dataset().subset(["A", "B"]), seed=0)
    assert model.is_fitted and not list((tmp_path / "pretrained_cache").glob("*"))
