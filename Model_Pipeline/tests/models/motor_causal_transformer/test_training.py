import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.core.data import Dataset, Windows, WindowSpec
from scripts.core.errors import ConfigError
from scripts.models.motor_causal_transformer.data import prepare
from scripts.models.motor_causal_transformer.network import MotorNetwork
from scripts.models.motor_causal_transformer.pretraining import MovementForecast
from scripts.models.motor_causal_transformer.training import (
    DiarySettings, PretrainSettings, class_counts, prefetch, pretrain, score_hours, train_diary, within_scores)

H = 3600
LABELS = ("sleep_now", "state_now", "tremor_now", "sleep_next_hour", "state_next_hour", "tremor_next_hour")
SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)
SMALL = {"movement_params": {"hidden_dim": 16, "output_dim": 16}, "d_model": 16, "dropout": 0.1,
         "decoder": {"heads": 2, "query_block": 32}, "heads": {"hidden": 8}}


class Cache:
    """Features that carry the hour's state: the first 256 shift with the state class, plus noise.

    Spread over many features, so it survives the random frozen projection (no pretraining here)."""

    def __init__(self, states):
        self.states = states

    def get(self, pid, windows):
        t = windows.time_s[windows.worn]
        rng = np.random.default_rng(abs(hash(pid)) % 2**32)
        x = rng.normal(size=(len(t), 1024)).astype(np.float32)
        x[:, :256] += 0.5 * np.array([self.states[pid].get(int(h), 3) for h in t // H], dtype=np.float32)[:, None]
        return x


def make(pids=("A", "B", "C", "D"), n_hours=20, seed=0):
    """Windows every 2 min for n_hours; every hour answered: OFF (0) or good (3) at random, awake."""
    rng = np.random.default_rng(seed)
    windows, hours, labels, states = {}, [], [], {}
    for pid in pids:
        t = np.arange(0, n_hours * H, 120.0)
        windows[pid] = Windows(t, np.ones(len(t), bool), np.zeros((len(t), 3, 300), np.float32))
        states[pid] = {}
        for h in range(n_hours):
            s = float(rng.choice([0, 3]))
            states[pid][h] = s
            hours.append({"participant_id": pid, "hour_index": h})
            labels.append({"sleep_now": 0.0, "state_now": s, "tremor_now": float(s == 0), "sleep_next_hour": np.nan,
                           "state_next_hour": np.nan, "tremor_next_hour": np.nan})
    patients = pd.DataFrame({"participant_id": list(pids), "more_affected_side": [True] * len(pids),
                             "dbs": [False] * len(pids)})
    events = pd.DataFrame(columns=["participant_id", "time_s", "kind", "ledd", "source", "kinesia_score",
                                   "tremor_score"])
    data = Dataset(SPEC, pd.DataFrame(hours), pd.DataFrame(labels), windows=windows, events=events,
                   patients=patients)
    return prepare(data, Cache(states), LABELS, window_s=10, dose_span_s=0.0)


def network():
    torch.manual_seed(0)
    return MotorNetwork(**SMALL)


def test_settings_default_to_the_final_recipe_and_refuse_bad_values():
    d = DiarySettings()
    assert (d.max_epochs, d.patience, d.chunk_h, d.past_h, d.lr, d.weight_decay, d.feature_noise) == \
        (15, 3, 8, 1, 3e-4, 0.1, 0.1)
    assert d.learn == ("heads", "pooling") and d.eval_past_h == (1, 2, 4, 8) and d.check_ins_per_day == 4.0
    p = PretrainSettings()
    assert (p.epochs, p.weight_decay, p.check_ins_per_day) == (20, 0.01, 4.0)
    for kw, match in (({"batch": 0}, "batch"), ({"lr": 0.0}, "lr"), ({"feature_noise": -1}, "feature_noise"),
                      ({"learn": ("heads", "attention")}, "learn"), ({"eval_past_h": ()}, "eval_past_h"),
                      ({"patience": 1.5}, "patience")):
        with pytest.raises(ConfigError, match=match):
            DiarySettings(**kw)


def test_prefetch_keeps_order_and_raises_the_threads_error():
    assert list(prefetch(iter(range(50)))) == list(range(50))

    def broken():
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(prefetch(broken()))
    it = prefetch(iter(range(10**6)))
    assert next(it) == 0
    it.close()                                                   # stopping early stops the thread


def test_within_scores_are_per_patient_and_averaged_over_past_lengths():
    rows = []
    for past in (1, 2):
        for pid in ("A", "B"):
            for h, y in enumerate([0, 0, 3, 3, 1, 4]):
                good = past == 1                                     # perfect with 1 h, backwards with 2 h
                p_off = (y <= 2) if good else (y > 2)
                state_p = np.full(7, 0.01)
                state_p[0 if p_off else 3] = 1 - 0.06
                rows.append({"participant_id": pid, "hour_index": h, "past_h": past, "state_now_y": float(y),
                             "state_now_p": state_p, "tremor_now_y": float(h % 2),
                             "tremor_now_p": np.array([0.2, 0.4, 0.4]) if h % 2 else np.array([0.8, 0.1, 0.1])})
    s = within_scores(pd.DataFrame(rows))
    assert s["off_auc_within"] == pytest.approx(0.5)                # 1.0 and 0.0, averaged over the pasts
    assert s["tremor_any_auc_within"] == pytest.approx(1.0)
    assert math.isfinite(s["within_auc"])
    assert math.isnan(within_scores(pd.DataFrame())["within_auc"])


def test_class_counts_count_now_answers_plus_one():
    counts = class_counts(make(pids=("A",), n_hours=10))
    assert counts["sleep"].tolist() == [11, 1] and counts["state"].sum() == 10 + 7
    assert counts["state"][[1, 2, 4, 5, 6]].tolist() == [1] * 5


def test_pretraining_trains_the_body_only_and_logs_every_target():
    recs = make(pids=("A", "B"), n_hours=12)
    net = network()
    net.movement.fit(r.features for r in recs.values())
    forecast = MovementForecast(d_model=16, feature_dim=1024, averages=((600, 1800),), min_windows=5)
    heads_before = {k: v.clone() for k, v in net.heads.state_dict().items()}
    decoder_before = {k: v.clone() for k, v in net.decoder.state_dict().items()}
    history = pretrain(net, forecast, recs, PretrainSettings(epochs=2, chunk_h=4), "cpu", np.random.default_rng(0))
    assert len(history) == 2 and all(math.isfinite(v) for line in history for k, v in line.items()
                                     if k.endswith("/loss"))
    assert {"ahead_20s/loss", "ahead_120s/bar", "average_10_30min/loss"} <= set(history[0])
    assert all(torch.equal(v, heads_before[k]) for k, v in net.heads.state_dict().items())
    assert any(not torch.equal(v, decoder_before[k]) for k, v in net.decoder.state_dict().items())


def test_pretraining_measures_validation_skill_without_changing_the_training():
    recs = make(pids=("A", "B"), n_hours=12)
    val = make(pids=("C",), n_hours=12)

    def run(with_val):
        torch.manual_seed(0)
        net = network()
        net.movement.fit(r.features for r in recs.values())
        forecast = MovementForecast(d_model=16, feature_dim=1024, averages=((600, 1800),), min_windows=5)
        history = pretrain(net, forecast, recs, PretrainSettings(epochs=2, chunk_h=4), "cpu",
                           np.random.default_rng(0), val=val if with_val else None)
        return history, net

    (with_val, net_a), (without, net_b) = run(True), run(False)
    for a, b in zip(with_val, without):                  # the same training, to the last digit (NaN as text)
        assert json.dumps({k: v for k, v in a.items() if not k.startswith("val/") and k != "seconds"}) == \
               json.dumps({k: v for k, v in b.items() if k != "seconds"})
    assert all(torch.equal(x, y) for x, y in zip(net_a.state_dict().values(), net_b.state_dict().values()))
    line = with_val[-1]
    for name in ("ahead_120s", "ahead_600s", "average_10_30min"):     # windows every 2 min: no 20-s target
        assert line[f"val/{name}/skill"] == pytest.approx(1 - line[f"val/{name}/loss"] / line[f"val/{name}/bar"])
        assert line[f"{name}/skill"] == pytest.approx(1 - line[f"{name}/loss"] / line[f"{name}/bar"])
    assert line["seconds"] > 0 and not any(k.startswith("val/") for k in without[-1])


def test_the_diary_step_learns_the_heads_keeps_the_body_and_the_best_epoch():
    recs = make()
    train, val = {k: recs[k] for k in ("A", "B", "C")}, {"D": recs["D"]}
    net = network()
    net.movement.fit(r.features for r in train.values())
    body = {k: v.clone() for k, v in net.state_dict().items() if k.startswith(("movement.", "tokens.", "decoder."))}
    settings = DiarySettings(max_epochs=6, patience=2, chunk_h=4, eval_past_h=(1, 2), lr=3e-3)
    result = train_diary(net, train, val, settings, "cpu", np.random.default_rng(0))
    assert 1 <= result.best_epoch <= len(result.history) <= 6
    assert result.best_score == max(line["within_auc"] for line in result.history)
    assert result.best_score > 0.8                                   # the features give OFF away
    state = net.state_dict()
    assert all(torch.equal(state[k], v) for k, v in body.items())    # frozen: exactly as before
    assert not net.decoder.training or not any(m.training for m in net.decoder.modules())


def test_without_validation_it_trains_the_given_epochs_and_is_reproducible():
    recs = make(pids=("A", "B"), n_hours=10)

    def run():
        net = network()
        net.movement.fit(r.features for r in recs.values())
        torch.manual_seed(1)
        result = train_diary(net, recs, None, DiarySettings(chunk_h=4, eval_past_h=(1,)), "cpu",
                             np.random.default_rng(0), epochs=2)
        return result, net

    (a, net_a), (b, net_b) = run(), run()
    assert a.best_epoch == 2 and len(a.history) == 2 and math.isnan(a.best_score)
    assert a.history == b.history
    assert all(torch.equal(x, y) for x, y in zip(net_a.state_dict().values(), net_b.state_dict().values()))


def test_score_hours_gives_each_answered_hour_at_each_past_length():
    recs = make(pids=("A",), n_hours=6)
    net = network()
    net.movement.fit(r.features for r in recs.values())
    rows = score_hours(net, recs, (0, 2), "cpu", batch_size=4)
    assert len(rows) == 12 and sorted(rows.past_h.unique()) == [0, 2]
    assert rows.state_now_p.iloc[0].shape == (7,) and np.isclose(rows.state_now_p.iloc[0].sum(), 1)
    assert (rows.state_now_y.to_numpy() == [recs["A"].answer("state_now", h) for h in rows.hour_index]).all()
