import numpy as np
import pandas as pd
import pytest

from model_pipeline.core.data import Dataset, Windows, WindowSpec
from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer.data import (
    answered_hours, batches, collate, make_example, prepare, tile_chunks)
from model_pipeline.models.motor_causal_transformer.tokens import EVENT_FIELDS

F = {name: i for i, name in enumerate(EVENT_FIELDS)}
LABELS = ("sleep_now", "state_now", "tremor_now", "sleep_next_hour", "state_next_hour", "tremor_next_hour")
SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)
H = 3600


class Cache:
    """Stands in for harnet.FeatureCache: each worn window's features are its time, repeated."""

    def get(self, participant, windows):
        t = np.asarray(windows.time_s, dtype=np.float32)[windows.worn]
        return np.repeat(t[:, None], 4, axis=1)


def dataset():
    """P1: 12 h of windows from hour 2, unworn 5:00-6:00, answers in hours 3-9 (6 asleep). P2: 3 h."""
    windows, hours, labels = {}, [], []
    for pid, (first, n_hours) in {"P1": (2, 12), "P2": (0, 3)}.items():
        t = np.arange(first * H, (first + n_hours) * H, 10.0)
        worn = ~((t >= 5 * H) & (t < 6 * H)) if pid == "P1" else np.ones(len(t), bool)
        windows[pid] = Windows(t, worn, np.zeros((len(t), 3, 300), np.float32))
        for h in range(first, first + n_hours):
            hours.append({"participant_id": pid, "hour_index": h})
            answered = pid == "P1" and 3 <= h <= 9
            asleep = h == 6
            labels.append({"sleep_now": float(asleep) if answered else np.nan,
                           "state_now": float(h % 7) if answered and not asleep else np.nan,
                           "tremor_now": float(h % 3) if answered and not asleep else np.nan,
                           "sleep_next_hour": np.nan, "state_next_hour": np.nan, "tremor_next_hour": np.nan})
    events = pd.DataFrame([{"participant_id": "P1", "time_s": 4 * H + 5.0, "kind": "dose", "ledd": 100.0,
                            "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan}])
    patients = pd.DataFrame({"participant_id": ["P1", "P2"], "more_affected_side": [True, False],
                             "dbs": [False, True]})
    return Dataset(SPEC, pd.DataFrame(hours), pd.DataFrame(labels), windows=windows, events=events,
                   patients=patients)


def recordings(dose_span_s=0.0):
    return prepare(dataset(), Cache(), LABELS, window_s=10, dose_span_s=dose_span_s)


def test_prepare_keeps_worn_windows_answers_and_whole_recording_events():
    rec = recordings(dose_span_s=1800)["P1"]
    assert len(rec.time_s) == 11 * 360 and not ((rec.time_s >= 5 * H) & (rec.time_s < 6 * H)).any()
    assert np.array_equal(rec.features[:, 0], rec.time_s.astype(np.float32))
    assert answered_hours(rec) == [3, 4, 5, 6, 7, 8, 9]
    assert rec.answer("state_now", 6) != rec.answer("state_now", 6)          # asleep: no waking state (NaN)
    assert rec.answer("sleep_now", 6) == 1.0
    dosed = np.flatnonzero(rec.events[:, F["dose"]])
    assert rec.time_s[dosed[0]] == 4 * H and len(dosed) == 181       # its own window, then 30 min of them
    assert rec.patient.tolist() == [1.0, -1.0]
    assert recordings()["P2"].answers == {}


def test_an_example_is_its_hours_with_past_in_front():
    rec = recordings()["P1"]
    e = make_example(rec, start_hour=4, hours=3, past_h=1)
    assert e.time_s[0] == 3 * H and e.time_s[-1] == 7 * H - 10 and e.n_input == len(e.time_s)
    assert e.hours.tolist() == [4, 5, 6]
    assert e.targets[:, LABELS.index("state_now")][0] == 4.0 and np.isnan(e.targets[2, 1])   # hour 6 asleep
    assert e.has_graded_windows and e.check_ins == 0
    a = int(np.searchsorted(rec.time_s, 3 * H))
    np.testing.assert_array_equal(e.events, rec.events[a:a + e.n_input])


def test_future_runs_on_but_the_model_reads_only_the_chunk():
    rec = recordings()["P1"]
    e = make_example(rec, start_hour=4, hours=2, past_h=1, future_h=3)
    assert e.time_s[e.n_input - 1] == 5 * H - 10 and e.time_s[-1] == 9 * H - 10     # 5:00-6:00 unworn
    assert len(e.events) == e.n_input
    c = collate([e], LABELS, future=True)
    assert c.batch.movement.shape[1] == e.n_input and c.future[0].shape[1] == len(e.time_s)
    np.testing.assert_array_equal(c.future[1][0, :e.n_input].numpy(), c.batch.time_s[0].numpy())


def test_check_ins_go_on_answered_waking_hours_past_included_with_that_hours_answer():
    rec = recordings()["P1"]
    rng = np.random.default_rng(0)
    seen_hours, seen_past = set(), False
    for _ in range(200):
        e = make_example(rec, start_hour=5, hours=4, past_h=2, check_ins_per_day=16, rng=rng)   # every hour
        at = np.flatnonzero(e.events[:, F["check_in"]])
        hours = (e.time_s[at] // H).astype(int)
        assert len(at) == e.check_ins
        for i, h in zip(at, hours):
            assert h in (3, 4, 7, 8)                                   # answered and awake; 5 unworn, 6 asleep
            assert e.events[i, F["kinesia"]] == pytest.approx((rec.answer("state_now", h) - 3) / 3)
            assert e.events[i, F["tremor"]] == pytest.approx(rec.answer("tremor_now", h) / 2)
        seen_hours |= set(hours.tolist())
        seen_past |= bool((hours < 5).any())
    assert seen_hours == {3, 4, 7, 8} and seen_past
    assert not rec.events[:, F["check_in"]].any()                    # the recording is untouched
    rate = np.mean([make_example(rec, 3, 7, 0, check_ins_per_day=4, rng=rng).check_ins for _ in range(2000)])
    assert rate == pytest.approx(5 * 4 / 16, rel=0.1)    # hours 3, 4, 7, 8, 9 (5 unworn, 6 asleep) x 4/16


def test_tiling_covers_the_recording_from_a_random_offset():
    recs = recordings()
    starts = set()
    for seed in range(30):
        chunks = tile_chunks({"P1": recs["P1"]}, 4, np.random.default_rng(seed), answered_only=False)
        s = [start for _, start in chunks]
        assert all(b - a == 4 for a, b in zip(s, s[1:])) and s[0] <= 2 and s[-1] + 4 > 13
        starts.add(s[0])
    assert starts == {-1, 0, 1, 2}                                     # the offset moves the boundaries
    answered = tile_chunks(recs, 4, np.random.default_rng(0))
    assert all(pid == "P1" and any(h in recs[pid].answers for h in range(s, s + 4)) for pid, s in answered)
    assert any(pid == "P2" for pid, _ in tile_chunks(recs, 4, np.random.default_rng(0), answered_only=False))


def test_batches_shuffle_and_cover_every_chunk():
    chunks = [("P1", i) for i in range(10)]
    out = batches(chunks, 4, np.random.default_rng(0))
    assert [len(b) for b in out] == [4, 4, 2] and sorted(c for b in out for c in b) == chunks


def test_collate_pads_after_the_real_windows_and_marks_graded_ones():
    recs = recordings()
    long, short = make_example(recs["P1"], 4, 2, 1), make_example(recs["P2"], 1, 2, 1)
    c = collate([long, short], LABELS)
    b = c.batch
    assert long.n_input < short.n_input                               # P1's hour 5 wasn't worn
    assert b.movement.shape == (2, short.n_input, 4) and b.valid[0].sum() == long.n_input
    assert not b.valid[0, long.n_input:].any() and (b.hour_of_window[0, long.n_input:] == -1).all()
    assert b.patient.shape == (2, 1, 2) and b.hours.tolist() == [[4, 5], [1, 2]]
    assert c.graded[0].sum() == 360 and not c.graded[0, :360].any()   # hour 4 graded (5 unworn), not hour 3
    assert b.targets["state_now"][0].tolist()[0] == 4.0 and np.isnan(b.targets["state_now"][1]).all()


def test_refuses_bad_settings_and_batches():
    rec = recordings()["P1"]
    for kw, match in (({"hours": 0}, "hours"), ({"hours": 2, "past_h": -1}, "past_h"),
                      ({"hours": 2, "check_ins_per_day": 1}, "rng")):
        with pytest.raises(ConfigError, match=match):
            make_example(rec, 4, **{"past_h": 1, **kw})
    with pytest.raises(ModelError, match="same number of graded hours"):
        collate([make_example(rec, 4, 2, 1), make_example(rec, 4, 3, 1)], LABELS)
    with pytest.raises(ConfigError, match="batch_size"):
        batches([], 0, np.random.default_rng(0))
    with pytest.raises(ConfigError, match="chunk_h"):
        tile_chunks({}, 0, np.random.default_rng(0))
