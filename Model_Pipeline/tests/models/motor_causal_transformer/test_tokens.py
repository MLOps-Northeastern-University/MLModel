import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.core.errors import ConfigError, ModelError, SchemaError
from scripts.models.motor_causal_transformer.tokens import (
    EVENT_FIELDS, PATIENT_FIELDS, TokenBuilder, event_fields, mark_check_ins, patient_fields)

F = {name: i for i, name in enumerate(EVENT_FIELDS)}


def table(*rows):
    """Events from (time_s, kind, ledd, source, kinesia, tremor) rows."""
    return pd.DataFrame(rows, columns=["time_s", "kind", "ledd", "source", "kinesia_score", "tremor_score"])


def dose(time_s, ledd=100.0, source="on_time"):
    return (time_s, "dose", ledd, source, np.nan, np.nan)


def check_in(time_s, kinesia=np.nan, tremor=np.nan):
    return (time_s, "check_in", np.nan, None, kinesia, tremor)


T = np.arange(0, 600, 10.0)   # 60 worn windows, 10 s each


def test_nothing_happened_is_exactly_zero():
    out = event_fields(T, table(), window_s=10)
    assert out.fields.shape == (60, len(EVENT_FIELDS)) and out.fields.dtype == np.float32
    assert not out.fields.any() and out.dropped == 0


def test_a_dose_marks_its_own_window_without_a_span():
    out = event_fields(T, table(dose(125.0, ledd=150.0, source="late")), window_s=10)
    marked = np.flatnonzero(out.fields[:, F["dose"]])
    assert marked.tolist() == [12]
    assert out.fields[12, F["ledd"]] == pytest.approx(1.5)
    assert out.fields[12, F["late"]] == 1 and out.fields[12, F["on_time"]] == 0


def test_a_dose_span_marks_from_the_dose_on_and_never_before():
    out = event_fields(T, table(dose(125.0)), window_s=10, dose_span_s=60)
    marked = np.flatnonzero(out.fields[:, F["dose"]])
    # Its own window (120) and every window starting before 125 + 60 = 185: 130 .. 180.
    assert marked.tolist() == list(range(12, 19))
    assert not out.fields[:12].any() and not out.fields[19:].any()


def test_overlapping_doses_add_up_and_keep_the_least_trustworthy_source():
    out = event_fields(T, table(dose(100.0, 100.0, "on_time"), dose(150.0, 50.0, "none_logged")),
                       window_s=10, dose_span_s=100)
    f = out.fields
    assert f[12, F["ledd"]] == pytest.approx(1.0) and f[12, F["on_time"]] == 1     # first dose only
    assert f[17, F["ledd"]] == pytest.approx(1.5) and f[17, F["none_logged"]] == 1  # both
    assert f[17, F["on_time"]] == 0
    assert f[22, F["ledd"]] == pytest.approx(0.5) and f[22, F["none_logged"]] == 1  # second only
    assert (f[:, F["on_time"]] + f[:, F["late"]] + f[:, F["none_logged"]] <= 1).all()   # one-hot


def test_an_event_while_not_worn_goes_on_the_next_worn_window():
    t = np.array([0.0, 10.0, 20.0, 1000.0, 1010.0])           # not worn from 30 s to 1000 s
    out = event_fields(t, table(dose(500.0), check_in(600.0, kinesia=-2)), window_s=10)
    assert out.fields[3, F["dose"]] == 1 and out.fields[3, F["check_in"]] == 1
    assert not out.fields[:3].any()


def test_events_after_the_last_window_are_counted_not_placed():
    out = event_fields(T, table(dose(595.0), dose(700.0), check_in(9_999.0, tremor=1)), window_s=10)
    assert out.dropped == 2 and out.fields[59, F["dose"]] == 1


def test_check_in_answers_are_scaled_and_a_zero_answer_still_shows():
    out = event_fields(T, table(check_in(55.0, kinesia=-3, tremor=2), check_in(305.0, kinesia=0, tremor=0),
                                check_in(405.0, kinesia=1)), window_s=10)
    f = out.fields
    assert (f[5, F["kinesia"]], f[5, F["tremor"]]) == (-1.0, 1.0)
    assert f[30, F["check_in"]] == 1 and f[30, F["kinesia"]] == 0 and f[30, F["tremor"]] == 0
    assert f[40, F["kinesia"]] == pytest.approx(1 / 3) and f[40, F["tremor"]] == 0     # tremor not answered
    assert not f[:, F["dose"]].any()


def test_events_out_of_order_are_placed_the_same():
    rows = [dose(100.0), check_in(105.0, kinesia=2), dose(300.0, 80.0, "late")]
    a = event_fields(T, table(*rows), window_s=10, dose_span_s=50)
    b = event_fields(T, table(*rows[::-1]), window_s=10, dose_span_s=50)
    assert np.array_equal(a.fields, b.fields)


def test_no_tokens_means_every_event_is_dropped():
    out = event_fields(np.array([]), table(dose(10.0)), window_s=10)
    assert out.fields.shape == (0, len(EVENT_FIELDS)) and out.dropped == 1


def test_refuses_bad_times_and_settings():
    with pytest.raises(ModelError, match="increasing"):
        event_fields(np.array([0.0, 20.0, 10.0]), table(), window_s=10)
    with pytest.raises(ModelError, match="increasing"):
        event_fields(np.zeros((2, 2)), table(), window_s=10)
    for kw, match in (({"window_s": 0}, "window_s"), ({"window_s": 10, "dose_span_s": -1}, "dose_span_s"),
                      ({"window_s": 10, "ledd_scale": 0}, "ledd_scale"),
                      ({"window_s": 10, "dose_span_s": math.inf}, "dose_span_s")):
        with pytest.raises(ModelError, match=match):
            event_fields(T, table(), **kw)


def test_a_chunk_sliced_from_the_recording_gets_what_it_would_have_live():
    """A chunk is rows of the whole recording's fields, so doses from before its start behave as live."""
    hours = 3600.0
    # Worn 0-6 h and 7-10 h; a chunk starts at 8 h (token `c`).
    t = np.concatenate([np.arange(0, 6 * hours, 10.0), np.arange(7 * hours, 10 * hours, 10.0)])
    c = int(np.searchsorted(t, 8 * hours))
    events = table(dose(8 * hours - 600),          # 10 min before the chunk: 20 min of its span left
                   dose(3 * hours),                # 5 h before: long over
                   dose(6.5 * hours, 50.0))        # while the watch was off: goes on 7 h, the next worn window
    chunk = event_fields(t, events, window_s=10, dose_span_s=1800).fields[c:]
    assert np.flatnonzero(chunk[:, F["dose"]]).tolist() == list(range(120))     # 8:00 to 8:20, nothing else
    # Working on the chunk alone gets both wrong: an old dose lands on its first window...
    alone = event_fields(t[c:], events, window_s=10, dose_span_s=1800).fields
    assert alone[0, F["ledd"]] == pytest.approx(2.5)
    # ...and leaving out events from before it loses the dose that is still marked.
    later = events[events.time_s >= 8 * hours]
    assert not event_fields(t[c:], later, window_s=10, dose_span_s=1800).fields.any()
    whole = event_fields(t, events, window_s=10, dose_span_s=1800).fields
    assert whole[int(np.searchsorted(t, 7 * hours)), F["ledd"]] == pytest.approx(0.5)


def test_mark_check_ins_adds_to_a_copy():
    recording = event_fields(T, table(dose(100.0)), window_s=10).fields
    chunk = recording[5:25]                                          # a view, as a sliced chunk is
    out = mark_check_ins(chunk, [0, 7, 7], [-2, 1, np.nan], [np.nan, 0, 2])
    assert not recording[:, F["check_in"]].any()                    # the recording is untouched
    assert out[0, F["check_in"]] == 1 and out[0, F["kinesia"]] == pytest.approx(-2 / 3)
    assert out[7, F["kinesia"]] == pytest.approx(1 / 3) and out[7, F["tremor"]] == 1.0   # later answer wins
    assert out[5, F["dose"]] == 1                                    # the dose is kept
    assert np.array_equal(mark_check_ins(chunk, [], [], []), chunk)


@pytest.mark.parametrize("at, kinesia, tremor, match", [
    ([20], [1], [0], "tokens 0 to 19"), ([-1], [1], [0], "tokens 0 to 19"), ([1], [4], [0], "kinesia"),
    ([1], [0.5], [0], "kinesia"), ([1], [0], [3], "tremor"), ([1], [np.nan], [np.nan], "at least one"),
    ([1, 2], [0], [0], "one token"),
])
def test_mark_check_ins_refuses_bad_check_ins(at, kinesia, tremor, match):
    with pytest.raises(ModelError, match=match):
        mark_check_ins(np.zeros((20, len(EVENT_FIELDS)), dtype=np.float32), at, kinesia, tremor)


def test_refuses_unknown_kinds_and_sources():
    with pytest.raises(SchemaError, match="events.kind"):
        event_fields(T, table((50.0, "nap", np.nan, None, np.nan, np.nan)), window_s=10)
    with pytest.raises(SchemaError, match="events.source"):
        event_fields(T, table(dose(50.0, source="guess")), window_s=10)


def test_patient_fields():
    assert patient_fields(True, False).tolist() == [1.0, 0.0]
    assert patient_fields(np.bool_(False), np.bool_(True)).tolist() == [0.0, 1.0]
    for bad in (np.nan, None, 1, "yes"):
        with pytest.raises(SchemaError, match="True or False"):
            patient_fields(bad, False)
    assert PATIENT_FIELDS == ("more_affected_side", "dbs")


def builder(**kw):
    torch.manual_seed(0)
    return TokenBuilder(**{"movement_dim": 32, "d_model": 16, "dropout": 0.0, **kw}).eval()


def test_token_shape_and_patient_broadcast_equals_patient_per_token():
    tb = builder()
    movement, events = torch.randn(2, 10, 32), torch.zeros(2, 10, 8)
    patient = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    once = tb(movement, patient, events)
    assert once.shape == (2, 10, 16)
    assert torch.equal(once, tb(movement, patient.expand(2, 10, 2).contiguous(), events))


def test_no_event_adds_exactly_nothing_and_a_dose_is_as_visible_as_the_other_parts():
    tb = TokenBuilder(movement_dim=256, d_model=64, dropout=0.0).eval()
    assert tb.events.bias is None
    assert torch.equal(tb.events(torch.zeros(3, 8)), torch.zeros(3, 64))
    torch.manual_seed(1)
    lengths = []
    for _ in range(200):
        fresh = TokenBuilder(movement_dim=256, d_model=64, dropout=0.0)
        a_dose = torch.zeros(8)
        a_dose[[F["dose"], F["ledd"], F["on_time"]]] = 1.0
        lengths.append(fresh.events(a_dose).norm().item())
    # A LayerNorm'd part has length sqrt(64) = 8; the dose part should start at about that, not 3x less.
    assert 6.5 < np.mean(lengths) < 9.5


def test_dropout_only_while_training():
    tb = builder(dropout=0.3)
    args = (torch.randn(1, 10, 32), torch.ones(1, 1, 2), torch.zeros(1, 10, 8))
    assert torch.equal(tb(*args), tb(*args))
    tb.train()
    assert not torch.allclose(tb(*args), tb(*args))


def test_refuses_mismatched_parts():
    tb = builder()
    movement, patient, events = torch.randn(2, 10, 32), torch.ones(2, 1, 2), torch.zeros(2, 10, 8)
    with pytest.raises(ModelError, match="movement"):
        tb(torch.randn(2, 10, 31), patient, events)
    with pytest.raises(ModelError, match="events"):
        tb(movement, patient, torch.zeros(2, 9, 8))
    with pytest.raises(ModelError, match="patient"):
        tb(movement, torch.ones(2, 5, 2), events)
    with pytest.raises(ModelError, match="patient"):
        tb(movement, torch.ones(1, 1, 2), events)


@pytest.mark.parametrize("kw, match", [({"movement_dim": 0}, "movement_dim"), ({"d_model": 8.0}, "d_model"),
                                       ({"dropout": 1}, "dropout")])
def test_refuses_bad_settings(kw, match):
    with pytest.raises(ConfigError, match=match):
        TokenBuilder(**kw)
