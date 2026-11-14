import math

import pytest
import torch

from scripts.core.errors import ConfigError, ModelError
from scripts.models.motor_causal_transformer.movement import build_movement_encoder
from scripts.models.motor_causal_transformer.pretraining import (
    Future, MovementForecast, harnet_targets, parts_as_floats, window_ahead, window_averages)

F = 6


def times(n, step=10.0, gaps=(), start=7 * 3600.0):
    """n window times, step apart, with extra seconds added before the indices in gaps."""
    steps = torch.full((n,), step, dtype=torch.float64)
    for i, extra in gaps:
        steps[i] += extra
    return (start + torch.cumsum(steps, 0) - step)[None]


def test_window_ahead_finds_the_first_window_in_its_span():
    t = times(10, gaps=[(5, 100.0)])                    # 0, 10, 20, 30, 40, 150, 160, ...
    valid = torch.ones(1, 10, dtype=torch.bool)
    j, has = window_ahead(t, valid, ahead_s=20, slack_s=5)
    assert j[0, :3].tolist() == [2, 3, 4] and has[0, :3].all()
    assert not has[0, 3] and not has[0, 4]                         # 50-55 s and 60-65 s: in the gap
    assert has[0, 5:8].all() and not has[0, 8:].any()              # the end: nothing 20 s later


def test_padding_gives_a_row_exactly_its_targets_alone():
    """The prototype's search lost targets on padded rows (time 0 padding isn't sorted)."""
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        n, pad = int(torch.randint(50, 400, (1,), generator=g)), int(torch.randint(1, 400, (1,), generator=g))
        t = times(n, gaps=[(int(torch.randint(1, n, (1,), generator=g)), 300.0)])
        alone = window_ahead(t, torch.ones(1, n, dtype=torch.bool), 120, 60)
        padded_t = torch.cat([t, torch.zeros(1, pad, dtype=t.dtype)], 1)
        padded_v = torch.cat([torch.ones(1, n, dtype=torch.bool), torch.zeros(1, pad, dtype=torch.bool)], 1)
        j, has = window_ahead(padded_t, padded_v, 120, 60)
        assert torch.equal(has[:, :n], alone[1]) and not has[:, n:].any()
        assert torch.equal(j[:, :n][alone[1]], alone[0][alone[1]])


def test_window_averages_match_a_plain_loop():
    g = torch.Generator().manual_seed(1)
    t = times(60, step=60.0, gaps=[(20, 1800.0)])
    valid = torch.ones(1, 60, dtype=torch.bool)
    valid[0, 50:] = False                                            # padding at the end
    x = torch.randn(1, 60, F, generator=g)
    spans = [(-600, 0), (300, 1200)]
    for (start, end), (mean, has) in zip(spans, window_averages(x, t, valid, spans, min_windows=5)):
        for i in range(50):
            inside = [k for k in range(50) if t[0, i] + start <= t[0, k] < t[0, i] + end]
            assert bool(has[0, i]) == (len(inside) >= 5)
            if inside:
                torch.testing.assert_close(mean[0, i], x[0, inside].mean(0))
        assert not has[0, 50:].any()


def forecast(**kw):
    torch.manual_seed(0)
    return MovementForecast(**{"d_model": 8, "feature_dim": F, "windows": ((20, 20), (120, 60)),
                               "averages": ((600, 1200),), "min_windows": 10, **kw})


def inputs(b=2, n=300, extra=200, seed=2):
    """b rows of n 10-s windows (row 1 shorter, padded), and `future` running on extra windows."""
    g = torch.Generator().manual_seed(seed)
    total = n + extra
    t_f = times(total).repeat(b, 1)
    v_f = torch.ones(b, total, dtype=torch.bool)
    v_f[1, total - 50:] = False
    x_f = torch.randn(b, total, F, generator=g)
    t, v, x = t_f[:, :n].clone(), torch.ones(b, n, dtype=torch.bool), x_f[:, :n].clone()
    v[1, n - 40:] = False      # row 1: 260 real windows, then padding; its future runs on from window 260
    h = torch.randn(b, n, 8, generator=g, requires_grad=True)
    return h, x, t, v, Future(x_f, t_f, v_f)


def test_loss_matches_the_targets_written_out():
    m = forecast()
    h, x, t, v, fut = inputs()
    out = m(h, x, t, v, future=fut)
    # The 20 s target, by hand: window i+2 (two windows on), for every window that has one.
    p = m.heads["ahead_20s"](h)
    has = torch.zeros_like(v)
    has[0, :298] = True
    has[1, :258] = True
    err = torch.stack([((p[b, i] - x[b, i + 2]) ** 2).mean() for b in range(2) for i in range(300) if has[b, i]])
    torch.testing.assert_close(out.parts["ahead_20s"].loss, err.mean())
    assert int(out.parts["ahead_20s"].windows) == int(has.sum())
    torch.testing.assert_close(out.total, torch.stack([q.loss for q in out.parts.values()]).mean())


def plain_loss(m, h, x, t, v, fut, graded=None):
    """The loss the obvious way: build every guess, then the mean squared error. The reference."""
    graded = v if graded is None else graded & v
    losses = []
    for name, (ahead, slack) in zip(m.names, m.windows):
        j, has = window_ahead(t, v, ahead, slack)
        target = torch.gather(x, 1, j.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        err = (m.heads[name](h) - target).square().mean(-1)
        losses.append(torch.where(has & graded, err, 0).sum() / (has & graded).sum().clamp_min(1))
    if m.averages:
        _, *averages = window_averages(fut.features, fut.time_s, fut.valid, ((-3600, 0), *m.averages),
                                       m.min_windows, queries=t.shape[1])
        for name, (target, has) in zip(m.names[len(m.windows):], averages):
            err = (m.heads[name](h) - target).square().mean(-1)
            losses.append(torch.where(has & graded, err, 0).sum() / (has & graded).sum().clamp_min(1))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_the_expanded_square_is_the_plain_squared_error_loss_and_gradients(dtype, tol):
    m = forecast().to(dtype)
    with torch.no_grad():                                                # biases and gains that aren't 0 and 1
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    h, x, t, v, fut = inputs()
    h = h.detach().to(dtype).requires_grad_(True)
    x, fut = x.to(dtype), Future(fut.features.to(dtype), fut.time_s, fut.valid)
    got = m(h, x, t, v, future=fut).total
    want = plain_loss(m, h, x, t, v, fut)
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)
    g_got = torch.autograd.grad(got, [h, *m.parameters()])
    g_want = torch.autograd.grad(want, [h, *m.parameters()])
    for a, b in zip(g_got, g_want):
        torch.testing.assert_close(a, b, rtol=tol * 10, atol=tol * 10)


def test_a_perfect_guess_has_zero_error_not_a_negative_one():
    torch.manual_seed(3)
    head = MovementForecast(d_model=8, feature_dim=F, windows=((10, 0),), averages=()).heads["ahead_10s"]
    z = head.norm(torch.randn(50, 8))
    y = head.linear(z).detach()                                           # exactly the guess
    err = head.squared_error(z, head.project(y), y.square().sum(-1))
    assert (err >= 0).all() and err.max() < 1e-5


def test_averages_read_the_future_but_only_grade_the_batchs_windows():
    m = forecast()
    h, x, t, v, fut = inputs()
    out = m(h, x, t, v, future=fut)
    out.total.backward()
    assert torch.all(h.grad[1, 260:] == 0)                              # padding is never graded
    # Late windows of row 0 have average targets thanks to the future: 10-20 min after window 290 is
    # windows 350-410, beyond the batch's 300.
    late = MovementForecast(d_model=8, feature_dim=F, windows=(), averages=((600, 1200),), min_windows=10)
    late_out = late(h, x, t, v, future=fut)
    assert int(late_out.parts["average_10_20min"].windows) > 0
    with pytest.raises(ModelError, match="future"):
        m(h, x, t, v)


def test_graded_leaves_windows_out_and_bars_are_reported():
    m = forecast()
    h, x, t, v, fut = inputs()
    graded = torch.zeros_like(v)
    graded[:, 100:] = True                                              # an ungraded past in front
    out = m(h, x, t, v, graded=graded, future=fut)
    out.total.backward()
    assert torch.all(h.grad[:, :100] == 0)
    floats = parts_as_floats(out.parts)
    assert set(floats) == {"ahead_20s", "ahead_120s", "average_10_20min"}
    assert all(math.isfinite(f["loss"]) and math.isfinite(f["bar"]) and f["windows"] > 0 for f in floats.values())


def test_no_window_with_a_target_gives_zero_not_nan():
    m = forecast()
    h, x, t, v, fut = inputs()
    out = m(h, x, t, v, graded=torch.zeros_like(v), future=fut)
    assert out.total.item() == 0.0
    out.total.backward()
    assert torch.isfinite(h.grad).all()
    assert all(math.isnan(f["bar"]) and f["windows"] == 0 for f in parts_as_floats(out.parts).values())


def test_heads_run_under_autocast_and_the_loss_in_float32():
    m = forecast()
    h, x, t, v, fut = inputs()
    want = m(h, x, t, v, future=fut).total
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = m(h, x, t, v, future=fut).total
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu():
    m = forecast()
    h, x, t, v, fut = inputs()
    want = m(h, x, t, v, future=fut).total
    got = m.cuda()(h.detach().cuda(), x.cuda(), t.cuda(), v.cuda(), future=Future(*(a.cuda() for a in fut))).total
    torch.testing.assert_close(got.cpu(), want)


def test_harnet_targets_use_the_encoders_standardisation():
    enc = build_movement_encoder("harnet10")
    enc.set_standardisation(torch.full((1024,), 2.0), torch.full((1024,), 4.0))
    x = torch.full((1, 3, 1024), 6.0)
    assert torch.allclose(harnet_targets(enc, x), torch.ones(1, 3, 1024))


def test_defaults_are_the_final_recipe():
    m = MovementForecast()
    assert m.names == ("ahead_20s", "ahead_120s", "ahead_600s", "average_30_90min", "average_90_150min")
    assert m.future_s == 9000 and m.min_windows == 180
    assert sum(p.numel() for p in m.parameters()) == 5 * (2 * 64 + 64 * 1024 + 1024)


def test_refuses_bad_settings_and_inputs():
    for kw, match in (({"windows": ((0, 10),)}, "valid span"), ({"averages": ((100, 50),)}, "valid span"),
                      ({"windows": (("a", 1),)}, "pairs of seconds"), ({"windows": (), "averages": ()}, "at least one"),
                      ({"feature_dim": 0}, "feature_dim"), ({"min_windows": 1.5}, "min_windows")):
        with pytest.raises(ConfigError, match=match):
            forecast(**kw)
    m = forecast()
    h, x, t, v, fut = inputs()
    with pytest.raises(ModelError, match="pretraining needs"):
        m(h[..., :4], x, t, v, future=fut)
    with pytest.raises(ModelError, match="future must be"):
        m(h, x, t, v, future=Future(fut.features[:, :10], fut.time_s[:, :10], fut.valid[:, :10]))
