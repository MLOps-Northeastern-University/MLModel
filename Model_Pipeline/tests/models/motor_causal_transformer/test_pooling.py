import math

import pytest
import torch

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer.pooling import GatedPooling, Pooled, hour_members

LABELS = ("sleep_now", "state_now", "tremor_now")
CLASSES = {"sleep_now": 2, "state_now": 7, "tremor_now": 3}


def example(seed=0, b=1, dtype=torch.float32):
    """14 windows: hour 5 (windows 0-5), hour 6 (6-11), 2 padding; windows 2 and 9 unworn."""
    torch.manual_seed(seed)
    hour = torch.tensor([[5] * 6 + [6] * 6 + [-1] * 2] * b)
    worn = torch.ones(b, 14, dtype=torch.bool)
    worn[:, [2, 9, 12, 13]] = False
    members = hour_members(hour, worn, torch.tensor([[5, 6]] * b))
    probs = {lab: torch.softmax(torch.randn(b, 14, k, dtype=dtype), -1) for lab, k in CLASSES.items()}
    return torch.randn(b, 14, 16, dtype=dtype), probs, members


def pooling(**kw):
    torch.manual_seed(0)
    return GatedPooling(LABELS, **{"d_model": 16, "gate_dim": 8, **kw})


def reference(m, h, probs, members):
    """The pooling written out plainly: one hour and one output at a time, the softmax by hand."""
    scores = m.score(torch.tanh(m.V(h)) * torch.sigmoid(m.U(h)))
    out = {}
    for i, label in enumerate(m.labels):
        att = torch.zeros(*members.shape[:2], probs[label].shape[-1], dtype=h.dtype)
        mean = torch.zeros_like(att)
        for b in range(members.shape[0]):
            for j in range(members.shape[1]):
                idx = torch.nonzero(members[b, j]).flatten()
                if not len(idx):
                    continue
                s = scores[b, idx, i] / m.log_temperature[i].exp()
                e = torch.exp(s - s.max())
                att[b, j] = ((e / e.sum())[:, None] * probs[label][b, idx]).sum(0)
                mean[b, j] = probs[label][b, idx].mean(0)
        out[label] = (att, mean)
    return out


def test_matches_the_pooling_written_out_plainly():
    m = pooling()
    with torch.no_grad():
        m.log_temperature.copy_(torch.tensor([0.3, -1.0, 2.0]))      # a different temperature each
    h, probs, members = example(b=3)
    members[1, 0] = False                                              # an empty hour, too
    out, has = m(h, probs, members)
    want = reference(m, h, probs, members)
    for label in LABELS:
        torch.testing.assert_close(out[label].attention, want[label][0])
        torch.testing.assert_close(out[label].mean, want[label][1])
    assert has.tolist() == [[True, True], [False, True], [True, True]]


def test_weights_add_up_to_one_within_each_hour_and_skip_unworn_windows_and_padding():
    h, probs, members = example()
    out, has = pooling()(h, probs, members)
    for p in out.values():
        assert isinstance(p, Pooled)
        torch.testing.assert_close(p.weights.sum(-1), torch.ones(1, 2))
        assert p.weights[0, 0, 6:].sum() == 0 and p.weights[0, 1, :6].sum() == 0   # each hour: its own windows
        assert p.weights[0, :, [2, 9, 12, 13]].sum() == 0                         # unworn, padding: exactly 0
        torch.testing.assert_close(p.attention.sum(-1), torch.ones(1, 2))         # a proper mix
        assert (p.attention >= 0).all()
    assert has.all()


def test_a_large_temperature_is_the_plain_average_and_a_small_one_lets_a_window_decide():
    h, probs, members = example()
    big = pooling(init_temperature=1e6)
    out, _ = big(h, probs, members)
    torch.testing.assert_close(out["state_now"].attention, out["state_now"].mean, rtol=1e-4, atol=1e-5)
    small = pooling(init_temperature=0.01)
    out, _ = small(h, probs, members)
    w = out["tremor_now"].weights[0, 0]
    top = int(w.argmax())
    assert w[top] > 0.99
    torch.testing.assert_close(out["tremor_now"].attention[0, 0], probs["tremor_now"][0, top],
                               rtol=1e-2, atol=1e-2)


def test_an_hour_with_no_worn_windows_gets_zeros_and_no_nan_even_in_gradients():
    m = pooling()
    h, probs, members = example()
    members[0, 1] = False
    h.requires_grad_(True)
    out, has = m(h, probs, members)
    assert has.tolist() == [[True, False]]
    for p in out.values():
        assert torch.isfinite(p.attention).all() and p.attention[0, 1].sum() == 0 and p.mean[0, 1].sum() == 0
    sum(p.attention.sum() for p in out.values()).backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_hours_dont_affect_each_other():
    m = pooling()
    h, probs, members = example()
    out, _ = m(h, probs, members)
    h2, probs2 = h.clone(), {k: v.clone() for k, v in probs.items()}
    h2[0, 6:] = torch.randn(8, 16)                                   # change hour 6 and the padding entirely
    for k, v in probs2.items():
        v[0, 6:] = torch.softmax(torch.randn(8, v.shape[-1]), -1)
    out2, _ = m(h2, probs2, members)
    for label in LABELS:
        torch.testing.assert_close(out[label].attention[0, 0], out2[label].attention[0, 0])


def test_each_output_has_its_own_temperature_and_they_start_where_asked():
    m = pooling(init_temperature=2.5)
    assert list(m.temperatures) == list(LABELS)
    assert all(t == pytest.approx(2.5) for t in m.temperatures.values())
    with torch.no_grad():
        m.log_temperature[1] = math.log(0.1)
    h, probs, members = example()
    out, _ = m(h, probs, members)
    # Same scores for every output would still pool differently: only state_now's temperature moved.
    assert out["state_now"].weights.max() > out["sleep_now"].weights.max()


def test_gradients_reach_the_windows_the_gate_and_the_temperatures():
    m = pooling()
    h, probs, members = example()
    h.requires_grad_(True)
    probs = {k: v.requires_grad_(True) for k, v in probs.items()}
    out, _ = m(h, probs, members)
    target = torch.tensor([0, 1])
    loss = sum(-torch.log(p.attention[0, torch.arange(2), target]).mean() for p in out.values())
    loss.backward()
    assert h.grad.abs().sum() > 0 and all(v.grad.abs().sum() > 0 for v in probs.values())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_gradcheck_in_float64():
    m = pooling().double()
    h, probs, members = example(b=2, dtype=torch.float64)
    members[1, 1] = False
    h.requires_grad_(True)

    def f(h, s):
        return m(h, {**probs, "state_now": s}, members)[0]["state_now"].attention

    assert torch.autograd.gradcheck(f, (h, probs["state_now"].clone().requires_grad_(True)))


def test_runs_in_float32_under_mixed_precision():
    m = pooling()
    h, probs, members = example()
    want, _ = m(h, probs, members)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, _ = m(h.bfloat16(), {k: v.bfloat16() for k, v in probs.items()}, members)
    for label in LABELS:
        assert out[label].attention.dtype == torch.float32
        torch.testing.assert_close(out[label].attention, want[label].attention, rtol=0.05, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu():
    m = pooling()
    h, probs, members = example(b=2)
    want, _ = m(h, probs, members)
    out, _ = m.cuda()(h.cuda(), {k: v.cuda() for k, v in probs.items()}, members.cuda())
    for label in LABELS:
        torch.testing.assert_close(out[label].attention.cpu(), want[label].attention)


def test_the_weights_and_names_are_the_prototypes():
    m = GatedPooling(LABELS)
    assert sorted(m.state_dict()) == sorted(["V.weight", "V.bias", "U.weight", "U.bias", "score.weight",
                                             "score.bias", "log_temperature"])
    assert sum(p.numel() for p in m.parameters()) == 2 * (64 * 64 + 64) + (64 * 3 + 3) + 3
    assert m.score.bias.abs().sum() == 0                              # init_weights: zero biases


def test_hour_members():
    hour = torch.tensor([[3, 3, 4, 4, -1]])
    worn = torch.tensor([[True, False, True, True, False]])
    members = hour_members(hour, worn, torch.tensor([[3, 4, 9]]))
    assert members.tolist() == [[[True, False, False, False, False], [False, False, True, True, False],
                                 [False] * 5]]
    with pytest.raises(ModelError, match="worn must be bool"):
        hour_members(hour, worn.int(), torch.tensor([[3]]))
    with pytest.raises(ModelError, match="integers"):
        hour_members(hour.float(), worn, torch.tensor([[3]]))
    with pytest.raises(ModelError, match="batch, seq"):
        hour_members(hour, worn[:, :4], torch.tensor([[3]]))
    with pytest.raises(ModelError, match="n_hours"):
        hour_members(hour, worn, torch.tensor([3]))


def test_refuses_bad_inputs():
    m = pooling()
    h, probs, members = example()
    with pytest.raises(ModelError, match="window probabilities for"):
        m(h, {"state_now": probs["state_now"]}, members)
    with pytest.raises(ModelError, match="reads"):
        m(h[..., :8], probs, members)
    with pytest.raises(ModelError, match="members"):
        m(h, probs, members.int())
    with pytest.raises(ModelError, match="members"):
        m(h, probs, members[..., :10])
    with pytest.raises(ModelError, match="tremor_now"):
        m(h, {**probs, "tremor_now": probs["tremor_now"][:, :10]}, members)


@pytest.mark.parametrize("kw, match", [
    ({"labels": ()}, "labels"), ({"labels": ("a", "a")}, "labels"), ({"labels": ("a", "")}, "labels"),
    ({"d_model": 0}, "d_model"), ({"gate_dim": 8.0}, "gate_dim"), ({"init_temperature": 0}, "init_temperature"),
    ({"init_temperature": math.inf}, "init_temperature"), ({"init_temperature": True}, "init_temperature"),
])
def test_refuses_bad_settings(kw, match):
    with pytest.raises(ConfigError, match=match):
        GatedPooling(**{"labels": LABELS, **kw})
