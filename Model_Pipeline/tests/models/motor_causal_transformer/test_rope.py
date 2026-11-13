import math

import pytest
import torch

from scripts.core.errors import ConfigError, ModelError
from scripts.models.motor_causal_transformer.rope import Rotation, TimeRoPE

DAY = 86_400.0


def reference(x, time_s, base=10_000.0, step_s=10.0):
    """RoPE written out by hand, in float64, for x of (*time_s.shape, ..., d).

    Each pair (a, b) rotated by its angle: (a cos - b sin, a sin + b cos), angle = (t / step) * theta.
    """
    x = x.to(torch.float64)
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angles = (torch.as_tensor(time_s, dtype=torch.float64) / step_s).unsqueeze(-1) * theta
    angles = angles.view(*angles.shape[:-1], *([1] * (x.dim() - angles.dim())), d // 2)
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * angles.cos() - b * angles.sin()
    out[..., 1::2] = a * angles.sin() + b * angles.cos()
    return out


def score(rope, q, k, tq, tk):
    """The attention score of one query and one key at their times."""
    return (rope(q, tq) * rope(k, tk)).sum()


def test_matches_rope_written_out_by_hand():
    torch.manual_seed(0)
    x = torch.randn(5, 16, dtype=torch.float64)
    t = torch.tensor([0.0, 10.0, 3_600.0, 28_800.0, 30 * DAY], dtype=torch.float64)
    assert torch.allclose(TimeRoPE(16)(x, t), reference(x, t), atol=1e-12)


def test_score_depends_only_on_the_time_between():
    torch.manual_seed(1)
    rope = TimeRoPE(16)
    q, k = torch.randn(1, 16), torch.randn(1, 16)
    tq, tk = torch.tensor([7_200.0]), torch.tensor([1_800.0])
    near = score(rope, q, k, tq, tk)
    for shift in (10.0, 3 * 3_600.0, 30 * DAY):
        assert torch.isclose(near, score(rope, q, k, tq + shift, tk + shift), atol=1e-5), shift


def test_score_changes_with_the_time_between():
    torch.manual_seed(2)
    rope = TimeRoPE(16)
    q, k = torch.randn(1, 16), torch.randn(1, 16)
    t0 = torch.tensor([0.0])
    scores = [score(rope, q, k, t0, torch.tensor([gap])).item() for gap in (0.0, 10.0, 600.0, 7_200.0)]
    assert len({round(s, 4) for s in scores}) == len(scores)


def test_time_zero_leaves_x_unchanged_and_length_is_kept():
    torch.manual_seed(3)
    rope = TimeRoPE(16)
    x = torch.randn(3, 16)
    assert torch.allclose(rope(x, torch.zeros(3)), x, atol=1e-7)
    y = rope(x, torch.tensor([10.0, 5_000.0, 9 * DAY]))
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_every_head_shares_its_tokens_rotation():
    torch.manual_seed(4)
    x = torch.randn(2, 6, 4, 16)                       # (batch, seq, heads, head_dim)
    t = torch.rand(2, 6) * DAY                         # (batch, seq)
    y = TimeRoPE(16)(x, t)
    assert y.shape == x.shape
    for h in range(4):
        assert torch.allclose(y[:, :, h], reference(x[:, :, h], t).float(), atol=1e-5)


def test_queries_and_keys_stacked_rotate_like_each_alone():
    torch.manual_seed(5)
    rope = TimeRoPE(16)
    qk = torch.randn(2, 6, 2, 4, 16)                   # (batch, seq, q/k, heads, head_dim)
    t = torch.rand(2, 6) * DAY
    rot = rope.rotation(t)
    both = rope.rotate(qk, rot)
    assert torch.allclose(both[:, :, 0], rope.rotate(qk[:, :, 0], rot))
    assert torch.allclose(both[:, :, 1], rope.rotate(qk[:, :, 1], rot))
    assert torch.allclose(both.double(), reference(qk, t), atol=1e-5)


def test_one_rotation_serves_every_layer_and_matches_calling_the_module():
    torch.manual_seed(6)
    rope = TimeRoPE(16)
    t = torch.rand(3, 8) * DAY
    rot = rope.rotation(t)
    assert isinstance(rot, Rotation) and rot.time_shape == (3, 8) and rot.turn.dtype == torch.complex64
    for _ in range(2):                                 # two layers, the same rotation
        x = torch.randn(3, 8, 4, 16)
        assert torch.equal(rope.rotate(x, rot), rope(x, t))


def test_a_slice_of_a_fused_projection_rotates_correctly():
    """q and k sliced out of one q/k/v tensor, in any layout, without copying first."""
    torch.manual_seed(7)
    rope = TimeRoPE(16)
    t = torch.rand(2, 5) * DAY
    qkv = torch.randn(2, 5, 3, 4, 16)
    for x in (qkv[:, :, :2], qkv[:, :, 1], qkv.transpose(2, 3)[:, :, :, 0]):
        assert torch.allclose(rope(x, t).double(), reference(x, t), atol=1e-5)
    odd = torch.randn(2, 5, 4, 17)[..., 1:]            # starts one number in: no complex view possible
    assert torch.allclose(rope(odd, t).double(), reference(odd, t), atol=1e-5)


def test_integer_times_are_accepted():
    x = torch.randn(3, 16)
    t = torch.tensor([0, 10, 20], dtype=torch.int64)
    assert torch.allclose(TimeRoPE(16)(x, t), TimeRoPE(16)(x, t.double()))


def test_keeps_the_input_dtype_and_rotates_bf16_accurately():
    torch.manual_seed(8)
    x = torch.randn(64, 16).to(torch.bfloat16)
    t = torch.rand(64) * 30 * DAY
    y = TimeRoPE(16)(x, t)
    assert y.dtype == torch.bfloat16
    # Only the final cast to bf16 loses accuracy: within bf16's own rounding of the exact result.
    assert torch.allclose(y.double(), reference(x, t), rtol=2 ** -7, atol=2 ** -7)


def test_float64_input_is_rotated_in_float64():
    x = torch.randn(4, 16, dtype=torch.float64)
    t = torch.rand(4, dtype=torch.float64) * 365 * DAY
    y = TimeRoPE(16)(x, t)
    assert y.dtype == torch.float64
    assert torch.allclose(y, reference(x, t), atol=1e-12)


def test_stays_accurate_after_years_of_wear():
    torch.manual_seed(9)
    x = torch.randn(4, 16)
    t = torch.tensor([3 * 365 * DAY + s for s in (0.0, 10.0, 20.0, 30.0)], dtype=torch.float64)
    assert torch.allclose(TimeRoPE(16)(x, t).double(), reference(x, t), atol=1e-5)


def test_gradients_are_exact():
    rope = TimeRoPE(4)
    x = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.0, 25.0, 7_200.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: rope(x, t), (x,))


def test_wavelengths_for_the_model_settings():
    w = TimeRoPE(16).wavelengths_s()
    assert len(w) == 8
    assert math.isclose(w[0].item(), 2 * math.pi * 10, rel_tol=1e-12)                 # about 63 s
    assert math.isclose(w[-1].item() / 3_600, 55.19, rel_tol=1e-3)                    # about 55 h
    assert torch.all(w[1:] > w[:-1])
    assert w[-1] > 8 * 3_600                                                          # beyond the 8 h reach


def test_has_nothing_to_train_or_save():
    rope = TimeRoPE(16)
    assert list(rope.parameters()) == []
    assert rope.state_dict() == {}
    assert "head_dim=16" in repr(rope)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_the_gpu():
    rope = TimeRoPE(16).cuda()
    x = torch.randn(2, 6, 4, 16, device="cuda", dtype=torch.bfloat16)
    t = torch.rand(2, 6, dtype=torch.float64) * DAY                    # times may come from the CPU
    y = rope(x, t)
    assert y.device.type == "cuda" and y.dtype == torch.bfloat16
    assert torch.allclose(y.double().cpu(), reference(x.cpu(), t), rtol=2 ** -7, atol=2 ** -7)


@pytest.mark.parametrize("head_dim", [0, 1, 7, 16.0, True, "16"])
def test_refuses_a_head_dim_it_cannot_pair(head_dim):
    with pytest.raises(ConfigError, match="head_dim"):
        TimeRoPE(head_dim)


@pytest.mark.parametrize("base", [1.0, 0.5, -10.0, float("inf"), float("nan"), True, "10000"])
def test_refuses_a_bad_base(base):
    with pytest.raises(ConfigError, match="base"):
        TimeRoPE(16, base=base)


@pytest.mark.parametrize("step_s", [0.0, -10.0, float("inf"), float("nan"), True, "10"])
def test_refuses_a_bad_step(step_s):
    with pytest.raises(ConfigError, match="step_s"):
        TimeRoPE(16, step_s=step_s)


def test_refuses_the_wrong_head_size():
    with pytest.raises(ModelError, match="16 numbers per head"):
        TimeRoPE(16)(torch.randn(3, 8), torch.zeros(3))
    with pytest.raises(ModelError, match="16 numbers per head"):
        TimeRoPE(16)(torch.randn(16), torch.zeros(16))


def test_refuses_tokens_that_do_not_match():
    rope = TimeRoPE(16)
    with pytest.raises(ModelError, match="doesn't start with"):
        rope(torch.randn(2, 6, 4, 16), torch.zeros(2, 5))
    with pytest.raises(ModelError, match="doesn't start with"):
        rope(torch.randn(6, 16), torch.zeros(7))
    with pytest.raises(ModelError, match="at least one dimension"):
        rope.rotation(torch.tensor(5.0))
    with pytest.raises(ModelError, match="float32 or float64"):
        rope.rotation(torch.zeros(3), dtype=torch.bfloat16)
