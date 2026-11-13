import math

import pytest
import torch
from torch.nn import functional as F

from scripts.core.errors import ConfigError, ModelError
from scripts.models.motor_causal_transformer.decoder import Decoder
from tests.models.motor_causal_transformer.test_rope import reference as rope_reference

HOUR = 3600.0


def times(b, t, gaps=(), start=5 * 86_400.0):
    """(b, t) times of back-to-back 10-s windows from `start`, with extra seconds added before given tokens."""
    steps = torch.full((b, t), 10.0, dtype=torch.float64)
    for index, extra in gaps:
        steps[:, index] += extra
    return start + torch.cumsum(steps, dim=1)


def small(**kw):
    """A small float64 decoder, deterministic (eval mode), for exact comparisons."""
    settings = dict(d_model=16, heads=2, layers=2, dropout=0.0, history_s=HOUR, query_block=7)
    settings.update(kw)
    torch.manual_seed(0)
    return Decoder(**settings).double().eval()


def reference(model, tokens, time_s, valid):
    """The decoder written out plainly: every query against every key, the rule as one mask."""
    x = tokens.double()
    time_s = time_s.double()
    t = x.shape[1]
    i, j = torch.arange(t)[:, None], torch.arange(t)[None, :]
    gap = time_s[:, :, None] - time_s[:, None, :]
    allowed = (j <= i) & (gap < model.history_s) & valid[:, None, :]
    allowed = allowed | ((i == j) & ~valid[:, :, None])
    heads, hd = model.layers[0].heads, model.layers[0].head_dim
    safe_t = torch.where(valid, time_s, torch.zeros_like(time_s))
    for layer in model.layers:
        q, k, v = layer.qkv(layer.norm1(x)).unflatten(-1, (3, heads, hd)).unbind(2)   # (b, t, heads, hd)
        q, k = (rope_reference(z, safe_t, model.rope.base, model.rope.step_s) for z in (q, k))
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(hd)
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        att = torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v).flatten(2)
        x = x + layer.out(att)
        x = x + layer.ff(layer.norm2(x))
    return model.norm(x)


def test_matches_the_decoder_written_out_plainly():
    model = small()
    tokens = torch.randn(3, 30, 16, dtype=torch.float64)
    time_s = times(3, 30, gaps=[(10, 1_800.0), (21, 400.0)])
    valid = torch.ones(3, 30, dtype=torch.bool)
    valid[1, 24:] = False
    valid[2, 9:] = False
    out, ref = model(tokens, time_s, valid), reference(model, tokens, time_s, valid)
    assert torch.allclose(out[valid], ref[valid], atol=1e-10)


@pytest.mark.parametrize("query_block", [1, 5, 13, 30, 1000])
def test_query_block_changes_nothing_but_speed(query_block):
    tokens = torch.randn(2, 30, 16, dtype=torch.float64)
    time_s = times(2, 30, gaps=[(12, 2_000.0)])
    valid = torch.ones(2, 30, dtype=torch.bool)
    valid[1, 20:] = False
    a = small(query_block=7)(tokens, time_s, valid)
    b = small(query_block=query_block)(tokens, time_s, valid)
    assert torch.allclose(a[valid], b[valid], atol=1e-12)


def test_never_sees_the_future():
    model = small()
    tokens = torch.randn(1, 20, 16, dtype=torch.float64)
    time_s = times(1, 20)
    before = model(tokens, time_s)
    changed = tokens.clone()
    changed[:, 12:] = torch.randn(1, 8, 16, dtype=torch.float64) * 10
    after = model(changed, time_s)
    assert torch.allclose(before[:, :12], after[:, :12], atol=1e-12)
    assert not torch.allclose(before[:, 12:], after[:, 12:])


def test_one_layer_sees_back_history_s_and_two_layers_twice_that():
    """history_s is 100 s here. A layer sees keys 0 to 90 s old (100 s is out), so a query sees 9
    windows back with one layer and 18 with two (9 more through its oldest key's own past)."""
    tokens = torch.randn(1, 40, 16, dtype=torch.float64)
    time_s = times(1, 40)
    for layers, reach in ((1, 10), (2, 19)):
        model = small(layers=layers, history_s=100.0)
        assert model.reach_s == 100.0 * layers
        query = 39
        base = model(tokens, time_s)[0, query]
        for key in range(40):
            changed = tokens.clone()
            changed[0, key] += torch.randn(16, dtype=torch.float64) * 5   # random: LayerNorm erases a constant
            moved = not torch.allclose(model(changed, time_s)[0, query], base, atol=1e-12)
            assert moved == (query - reach < key <= query), (layers, key)


def test_a_gap_in_wear_counts_as_real_time():
    """Two neighbouring tokens 2 h apart don't see each other with history_s = 1 h."""
    model = small(layers=1)
    tokens = torch.randn(1, 6, 16, dtype=torch.float64)
    time_s = times(1, 6, gaps=[(3, 2 * HOUR)])
    base = model(tokens, time_s)
    changed = tokens.clone()
    changed[0, 2] += torch.randn(16, dtype=torch.float64) * 5     # the last token before the gap
    after = model(changed, time_s)
    assert not torch.allclose(base[0, 2], after[0, 2])              # the change is real...
    assert torch.allclose(base[0, 3:], after[0, 3:], atol=1e-12)    # ...but out of reach across the gap


def test_padding_changes_nothing_and_gives_no_nan():
    model = small()
    tokens = torch.randn(2, 25, 16, dtype=torch.float64)
    time_s = times(2, 25)
    valid = torch.ones(2, 25, dtype=torch.bool)
    valid[1, 15:] = False
    base = model(tokens, time_s, valid)
    junk_tokens, junk_times = tokens.clone(), time_s.clone()
    junk_tokens[1, 15:] = 1e6
    junk_times[1, 15:] = float("nan")                     # padding times are ignored, even NaN
    out = model(junk_tokens, junk_times, valid)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[valid], base[valid], atol=1e-12)
    alone = model(tokens[1:, :15], time_s[1:, :15])       # the padded row on its own, unpadded
    assert torch.allclose(out[1, :15], alone[0], atol=1e-12)


def test_an_all_padding_row_is_harmless():
    model = small()
    tokens = torch.randn(2, 10, 16, dtype=torch.float64)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1] = False
    out = model(tokens, times(2, 10), valid)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], model(tokens[:1], times(1, 10))[0], atol=1e-12)


def test_integer_times_give_the_same_result():
    model = small()
    tokens = torch.randn(1, 12, 16, dtype=torch.float64)
    time_s = times(1, 12)
    assert torch.allclose(model(tokens, time_s), model(tokens, time_s.to(torch.int64)), atol=1e-12)


def test_reach_can_be_reused_and_must_match():
    model = small()
    tokens = torch.randn(2, 12, 16, dtype=torch.float64)
    time_s = times(2, 12)
    reach = model.reach(time_s)
    assert torch.equal(model(tokens, time_s, reach=reach), model(tokens, time_s))
    with pytest.raises(ModelError, match="reach is for tokens"):
        model(tokens[:, :10], time_s[:, :10], reach=reach)


def test_blocks_only_hold_keys_in_reach():
    model = small(history_s=100.0, query_block=10)
    reach = model.reach(times(1, 40))
    assert [(b.start, b.end, b.first_key) for b in reach.blocks] == [(0, 10, 0), (10, 20, 1), (20, 30, 11),
                                                                     (30, 40, 21)]
    assert all(b.allowed.shape == (1, 1, b.end - b.start, b.end - b.first_key) for b in reach.blocks)


def test_dropout_only_while_training():
    model = small(dropout=0.3)
    tokens = torch.randn(1, 12, 16, dtype=torch.float64)
    time_s = times(1, 12)
    assert torch.equal(model(tokens, time_s), model(tokens, time_s))
    model.train()
    assert not torch.allclose(model(tokens, time_s), model(tokens, time_s))


def test_every_weight_learns_and_gradients_are_finite():
    model = small()
    model.train()
    tokens = torch.randn(2, 20, 16, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 14:] = False
    out = model(tokens, times(2, 20), valid)
    out[valid].pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad[1, 14:].abs().sum() == 0            # padding has no say in anything


def test_mixed_precision_stays_close_to_float32():
    torch.manual_seed(0)
    model = Decoder(d_model=16, heads=2, dropout=0.0).eval()
    tokens, time_s = torch.randn(2, 50, 16), times(2, 50)
    full = model(tokens, time_s)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        half = model(tokens, time_s)
    assert half.dtype == torch.float32                  # autocast always runs LayerNorm, the last step, in float32
    assert torch.isfinite(half).all()
    assert (half - full).abs().mean() < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu():
    torch.manual_seed(0)
    model = Decoder(d_model=16, heads=2, dropout=0.0, query_block=16).eval()
    tokens, time_s = torch.randn(2, 70, 16), times(2, 70, gaps=[(30, 900.0)])
    valid = torch.ones(2, 70, dtype=torch.bool)
    valid[1, 50:] = False
    cpu = model(tokens, time_s, valid)
    gpu = model.cuda()(tokens.cuda(), time_s, valid)
    assert torch.allclose(gpu.cpu()[valid], cpu[valid], atol=1e-4)


def test_the_model_settings_give_the_expected_size_and_start():
    model = Decoder()
    per_layer = (64 * 192 + 192) + (64 * 64 + 64) + 2 * 128 + (64 * 256 + 256) + (256 * 64 + 64)
    assert sum(p.numel() for p in model.parameters()) == 2 * per_layer + 128
    assert {k.split(".")[0] for k in model.state_dict()} == {"layers", "norm"}
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            assert p.abs().sum() == 0, name
        if "norm" in name and name.endswith("weight"):
            assert torch.all(p == 1), name


@pytest.mark.parametrize("kw, match", [
    ({"d_model": 0}, "d_model"), ({"d_model": 64.0}, "d_model"), ({"heads": 3}, "split evenly"),
    ({"heads": 64}, "head_dim"), ({"layers": 0}, "layers"), ({"ff_mult": True}, "ff_mult"),
    ({"query_block": -1}, "query_block"), ({"history_s": 0}, "history_s"),
    ({"history_s": float("inf")}, "history_s"), ({"dropout": 1.0}, "dropout"), ({"dropout": -0.1}, "dropout"),
])
def test_refuses_bad_settings(kw, match):
    with pytest.raises(ConfigError, match=match):
        Decoder(**kw)


def test_refuses_bad_inputs():
    model = small()
    tokens, time_s = torch.randn(1, 10, 16, dtype=torch.float64), times(1, 10)
    with pytest.raises(ModelError, match="tokens must be"):
        model(tokens[..., :8], time_s)
    with pytest.raises(ModelError, match="tokens must be"):
        model(tokens, time_s[:, :9])
    with pytest.raises(ModelError, match="at least one token"):
        model.reach(time_s[:, :0])
    with pytest.raises(ModelError, match="valid"):
        model(tokens, time_s, torch.ones(1, 9, dtype=torch.bool))
    gappy = torch.ones(1, 10, dtype=torch.bool)
    gappy[0, 4] = False
    with pytest.raises(ModelError, match="padding must come after"):
        model(tokens, time_s, gappy)
    backwards = time_s.clone()
    backwards[0, 6] = backwards[0, 2]
    with pytest.raises(ModelError, match="increase"):
        model(tokens, backwards)
    broken = time_s.clone()
    broken[0, 3] = float("nan")
    with pytest.raises(ModelError, match="finite"):
        model(tokens, broken)


def test_sdpa_mask_convention_is_true_means_attend():
    """The decoder relies on it: a boolean attn_mask marks the keys a query may see."""
    q = k = v = torch.eye(2)[None, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor([[True, False], [True, False]]))
    assert torch.allclose(out[0, 0], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
