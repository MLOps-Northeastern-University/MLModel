import math

import pytest
import torch

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer.network import PARTS, Batch, MotorNetwork

SMALL = {"movement_params": {"hidden_dim": 16, "output_dim": 24}, "d_model": 16, "dropout": 0.0,
         "decoder": {"heads": 2, "history_s": 600.0, "query_block": 16}, "heads": {"hidden": 8}}


def network(**kw):
    torch.manual_seed(0)
    net = MotorNetwork(**{**SMALL, **kw})
    net.movement.fit(torch.randn(200, 1024))
    return net.eval()


def batch(b=2, t=40, hours=True, targets=True, seed=1):
    """b sequences of t windows, 10 s apart, hours of 12 windows; row 1 ends in 5 padding windows."""
    g = torch.Generator().manual_seed(seed)
    time_s = torch.arange(t).repeat(b, 1) * 10.0 + 3600.0
    valid = torch.ones(b, t, dtype=torch.bool)
    if b > 1:
        valid[1, t - 5:] = False
    events = torch.zeros(b, t, 8)
    events[0, 3, :3] = torch.tensor([1.0, 1.0, 1.0])                 # a dose
    extra = {}
    if hours:
        extra["hour_of_window"] = torch.arange(t).div(12, rounding_mode="floor").repeat(b, 1)
        extra["hours"] = torch.tensor([[0, 1, 2]] * b)
    if targets:
        extra["targets"] = {"state_now": torch.tensor([[3.0, 1.0, math.nan]] * b),
                            "sleep_next_hour": torch.tensor([[0.0, 1.0, 0.0]] * b)}
    return Batch(movement=torch.randn(b, t, 1024, generator=g), time_s=time_s, valid=valid,
                 patient=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])[:b], events=events, **extra)


def test_builds_the_final_recipe_by_default():
    net = MotorNetwork()
    assert net.labels == ("sleep_now", "state_now", "tremor_now", "sleep_next_hour", "state_next_hour",
                          "tremor_next_hour")
    size = net.size()
    assert size["decoder"] == 100_096 and size["learning"] == sum(size[p] for p in PARTS)
    assert net.movement.name == "harnet10" and net.decoder.history_s == 8 * 3600
    assert net.tokens.dropout.p == 0.3 and net.loss.lam == 0.3 and net.loss.beta == 0.2


def test_forward_gives_what_the_batch_asks_for():
    net = network()
    only_windows = net(batch(hours=False, targets=False))
    assert only_windows.pooled is None and only_windows.loss is None
    assert only_windows.logits["state_now"].shape == (2, 40, 7)
    pooled = net(batch(targets=False))
    assert pooled.loss is None and pooled.pooled["tremor_next_hour"].attention.shape == (2, 3, 3)
    assert pooled.has_windows.tolist() == [[True, True, True], [True, True, True]]
    out = net(batch())
    assert torch.isfinite(out.loss.total) and set(out.loss.parts) == {"state_now", "sleep_next_hour"}


def test_every_part_learns_in_pretraining_style_and_only_heads_and_pooling_after_freezing():
    net = network().train()
    net(batch()).loss.total.backward()
    assert all(p.grad is not None for p in net.parameters())
    net.zero_grad(set_to_none=True)
    learning = net.freeze_all_but(("heads", "pooling"))
    assert learning and all(n.startswith(("heads.", "pooling.")) for n in learning)
    net(batch()).loss.total.backward()
    for name, p in net.named_parameters():
        assert (p.grad is not None) == name.startswith(("heads.", "pooling."))
    net.freeze_all_but(PARTS)
    assert all(p.requires_grad for p in net.parameters())


def test_frozen_parts_have_no_dropout_while_the_rest_trains():
    net = network(dropout=0.3)
    net.freeze_all_but(("heads", "pooling"))
    net.train()
    assert net.heads.training and net.pooling.training
    assert not any(m.training for part in ("movement", "tokens", "decoder") for m in getattr(net, part).modules())
    x = batch(targets=False)
    torch.testing.assert_close(net.encode(x), net.encode(x))           # no dropout before the heads
    assert not torch.allclose(net(x).logits["state_now"], net(x).logits["state_now"])   # the heads' dropout
    net.eval()
    assert not net.heads.training
    net.train()
    assert net.heads.training and not net.decoder.training             # still frozen, still eval


def test_keep_dropout_and_unfreezing_bring_dropout_back():
    net = network(dropout=0.3)
    net.freeze_all_but(("heads", "pooling"), keep_dropout=True)
    net.train()
    assert net.decoder.training
    x = batch(targets=False)
    assert not torch.allclose(net.encode(x), net.encode(x))
    net.freeze_all_but(("heads",))
    net.freeze_all_but(PARTS)
    assert all(m.training for m in net.modules())


def test_parameter_groups_decay_matrices_only_and_skip_frozen_weights():
    net = network()
    net.freeze_all_but("heads")
    decay, no_decay = net.parameter_groups(0.1)
    heads = {id(p) for p in net.heads.parameters()}
    assert {id(p) for p in decay["params"] + no_decay["params"]} == heads
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.1, 0.0)
    net.freeze_all_but(())
    with pytest.raises(ModelError, match="nothing to train"):
        net.parameter_groups(0.1)
    with pytest.raises(ConfigError, match="weight_decay"):
        network().parameter_groups(-1)


def test_never_sees_the_future():
    net = network()
    a = batch(targets=False)
    later = a.movement.clone()
    later[:, 25:] = torch.randn(2, 15, 1024)
    events = a.events.clone()
    events[:, 30, 5:7] = torch.tensor([1.0, -1.0])                    # a check-in later on
    b = Batch(later, a.time_s, a.valid, a.patient, events, a.hour_of_window, a.hours)
    for label in net.labels:
        torch.testing.assert_close(net(a).logits[label][:, :25], net(b).logits[label][:, :25])
        assert not torch.allclose(net(a).logits[label][:, 25:35], net(b).logits[label][:, 25:35])


def test_a_check_in_changes_what_comes_after_it():
    net = network()
    a = batch(targets=False)
    events = a.events.clone()
    events[0, 10, 5:7] = torch.tensor([1.0, -2 / 3])
    b = Batch(a.movement, a.time_s, a.valid, a.patient, events, a.hour_of_window, a.hours)
    before, after = net(a).logits["state_now"], net(b).logits["state_now"]
    torch.testing.assert_close(before[0, :10], after[0, :10])
    assert not torch.allclose(before[0, 10:], after[0, 10:])


def test_padding_changes_nothing():
    net = network()
    a = batch(b=1, t=30, targets=False)
    longer = Batch(torch.cat([a.movement, torch.randn(1, 6, 1024)], 1),
                   torch.cat([a.time_s, torch.full((1, 6), 1e9)], 1),
                   torch.cat([a.valid, torch.zeros(1, 6, dtype=torch.bool)], 1), a.patient,
                   torch.cat([a.events, torch.randn(1, 6, 8)], 1),
                   torch.cat([a.hour_of_window, torch.zeros(1, 6, dtype=torch.long)], 1), a.hours)
    x, y = net(a), net(longer)
    for label in net.labels:
        torch.testing.assert_close(x.logits[label], y.logits[label][:, :30])
        torch.testing.assert_close(x.pooled[label].attention, y.pooled[label].attention)


def test_settings_rebuild_the_same_network():
    net = network()
    copy = MotorNetwork(**net.settings).eval()
    copy.load_state_dict(net.state_dict())                            # strict: same weights, same buffers
    x = batch()
    torch.testing.assert_close(copy(x).loss.total, net(x).loss.total)
    assert copy.settings == net.settings and copy.settings is not net.settings


def test_load_parts_copies_the_pretrained_parts_only():
    pretrained, fresh = network(), MotorNetwork(**SMALL)
    heads_before = {k: v.clone() for k, v in fresh.heads.state_dict().items()}
    fresh.load_parts(pretrained.state_dict(), ("movement", "tokens", "decoder"))
    for part in ("movement", "tokens", "decoder"):
        for k, v in getattr(fresh, part).state_dict().items():
            assert torch.equal(v, getattr(pretrained, part).state_dict()[k])
    assert fresh.movement.is_fitted                                   # the standardisation came too
    assert all(torch.equal(v, heads_before[k]) for k, v in fresh.heads.state_dict().items())


def test_load_parts_refuses_weights_that_dont_fit():
    state = network().state_dict()
    with pytest.raises(ModelError, match="missing"):
        MotorNetwork(**SMALL).load_parts({k: v for k, v in state.items() if not k.startswith("decoder.layers.1")},
                                         ("decoder",))
    wider = MotorNetwork(**{**SMALL, "movement_params": {"hidden_dim": 32, "output_dim": 24}})
    with pytest.raises(ModelError, match="wrong shape"):
        wider.load_parts(state, ("movement",))
    with pytest.raises(ConfigError, match="unknown network parts"):
        wider.load_parts(state, ("encoder",))


def test_runs_under_mixed_precision_with_a_float32_loss():
    net = network()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(batch())
    assert out.loss.total.dtype == torch.float32 and torch.isfinite(out.loss.total)
    torch.testing.assert_close(out.loss.total, net(batch()).loss.total, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu():
    net = network()
    want = net(batch()).loss.total
    got = net.cuda()(batch().to("cuda")).loss.total
    torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-4)


def test_refuses_bad_settings():
    for kw, match in (({"decoder": {"width": 3}}, "decoder settings"), ({"heads": {"d_model": 8}}, "shared"),
                      ({"loss": {"lam": -1}}, "lam"), ({"movement": "harnet9"}, "must be one of"),
                      ({"pooling": {"labels": ("a",)}}, "shared")):
        with pytest.raises(ConfigError, match=match):
            MotorNetwork(**{**SMALL, **kw})
    with pytest.raises(ConfigError, match="unknown network parts"):
        network().freeze_all_but(("heads", "attention"))


def test_batch_refuses_mismatched_parts():
    a = batch()
    base = dict(movement=a.movement, time_s=a.time_s, valid=a.valid, patient=a.patient, events=a.events)
    for change, match in (({"valid": a.valid.int()}, "valid"), ({"movement": a.movement[:, :5]}, "movement"),
                          ({"events": a.events[..., :7]}, "events"), ({"patient": a.patient[:, 0]}, "patient"),
                          ({"time_s": a.time_s[0]}, "time_s"), ({"hours": a.hours}, "both or neither"),
                          ({"targets": a.targets}, "targets need")):
        with pytest.raises(ModelError, match=match):
            Batch(**{**base, **change})


def test_batch_to_moves_every_tensor():
    a = batch().to("meta")
    assert a.movement.device.type == "meta" and a.targets["state_now"].device.type == "meta"
    assert a.hours.device.type == "meta" and batch(hours=False, targets=False).to("meta").hours is None
