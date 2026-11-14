import pytest
import torch

from scripts.core.errors import ConfigError, ModelError
from scripts.core.targets import LABEL_NAMES, TARGETS
from scripts.models.motor_causal_transformer.heads import TARGET_CLASSES, Heads, probabilities

LABELS = ("sleep_now", "state_now", "tremor_now", "sleep_next_hour", "state_next_hour", "tremor_next_hour")


def heads(**kw):
    torch.manual_seed(0)
    return Heads(**{"d_model": 16, "hidden": 8, "dropout": 0.0, **kw}).eval()


def test_labels_and_classes_come_from_the_targets():
    assert TARGET_CLASSES == {"sleep": 2, "state": 7, "tremor": 3}
    assert all(TARGET_CLASSES[t] == len(TARGETS[t].classes) for t in TARGETS)
    h = heads()
    assert h.labels == LABELS and set(h.labels) == LABEL_NAMES
    out = h(torch.randn(2, 5, 16))
    assert tuple(out) == LABELS
    assert all(out[f"{t}_{hz}"].shape == (2, 5, k) for t, k in TARGET_CLASSES.items() for hz in ("now", "next_hour"))


def test_one_output_layer_split_equals_a_layer_per_target():
    """The fused layer is the prototype's three layers, rows stacked: sleep 0-1, state 2-8, tremor 9-11."""
    h = heads()
    x = torch.randn(3, 16)
    out = h(x)
    for horizon, head in h.heads.items():
        z = head.trunk(x)
        for target, rows in (("sleep", slice(0, 2)), ("state", slice(2, 9)), ("tremor", slice(9, 12))):
            assert head.rows(target) == rows
            want = z @ head.out.weight[rows].T + head.out.bias[rows]
            torch.testing.assert_close(out[f"{target}_{horizon}"], want)


def test_the_horizons_are_separate_heads():
    h = heads()
    with torch.no_grad():
        for p in h.heads["next_hour"].parameters():
            p.add_(torch.randn_like(p))
    x = torch.randn(4, 16)
    before = heads()(x)
    after = h(x)
    assert all(torch.equal(before[f"{t}_now"], after[f"{t}_now"]) for t in TARGET_CLASSES)
    assert not torch.allclose(before["state_next_hour"], after["state_next_hour"])
    only_now = heads(horizons=("now",))
    assert only_now.labels == LABELS[:3]


def test_priors_start_every_output_at_the_class_mix():
    h = heads(hidden=64, d_model=64)
    mix = {"sleep": [3, 1], "state": [1, 2, 3, 4, 3, 2, 1], "tremor": [0.7, 0.2, 0.1]}
    h.set_priors(mix)
    p = probabilities(h(torch.randn(500, 64)))
    for label, probs in p.items():
        freq = torch.tensor(mix[label.split("_")[0]], dtype=torch.float32)
        torch.testing.assert_close(probs.mean(0), freq / freq.sum(), atol=0.01, rtol=0)
    assert all(head.out.weight.abs().max() > 0 for head in h.heads.values())     # shrunk, not zeroed


def test_priors_leave_other_targets_alone_and_refuse_bad_ones():
    h = heads()
    state_rows = h.heads["now"].out.weight[2:9].clone()
    h.set_priors({"sleep": [1, 1]})
    assert torch.equal(h.heads["now"].out.weight[2:9], state_rows)
    for bad, match in (({"mood": [1, 1]}, "unknown target"), ({"sleep": [1, 1, 1]}, "2 positive"),
                       ({"sleep": [1, 0]}, "2 positive"), ({"tremor": [1, float("nan"), 1]}, "3 positive")):
        with pytest.raises(ConfigError, match=match):
            h.set_priors(bad)
    with pytest.raises(ConfigError, match="weight_scale"):
        h.set_priors({"sleep": [1, 1]}, weight_scale=0)


def test_a_linear_head():
    h = heads(hidden=0)
    assert isinstance(h.heads["now"].trunk, torch.nn.Identity)
    assert h(torch.randn(2, 16))["tremor_now"].shape == (2, 3)


def test_dropout_only_while_training():
    h = heads(dropout=0.3)
    x = torch.randn(4, 10, 16)
    assert torch.equal(h(x)["state_now"], h(x)["state_now"])
    h.train()
    assert not torch.allclose(h(x)["state_now"], h(x)["state_now"])


def test_every_weight_learns():
    h = heads().train()
    sum(z.square().sum() for z in h(torch.randn(3, 7, 16)).values()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in h.parameters())


def test_probabilities_are_float32_under_mixed_precision_and_add_up_to_one():
    h = heads()
    x = torch.randn(2, 6, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = h(x)
    p = probabilities(logits)
    assert all(v.dtype == torch.float32 for v in p.values())
    assert all(torch.allclose(v.sum(-1), torch.ones(2, 6)) for v in p.values())
    p64 = probabilities({"x": torch.randn(3, 4, dtype=torch.float64)})
    assert p64["x"].dtype == torch.float64


def test_size_at_the_model_settings():
    h = Heads()
    assert sum(p.numel() for p in h.parameters()) == 2 * ((64 * 64 + 64) + (64 * 12 + 12))


def test_refuses_bad_inputs_and_settings():
    with pytest.raises(ModelError, match="read"):
        heads()(torch.randn(2, 15))
    for kw, match in (({"d_model": 0}, "d_model"), ({"hidden": -1}, "hidden"), ({"hidden": 2.0}, "hidden"),
                      ({"dropout": 1.0}, "dropout"), ({"horizons": ()}, "horizons"),
                      ({"horizons": ("now", "now")}, "horizons"), ({"horizons": ("later",)}, "horizons")):
        with pytest.raises(ConfigError, match=match):
            Heads(**kw)
