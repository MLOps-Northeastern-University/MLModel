import math

import numpy as np
import pytest
import torch
from sklearn.utils.class_weight import compute_class_weight

from scripts.core.errors import ConfigError, ModelError
from scripts.core.targets import TARGETS
from scripts.models.motor_causal_transformer.loss import EPS, MILLoss, as_floats, balanced_weights
from scripts.models.motor_causal_transformer.pooling import Pooled, hour_members

K = {"sleep": 2, "state": 7, "tremor": 3}
LABELS = ("sleep_now", "state_now", "tremor_now", "state_next_hour")


def batch(seed=0, b=3, dtype=torch.float32, requires_grad=False):
    """3 sequences of 30 windows, 3 graded hours of 10 windows, some unworn, some answers missing."""
    g = torch.Generator().manual_seed(seed)
    hour = torch.arange(30).div(10, rounding_mode="floor").repeat(b, 1)
    worn = torch.rand(b, 30, generator=g) > 0.2
    worn[1, 10:20] = False                                           # hour 1 of row 1: nothing worn
    members = hour_members(hour, worn, torch.tensor([[0, 1, 2]] * b))
    logits = {lab: torch.randn(b, 30, K[lab.split("_")[0]], generator=g, dtype=dtype).requires_grad_(requires_grad)
              for lab in LABELS}
    attention = {lab: torch.softmax(torch.randn(b, 3, K[lab.split("_")[0]], generator=g, dtype=dtype), -1)
                 .requires_grad_(requires_grad) for lab in LABELS}
    pooled = {lab: Pooled(attention[lab], attention[lab], members.to(dtype)) for lab in LABELS}
    targets = {}
    for lab in LABELS:
        y = torch.randint(0, K[lab.split("_")[0]], (b, 3), generator=g).to(dtype)
        y[0, 2] = math.nan                                           # no answer
        targets[lab] = y
    return pooled, logits, members, targets, members.any(-1)


def expected_distance(p, true, target):
    scores = TARGETS[target].scores
    if scores[true] is None:
        return 0.0
    keep = [i for i, s in enumerate(scores) if s is not None]
    q = p[keep] / p[keep].sum()
    return sum(float(q[j]) * abs(scores[i] - scores[true]) for j, i in enumerate(keep))


def reference(loss, pooled, logits, members, targets, has):
    """The loss written out plainly, one hour and one window at a time."""
    total = 0.0
    for label, y in targets.items():
        target = label.split("_")[0]
        cw = loss.class_weights(target)
        hour_sum = safety_sum = weights = 0.0
        for b in range(y.shape[0]):
            for j in range(y.shape[1]):
                if not has[b, j] or math.isnan(float(y[b, j])):
                    continue
                c = int(y[b, j])
                p = pooled[label].attention[b, j]
                hour = -math.log(max(float(p[c]), EPS)) + loss.lam * expected_distance(p, c, target)
                windows = torch.nonzero(members[b, j]).flatten().tolist()
                costs = []
                for t in windows:
                    q = torch.softmax(logits[label][b, t].double(), -1)
                    costs.append(-math.log(float(q[c])) + loss.lam * expected_distance(q, c, target))
                w = float(cw[c])
                hour_sum += w * hour
                safety_sum += w * sum(costs) / len(costs)
                weights += w
        if weights:
            per_output = loss.hour_weight * hour_sum / weights + loss.beta * safety_sum / weights
            total += loss.label_weights.get(label, 1.0) * per_output
    return total


@pytest.mark.parametrize("settings", [{}, {"lam": 0.0}, {"lam": 1.0, "beta": 0.5, "hour_weight": 0.3},
                                      {"label_weights": {"state_now": 2.0}}])
def test_matches_the_loss_written_out_plainly(settings):
    loss = MILLoss(**settings)
    loss.set_class_weights({"state": [0.5, 1, 2, 1, 3, 1, 0.7], "sleep": [1.5, 0.4]})
    args = batch()
    out = loss(*args)
    assert out.total.item() == pytest.approx(reference(loss, *args), rel=1e-5)
    floats = as_floats(out.parts)
    assert floats["state_now"]["hours"] == 3 * 3 - 1 - 1             # one unanswered, one not worn


def test_a_near_miss_costs_less_than_a_far_one_and_sleep_has_no_distance():
    loss = MILLoss()
    members = torch.ones(1, 1, 4, dtype=torch.bool)

    def cost(label, guess, answer, k):
        p = torch.full((1, 1, k), 0.01)
        p[0, 0, guess] = 1 - 0.01 * (k - 1)
        logits = {label: p.expand(1, 4, k).log()}
        return loss({label: Pooled(p, p, members.float())}, logits, members,
                    {label: torch.tensor([[float(answer)]])}, torch.ones(1, 1, dtype=torch.bool)).total.item()

    assert cost("state_now", 2, 3, 7) < cost("state_now", 6, 3, 7)    # one score off vs three
    assert cost("tremor_now", 1, 0, 3) < cost("tremor_now", 2, 0, 3)
    no_lam = MILLoss(lam=0.0)
    p = torch.tensor([[[0.3, 0.7]]])
    args = ({"sleep_now": Pooled(p, p, members.float())}, {"sleep_now": p.expand(1, 4, 2).log()}, members,
            {"sleep_now": torch.tensor([[0.0]])}, torch.ones(1, 1, dtype=torch.bool))
    assert loss(*args).total.item() == pytest.approx(no_lam(*args).total.item())


def test_unanswered_and_unworn_hours_count_for_nothing():
    loss = MILLoss()
    pooled, logits, members, targets, has = batch(requires_grad=True)
    out = loss(pooled, logits, members, targets, has)
    out.total.backward()
    assert pooled["state_now"].attention.grad[0, 2].abs().sum() == 0      # no answer
    assert pooled["state_now"].attention.grad[1, 1].abs().sum() == 0      # nothing worn
    assert logits["state_now"].grad[1, 10:20].abs().sum() == 0            # its windows
    assert logits["state_now"].grad[0, ~members[0].any(0)].abs().sum() == 0   # unworn windows


def test_a_batch_with_no_answers_gives_exactly_zero_and_finite_gradients():
    loss = MILLoss()
    pooled, logits, members, targets, has = batch(requires_grad=True)
    targets = {k: torch.full_like(v, math.nan) for k, v in targets.items()}
    out = loss(pooled, logits, members, targets, has)
    assert out.total.item() == 0.0
    out.total.backward()
    assert all(torch.isfinite(v.grad).all() and v.grad.abs().sum() == 0 for v in logits.values())
    assert as_floats(out.parts)["sleep_now"] == {"hour": 0.0, "safety": 0.0, "hours": 0}
    assert MILLoss()(pooled, logits, members, {}, has).total.item() == 0.0


def test_class_weights_of_one_are_the_plain_mean_and_a_rare_class_counts_more():
    pooled, logits, members, targets, has = batch()
    plain = MILLoss()(pooled, logits, members, targets, has)
    ones = MILLoss()
    ones.set_class_weights({"state": torch.ones(7)})
    assert torch.equal(ones(pooled, logits, members, targets, has).total, plain.total)
    # Weighting the class of the worst-predicted hour up pulls the mean towards that hour.
    loss = MILLoss(beta=0.0)
    y = targets["tremor_now"]
    p = pooled["tremor_now"].attention
    cost = -p.gather(-1, torch.where(y.isnan(), 0, y).long().unsqueeze(-1)).squeeze(-1).log()
    cost[y.isnan() | ~has] = -1
    worst = int(y.flatten()[cost.flatten().argmax()])
    before = loss(pooled, logits, members, {"tremor_now": y}, has).parts["tremor_now"].hour
    w = torch.ones(3)
    w[worst] = 5.0
    loss.set_class_weights({"tremor": w})
    after = loss(pooled, logits, members, {"tremor_now": y}, has).parts["tremor_now"].hour
    assert after > before


def test_balanced_weights_are_scikit_learns_and_capped():
    counts = np.array([50, 30, 15, 5])
    y = np.repeat(np.arange(4), counts)
    want = compute_class_weight("balanced", classes=np.arange(4), y=y)
    torch.testing.assert_close(balanced_weights(counts, cap=100.0), torch.tensor(want, dtype=torch.float32))
    w = balanced_weights(counts, cap=3.0)
    assert w.max() / w.min() == pytest.approx(3.0)
    freq = counts / counts.sum()
    assert float((torch.tensor(freq, dtype=torch.float32) * w).sum()) == pytest.approx(1.0)   # averages 1
    torch.testing.assert_close(balanced_weights(counts, power=0.0), torch.ones(4))
    assert torch.isfinite(balanced_weights([10, 0, 3])).all()                 # a class with no hours


def test_gradcheck_in_float64():
    loss = MILLoss()
    loss.set_class_weights({"state": torch.linspace(0.5, 2, 7)})
    pooled, logits, members, targets, has = batch(dtype=torch.float64, b=2)

    def f(window_logits, attention):
        p = {**pooled, "state_now": Pooled(attention, attention, members.double())}
        return loss(p, {**logits, "state_now": window_logits}, members, targets, has).total

    attention = pooled["state_now"].attention.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (logits["state_now"].clone().requires_grad_(True), attention))


def test_runs_in_float32_under_mixed_precision():
    loss = MILLoss()
    pooled, logits, members, targets, has = batch()
    want = loss(pooled, logits, members, targets, has).total
    half = {k: Pooled(v.attention.bfloat16(), v.mean, v.weights) for k, v in pooled.items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = loss(half, {k: v.bfloat16() for k, v in logits.items()}, members, targets, has).total
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, want, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu():
    loss = MILLoss()
    loss.set_class_weights({"tremor": [0.5, 1.0, 3.0]})
    pooled, logits, members, targets, has = batch()
    want = loss(pooled, logits, members, targets, has).total
    gpu = lambda d: {k: (Pooled(*(x.cuda() for x in v)) if isinstance(v, Pooled) else v.cuda())   # noqa: E731
                     for k, v in d.items()}
    out = loss.cuda()(gpu(pooled), gpu(logits), members.cuda(), gpu(targets), has.cuda()).total
    torch.testing.assert_close(out.cpu(), want)


def test_class_weights_are_saved_and_the_distances_are_not():
    loss = MILLoss()
    loss.set_class_weights({"state": torch.arange(1.0, 8.0)})
    state = loss.state_dict()
    assert sorted(state) == ["class_weights_sleep", "class_weights_state", "class_weights_tremor"]
    fresh = MILLoss()
    fresh.load_state_dict(state)
    assert torch.equal(fresh.class_weights("state"), torch.arange(1.0, 8.0))


def test_refuses_bad_settings_and_inputs():
    for kw, match in (({"lam": -1}, "lam"), ({"beta": math.inf}, "beta"), ({"hour_weight": True}, "hour_weight"),
                      ({"label_weights": {"mood_now": 1.0}}, "unknown label"),
                      ({"label_weights": {"state_now": -1.0}}, "state_now")):
        with pytest.raises(ConfigError, match=match):
            MILLoss(**kw)
    loss = MILLoss()
    for w, match in (({"mood": [1.0]}, "unknown target"), ({"sleep": [1.0, 0.0]}, "2 positive"),
                     ({"state": [1.0] * 6}, "7 positive")):
        with pytest.raises(ConfigError, match=match):
            loss.set_class_weights(w)
    for counts, kw, match in (([[1, 2]], {}, "counts"), ([1, -2], {}, "counts"), ([1, 2], {"power": -1}, "power"),
                              ([1, 2], {"cap": 0.5}, "cap")):
        with pytest.raises(ConfigError, match=match):
            balanced_weights(counts, **kw)
    pooled, logits, members, targets, has = batch()
    with pytest.raises(ConfigError, match="unknown label"):
        loss(pooled, logits, members, {"mood_now": targets["state_now"]}, has)
    with pytest.raises(ModelError, match="no predictions"):
        loss({}, logits, members, targets, has)
    with pytest.raises(ModelError, match="targets must be"):
        loss(pooled, logits, members, {"state_now": targets["state_now"][:, :2]}, has)
    with pytest.raises(ModelError, match="members"):
        loss(pooled, logits, members, targets, has[:, :2])


def test_sample_weights_scale_a_batch_so_low_weight_batches_weigh_less():
    """The bug this guards against: dividing by the batch's own weighted sum made every batch weigh the same."""
    loss = MILLoss()
    pooled, logits, members, targets, has = batch()
    plain = loss(pooled, logits, members, targets, has).total
    assert torch.equal(loss(pooled, logits, members, targets, has, torch.ones(3, 3)).total, plain)
    for w in (0.5, 2.0):
        scaled = loss(pooled, logits, members, targets, has, torch.full((3, 3), w)).total
        assert torch.allclose(scaled, w * plain, rtol=1e-6)


def test_an_hour_of_weight_zero_learns_nothing():
    loss = MILLoss()
    pooled, logits, members, targets, has = batch(requires_grad=True)
    weights = torch.ones(3, 3)
    weights[2, :] = 0.0                                              # row 2's hours all weigh 0
    loss(pooled, logits, members, targets, has, weights).total.backward()
    for label in LABELS:
        assert torch.count_nonzero(logits[label].grad[2]) == 0
        assert torch.count_nonzero(pooled[label].attention.grad[2]) == 0
        assert torch.count_nonzero(logits[label].grad[0]) > 0


def test_sample_weights_must_match_the_hours():
    pooled, logits, members, targets, has = batch()
    with pytest.raises(ModelError, match="sample_weights"):
        MILLoss()(pooled, logits, members, targets, has, torch.ones(3, 2))
