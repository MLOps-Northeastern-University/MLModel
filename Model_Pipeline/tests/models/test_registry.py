import json

import pytest

from scripts.core.errors import ConfigError, ModelError
from scripts.models import registry
from scripts.models.motor_causal_transformer.model import MotorCausalTransformer
from scripts.models.window_classifier import WindowBoosting, WindowLogistic
from scripts.utils.config import load_config
from tests.models.test_window_classifier import SPEC, dataset, environment  # noqa: F401  (the fake harnet10)

WINDOWS = {"spec_version": 1, "seconds": 10, "rate_hz": 30, "axes": 3}


def config(*entries, **section):
    return {"windows": WINDOWS, "models": {"targets": ["state"], "horizons": ["now"], **section,
                                           "candidates": list(entries)}}


def test_every_family_imports_as_its_own_class():
    for family in registry.FAMILIES:
        assert registry.family_class(family).family == family
    assert registry.family_class("window_boosting") is WindowBoosting


def test_the_shipped_config_builds_every_candidate():
    built = registry.candidates(load_config())
    assert [name for name, _ in built] == ["window_logistic", "window_boosting", "motor_causal_transformer",
                                           "motor_causal_transformer_no_pretrain"]
    assert [type(m) for _, m in built] == [WindowLogistic, WindowBoosting, MotorCausalTransformer,
                                           MotorCausalTransformer]
    assert built[3][1].params["pretrain"]["epochs"] == 0 and built[3][1].params["pretrain"]["lr"] > 0
    assert all(m.labels == ("sleep_now", "sleep_next_hour", "state_now", "state_next_hour", "tremor_now",
                            "tremor_next_hour") and not m.is_fitted for _, m in built)


def test_a_candidate_takes_its_params_and_may_set_its_own_targets():
    built = registry.candidates(config(
        {"name": "lr", "family": "window_logistic", "params": {"c": 0.1}},
        {"name": "lr_tremor", "family": "window_logistic", "targets": ["tremor"], "horizons": ["now", "next_hour"]}))
    (_, a), (_, b) = built
    assert a.params["c"] == 0.1 and a.labels == ("state_now",) and a.spec == SPEC
    assert b.params["c"] == 0.01 and b.labels == ("tremor_now", "tremor_next_hour")


@pytest.mark.parametrize("cfg, match", [
    ({"windows": WINDOWS}, "models section"),
    (config(), "at least one candidate"),
    (config({"name": "x", "family": "svm"}), "unknown model family 'svm'"),
    (config({"name": "x", "family": "window_logistic", "lr": 1}), "unknown key"),
    (config({"name": "Bad Name", "family": "window_logistic"}), "name must be"),
    (config({"name": "x"}), "needs a family"),
    (config({"name": "x", "family": "window_logistic"}, {"name": "x", "family": "window_boosting"}), "unique"),
    (config({"name": "x", "family": "window_logistic", "params": {"c": -1}}), "candidate 'x': c must"),
    (config({"name": "x", "family": "window_logistic", "params": {"depth": 3}}), "candidate 'x'.*no setting"),
    (config({"name": "x", "family": "window_logistic", "targets": []}), "targets must be"),
    (config({"name": "x", "family": "window_logistic", "targets": ["gait"]}), "candidate 'x'"),
    (config({"name": "x", "family": "window_logistic"}, seeds=[0]), "unknown key"),
    ({"models": config({"name": "x", "family": "window_logistic"})["models"]}, "window spec"),
])
def test_bad_config_is_refused_before_anything_trains(cfg, match):
    with pytest.raises(ConfigError, match=match):
        registry.candidates(cfg)


def test_load_opens_any_saved_model_by_its_manifest(tmp_path):
    data = dataset()
    model = WindowLogistic(["state"], ["now"], spec=SPEC).fit(data.subset(["A", "B"]), seed=0)
    model.save(tmp_path / "m")
    loaded = registry.load(tmp_path / "m")
    assert type(loaded) is WindowLogistic and loaded.is_fitted
    assert (loaded.predict(data).probs["state_now"] == model.predict(data).probs["state_now"]).all()


def test_load_refuses_a_folder_without_a_manifest_or_of_an_unknown_family(tmp_path):
    with pytest.raises(ModelError, match="no manifest.json"):
        registry.load(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"family": "svm"}), encoding="utf-8")
    with pytest.raises(ModelError, match="doesn't know"):
        registry.load(tmp_path)
