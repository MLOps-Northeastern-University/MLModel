import numpy as np
import pytest
import torch
from torch import nn

from model_pipeline.core.errors import ConfigError, ModelError, NotFittedError
from model_pipeline.models.motor_causal_transformer import movement
from model_pipeline.models.motor_causal_transformer.movement import (
    ENCODERS, Harnet10Projection, MovementEncoder, build_movement_encoder, describe, register)


def features(n=500, seed=0):
    return np.random.default_rng(seed).normal(3.0, 2.0, (n, 1024))


def fitted(**params):
    torch.manual_seed(0)
    enc = build_movement_encoder("harnet10", params)
    enc.fit(features())
    return enc.eval()


def test_harnet10_is_registered_and_built_from_config():
    assert ENCODERS["harnet10"] is Harnet10Projection
    enc = build_movement_encoder("harnet10", {"output_dim": 32, "hidden_dim": 16})
    assert (enc.input_group, enc.input_shape, enc.output_dim) == ("harnet", (1024,), 32)
    assert "harnet10: harnet (1024) -> 32" in describe(enc)


def test_standardisation_is_fitted_from_training_windows_in_chunks():
    x = features(1000)
    one, chunked = build_movement_encoder("harnet10"), build_movement_encoder("harnet10")
    one.fit(x)
    chunked.fit([x[:100], x[100:650], torch.as_tensor(x[650:])])
    for enc in (one, chunked):
        standardised = torch.addcmul(enc.shift, torch.as_tensor(x, dtype=torch.float32), enc.scale)
        assert torch.allclose(standardised.mean(0), torch.zeros(1024), atol=1e-4)
        assert torch.allclose(standardised.std(0), torch.ones(1024), atol=1e-4)
    assert torch.equal(one.scale, chunked.scale) or torch.allclose(one.scale, chunked.scale, rtol=1e-6)


def test_a_feature_that_never_varies_is_centred_not_blown_up():
    x = features(100)
    x[:, 7] = 4.2
    enc = build_movement_encoder("harnet10")
    enc.fit(x)
    out = torch.addcmul(enc.shift, torch.as_tensor(x, dtype=torch.float32), enc.scale)
    assert torch.all(out[:, 7].abs() < 1e-3)


def test_output_shape_and_layernorm():
    enc = fitted()
    out = enc(torch.as_tensor(features(64, seed=1), dtype=torch.float32))
    assert out.shape == (64, 256)
    assert torch.allclose(out.mean(-1), torch.zeros(64), atol=1e-5)          # LayerNorm, gain 1 and bias 0 at start


def test_the_standardisation_is_saved_with_the_model():
    enc = fitted()
    copy = build_movement_encoder("harnet10")
    assert not copy.is_fitted
    copy.load_state_dict(enc.state_dict())
    assert copy.is_fitted
    x = torch.as_tensor(features(8, seed=2), dtype=torch.float32)
    assert torch.equal(copy.eval()(x), enc(x))


def test_dropout_only_while_training():
    enc = fitted()
    x = torch.as_tensor(features(8, seed=3), dtype=torch.float32)
    assert torch.equal(enc(x), enc(x))
    enc.train()
    assert not torch.allclose(enc(x), enc(x))


def test_refuses_use_before_fitting_and_bad_inputs():
    enc = build_movement_encoder("harnet10")
    with pytest.raises(NotFittedError):
        enc(torch.zeros(2, 1024))
    enc.fit(features(10))
    with pytest.raises(ModelError, match="reads"):
        enc(torch.zeros(2, 512))
    with pytest.raises(ModelError, match="reads"):
        enc(torch.zeros(1024))
    with pytest.raises(ModelError, match="at least 2"):
        build_movement_encoder("harnet10").fit(features(1))
    with pytest.raises(ModelError, match=r"\(N, 1024\)"):
        build_movement_encoder("harnet10").fit([features(5)[:, :100]])
    broken = features(5)
    broken[2, 3] = np.nan
    with pytest.raises(ModelError, match="finite"):
        build_movement_encoder("harnet10").fit(broken)
    with pytest.raises(ModelError, match="means and SDs"):
        build_movement_encoder("harnet10").set_standardisation(np.zeros(1024), -np.ones(1024))


@pytest.mark.parametrize("name, params, match", [
    ("harnet", {}, "must be one of"), ("harnet10", {"width": 3}, "unexpected"),
    ("harnet10", {"output_dim": 0}, "output_dim"), ("harnet10", {"hidden_dim": 2.0}, "hidden_dim"),
    ("harnet10", {"input_dropout": 1.0}, "input_dropout"), ("harnet10", {"dropout": -0.5}, "dropout"),
])
def test_refuses_bad_config(name, params, match):
    with pytest.raises(ConfigError, match=match):
        build_movement_encoder(name, params)


def test_another_encoder_can_be_registered_and_swapped_in(monkeypatch):
    """The swap the design is for: a new encoder by name, nothing else changed."""
    monkeypatch.setattr(movement, "ENCODERS", dict(ENCODERS))

    @movement.register
    class SignalMean(MovementEncoder):
        name, input_group, input_shape = "signal_mean_test", "signal", (3, 300)

        def __init__(self, output_dim=8):
            super().__init__()
            self.output_dim = output_dim
            self.linear = nn.Linear(3, output_dim)

        def forward(self, x):
            self.check_input(x)
            return self.linear(x.mean(-1))

    enc = movement.build_movement_encoder("signal_mean_test", {"output_dim": 5})
    assert enc.is_fitted and enc(torch.randn(4, 3, 300)).shape == (4, 5)
    with pytest.raises(ConfigError, match="already registered"):
        @movement.register
        class Clash(SignalMean):
            pass


def test_register_needs_a_name_and_input_group():
    class Nameless(MovementEncoder):
        def forward(self, x):
            return x

    with pytest.raises(ConfigError, match="needs a name"):
        register(Nameless)


def test_fitting_in_blocks_matches_the_whole_and_is_fitted_survives_a_reload(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 3.0, size=(1000, 1024)).astype(np.float32)
    whole = Harnet10Projection()
    whole.fit(x)
    monkeypatch.setattr(Harnet10Projection, "FIT_BLOCK", 7)            # many small blocks
    blocks = Harnet10Projection()
    blocks.fit(x)
    assert torch.allclose(whole.scale, blocks.scale) and torch.allclose(whole.shift, blocks.shift)
    fresh = Harnet10Projection()
    assert not fresh.is_fitted
    fresh.load_state_dict(whole.state_dict())
    assert fresh.is_fitted
    fresh.load_state_dict(Harnet10Projection().state_dict())           # an unfitted state makes it unfitted
    assert not fresh.is_fitted


def test_the_variance_is_stable_for_large_offsets_and_constant_features(monkeypatch):
    rng = np.random.default_rng(1)
    x = (1e4 + rng.normal(0.0, 0.5, size=(5000, 1024))).astype(np.float32)   # big mean, small spread
    x[:, 3] = np.float32(4.2)
    monkeypatch.setattr(Harnet10Projection, "FIT_BLOCK", 333)                # uneven blocks
    enc = Harnet10Projection()
    enc.fit(x)
    expected = np.std(x.astype(np.float64), axis=0, ddof=1)
    got = 1.0 / enc.scale.numpy().astype(np.float64)
    np.testing.assert_allclose(np.delete(got, 3), np.delete(expected, 3), rtol=1e-4)
    assert enc.scale[3] == 1.0                                               # constant: only centred
    out = torch.addcmul(enc.shift, torch.as_tensor(x), enc.scale)
    assert torch.all(out[:, 3].abs() < 1e-5)
