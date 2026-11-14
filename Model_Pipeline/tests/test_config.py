"""The config loads, agrees with the phone model, and the run command goes through every step."""

import pytest

from scripts import run_model_pipeline
from scripts.utils.config import load_config
from tests.helpers import FEATURE_NAMES, make_config

# The app's HourFeatures.NAMES for feature spec 3, as the Backend's WRIST_FEATURES lists them.
PHONE_INPUTS = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]


def test_the_model_inputs_are_the_phones_in_the_same_order(tmp_path):
    cfg = make_config(tmp_path)
    assert cfg["data"]["features"] == PHONE_INPUTS == FEATURE_NAMES
    assert cfg["export"]["feature_spec_version"] == 3
    assert cfg["export"]["output_name"] == "off_risk"


def test_output_folders_are_made_but_the_input_table_is_only_read(tmp_path):
    cfg = make_config(tmp_path, features_csv=tmp_path / "not_pulled_yet.csv")
    assert cfg["resolved_paths"]["artifacts_dir"] == tmp_path / "artifacts"
    assert cfg["resolved_paths"]["artifacts_dir"].is_dir()
    assert not cfg["resolved_paths"]["features_csv"].exists()


def test_overrides_change_only_what_they_name(tmp_path):
    cfg = make_config(tmp_path, bias={"max_recall_gap": 0.2})
    assert cfg["bias"]["max_recall_gap"] == 0.2
    assert cfg["bias"]["min_slice_rows"] == 20


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_the_run_command_goes_through_every_step(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(run_model_pipeline, "load_config", lambda path=None: cfg)
    assert run_model_pipeline.main([]) == 0
