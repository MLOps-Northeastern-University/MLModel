"""The config loads and agrees with the phone model, and a run can choose which candidates it trains."""

import pytest

from model_pipeline.utils.config import choose_candidates, load_config
from tests.helpers import FEATURE_NAMES, make_config

# The app's HourFeatures.NAMES for feature spec 3, as the Backend's WRIST_FEATURES lists them.
PHONE_INPUTS = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]


def test_the_model_inputs_are_the_phones_in_the_same_order(tmp_path):
    # The order is compared as well as the names, so a reordered list in model.yaml fails here rather than
    # pairing the phone's values with the wrong features.
    cfg = make_config(tmp_path)
    assert cfg["data"]["features"] == PHONE_INPUTS == FEATURE_NAMES


def test_output_folders_are_made_but_the_input_table_is_only_read(tmp_path):
    # features_csv names a file that doesn't exist, as before dvc pull: loading must neither fail nor create
    # it, while the output folders are made.
    cfg = make_config(tmp_path, features_csv=tmp_path / "not_pulled_yet.csv")
    assert cfg["resolved_paths"]["artifacts_dir"] == tmp_path / "artifacts"
    assert cfg["resolved_paths"]["artifacts_dir"].is_dir()
    assert not cfg["resolved_paths"]["features_csv"].exists()


def test_overrides_change_only_what_they_name(tmp_path):
    cfg = make_config(tmp_path, bias={"max_recall_gap": 0.2})
    assert cfg["bias"]["max_recall_gap"] == 0.2
    # min_slice_rows is model.yaml's: an override merges into its section instead of replacing it.
    assert cfg["bias"]["min_slice_rows"] == 20


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# ---- choosing the candidates a run trains ----------------------------------------------------

def names(cfg):
    """The names of the candidates the config trains, in order."""
    return [c["name"] for c in cfg["models"]["candidates"]]


def test_choosing_candidates_keeps_only_those_in_config_order(tmp_path):
    cfg = make_config(tmp_path)
    chosen = choose_candidates(cfg, ["motor_causal_transformer", "window_logistic"])
    assert names(chosen) == ["window_logistic", "motor_causal_transformer"]
    assert list(chosen["tune"]["candidates"]) == ["window_logistic"]       # the transformer isn't tuned
    assert len(names(cfg)) == 3                                           # the config itself is unchanged


def test_the_leakage_check_runs_on_a_chosen_candidate(tmp_path):
    cfg = make_config(tmp_path)
    # model.yaml runs the check on window_boosting. A run that trains it keeps that; a run that doesn't moves
    # the check to the first candidate it does train.
    assert cfg["train"]["leakage_check"]["candidates"] == ["window_boosting"]
    kept = choose_candidates(cfg, ["window_boosting", "window_logistic"])
    assert kept["train"]["leakage_check"]["candidates"] == ["window_boosting"]
    moved = choose_candidates(cfg, ["window_logistic"])
    assert moved["train"]["leakage_check"]["candidates"] == ["window_logistic"]


def test_a_leakage_check_turned_off_stays_off(tmp_path):
    cfg = make_config(tmp_path, train={"leakage_check": {"candidates": []}})
    assert choose_candidates(cfg, ["window_logistic"])["train"]["leakage_check"]["candidates"] == []


# Unknown names are checked before the count, so ["nope"] gets the "unknown candidate" message.
@pytest.mark.parametrize("chosen, match", [
    (["nope"], "unknown candidate"),
    ([], "once"),
    (["window_logistic", "window_logistic"], "once"),
])
def test_bad_candidate_names_are_refused(tmp_path, chosen, match):
    with pytest.raises(ValueError, match=match):
        choose_candidates(make_config(tmp_path), chosen)
