"""config/retrain.yaml: a PR bumps request to ask CI to train; anything else trains nothing."""

import json

import pytest

from model_pipeline import retrain_request
from model_pipeline.retrain_request import RetrainRequestError, dag_conf, read

KNOWN = ["window_logistic", "window_boosting", "motor_causal_transformer"]


def request(number=2, reason="deeper encoder", candidates=("motor_causal_transformer",), tune=True):
    return (f"request: {number}\nreason: {json.dumps(reason)}\n"
            f"candidates: {json.dumps(list(candidates))}\ntune: {str(tune).lower()}\n")


def test_the_repos_own_file_is_valid():
    assert retrain_request.main(["check"]) == 0


def test_a_bumped_request_becomes_the_dags_config():
    now = read(request(2), KNOWN)
    assert dag_conf(now, {"request": 1}) == {
        "dag_conf": {"candidates": ["motor_causal_transformer"], "tune": True, "reason": "deeper encoder",
                     "release": True},
        "compute": "gpu", "machine_type": "", "zone": ""}


def test_an_unchanged_request_trains_nothing():
    assert dag_conf(read(request(2), KNOWN), {"request": 2}) is None


def test_the_commit_that_adds_the_file_trains_nothing():
    assert dag_conf(read(request(1), KNOWN), None) is None


def test_empty_candidates_and_no_reason_are_allowed():
    now = read("request: 3\n", KNOWN)
    assert now == {"request": 3, "reason": "", "candidates": [], "tune": False, "compute": "gpu",
                   "machine_type": "", "zone": ""}


def test_the_machine_can_be_chosen():
    now = read("request: 2\ncompute: cpu\nmachine_type: e2-standard-16\nzone: us-central1-b\n", KNOWN)
    assert dag_conf(now, {"request": 1})["compute"] == "cpu"
    assert (now["machine_type"], now["zone"]) == ("e2-standard-16", "us-central1-b")


@pytest.mark.parametrize("text, match", [
    (request(candidates=["nope"]), "not in models.candidates"),
    (request(number=0), "whole number"),
    ("request: two\n", "whole number"),
    ("request: 2\ntune: maybe\n", "true or false"),
    ("request: 2\ncandidate: [window_logistic]\n", "unknown key"),
    ("- request\n", "mapping"),
    ("request: 2\ncompute: tpu\n", "compute must be one of"),
])
def test_a_malformed_file_is_refused(text, match):
    with pytest.raises(RetrainRequestError, match=match):
        read(text, KNOWN)


def test_conf_prints_the_config_only_when_the_request_changed(tmp_path, capsys):
    now, before = tmp_path / "now.yaml", tmp_path / "before.yaml"
    now.write_text(request(2), encoding="utf-8")
    before.write_text(request(1), encoding="utf-8")
    assert retrain_request.main(["conf", "--file", str(now), "--before", str(before)]) == 0
    assert json.loads(capsys.readouterr().out)["dag_conf"]["candidates"] == ["motor_causal_transformer"]
    before.write_text(request(2), encoding="utf-8")
    assert retrain_request.main(["conf", "--file", str(now), "--before", str(before)]) == 0
    assert capsys.readouterr().out == ""


def test_check_fails_on_a_bad_file(tmp_path):
    bad = tmp_path / "retrain.yaml"
    bad.write_text("request: 2\ncandidates: [nope]\n", encoding="utf-8")
    assert retrain_request.main(["check", "--file", str(bad)]) == 1
