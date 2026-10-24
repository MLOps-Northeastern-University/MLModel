import zipfile

import pytest

from scripts import run_pipeline
from scripts.utils.file_io import read_json
from tests.helpers import make_config, make_participant_zip


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


@pytest.fixture
def alerts(monkeypatch):
    """Record alerts instead of sending email or Slack messages."""
    sent = []
    monkeypatch.setattr(run_pipeline, "notify", lambda subject, body, cfg=None: sent.append(subject))
    return sent


def failed_report(cfg):
    return read_json(cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")


def test_failed_participant_is_reported_not_hidden(cfg, tmp_path, alerts):
    good = make_participant_zip(tmp_path, "COPS-1")
    bad = tmp_path / "COPS-2.zip"
    with zipfile.ZipFile(bad, "w") as z:  # A participant zip with no diary inside.
        z.writestr("COPS-2/readme.txt", "no diary")
    entries = [{"name": p.name, "local_path": str(p)} for p in [good, bad]]

    results, failures = run_pipeline.process_participants(cfg, entries)

    assert [r["participant_id"] for r in results] == ["COPS-1"]
    assert failures[0]["participant_id"] == "COPS-2"
    assert "No symptom diary" in failures[0]["error"]
    assert run_pipeline.report_failures(cfg, failures, total=2)  # 50% is over the 20% limit
    assert failed_report(cfg)[0]["participant_id"] == "COPS-2"
    assert len(alerts) == 1


def test_no_failures_writes_empty_report(cfg, alerts):
    assert not run_pipeline.report_failures(cfg, [], total=5)
    assert failed_report(cfg) == []
    assert alerts == []


def test_few_failures_alert_but_continue(cfg, alerts):
    failures = [{"participant_id": "COPS-1", "error": "DiaryError: bad file"}]
    assert not run_pipeline.report_failures(cfg, failures, total=10)  # 10% is under the limit
    assert len(alerts) == 1


def test_participants_in_parallel_match_one_at_a_time(tmp_path, alerts):
    zips = [make_participant_zip(tmp_path, f"COPS-{n}") for n in (3, 1, 2)]
    entries = [{"name": p.name, "local_path": str(p)} for p in zips]

    one = make_config(tmp_path / "one", run={"parallel_participants": 1})
    many = make_config(tmp_path / "many", run={"parallel_participants": 3})
    one_results, _ = run_pipeline.process_participants(one, entries)
    many_results, failures = run_pipeline.process_participants(many, entries)

    assert failures == []
    # Same participants, in the manifest's order, with the same features.
    assert [r["participant_id"] for r in many_results] == ["COPS-3", "COPS-1", "COPS-2"]
    for a, b in zip(one_results, many_results):
        assert open(a["table"], encoding="utf-8").read() == open(b["table"], encoding="utf-8").read()


def test_a_failure_in_a_worker_process_is_reported(tmp_path, alerts):
    good = make_participant_zip(tmp_path, "COPS-1")
    bad = tmp_path / "COPS-2.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("COPS-2/readme.txt", "no diary")
    entries = [{"name": p.name, "local_path": str(p)} for p in [good, bad]]

    results, failures = run_pipeline.process_participants(
        make_config(tmp_path, run={"parallel_participants": 2}), entries)

    assert [r["participant_id"] for r in results] == ["COPS-1"]
    assert failures[0]["participant_id"] == "COPS-2"
    assert "No symptom diary" in failures[0]["error"]
