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
