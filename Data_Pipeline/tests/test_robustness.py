"""Robustness: odd file formats, partial failures, bad config, and secrets kept out of logs."""

import io
import logging
import smtplib
import zipfile

import pandas as pd
import pytest

from scripts import alerts, pipeline_steps
from scripts.diary import DiaryError, load_diary, parse_diary
from scripts.features import ParticipantArchive, wrist_features
from scripts.patient_info import load_demographics
from scripts.utils.config import load_config
from scripts.utils.file_io import read_json, read_text_csv
from tests.helpers import csv_bytes, default_hours, diary_columns, make_config, signal


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def diary_text(sep=";"):
    rows = [row for row, _ in default_hours("COPS-1")]
    return pd.DataFrame(rows, columns=diary_columns()).to_csv(sep=sep, index=False)


# ---------- file formats ----------

@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_a_diary_parses_in_any_common_encoding(cfg, encoding):
    text = diary_text().replace("Levodopa", "Lévodopa").replace("Carbidopa", "Carbidopä")
    diary, _ = parse_diary(load_diary(io.BytesIO(text.encode(encoding))), cfg)
    assert len(diary) == 7
    assert diary["medication"].dropna().iloc[0] == "Lévodopa / Carbidopä"  # text survives intact


def test_a_diary_with_the_wrong_separator_says_so(cfg):
    with pytest.raises(DiaryError, match="missing columns"):
        parse_diary(load_diary(io.BytesIO(diary_text(sep=",").encode())), cfg)


def test_a_diary_with_no_rows_says_so(cfg):
    # Not a cryptic KeyError: the participant is reported with a reason, and the run carries on.
    header_only = ";".join(diary_columns()) + "\n"
    with pytest.raises(DiaryError, match="no rows"):
        parse_diary(load_diary(io.BytesIO(header_only.encode())), cfg)


def test_demographics_in_windows_encoding_and_commas_are_read(cfg, tmp_path):
    path = tmp_path / "Demographics.csv"
    path.write_bytes("ID,Age,Sex,DBS\nCOPS-1,61,M,Nein\nCOPS-2,70,F,Ja – rechts\n".encode("cp1252"))
    demo = load_demographics(path, cfg)
    assert list(demo["participant_id"]) == ["COPS-1", "COPS-2"]
    assert demo["dbs"].iloc[1] == "Ja – rechts"


def test_read_text_csv_takes_a_path_or_a_file(tmp_path):
    path = tmp_path / "t.csv"
    path.write_bytes("a;b\n1;ü\n".encode("cp1252"))
    for source in [path, io.BytesIO(path.read_bytes())]:
        assert read_text_csv(source, sep=";").iloc[0]["b"] == "ü"


def _archive_with(tmp_path, name, payload):
    path = tmp_path / "COPS-1.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"COPS-1/Accelerometry/{name}", payload)
    return ParticipantArchive(path)


def test_a_sensor_zip_with_no_csv_inside_is_marked_corrupt(cfg, tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("readme.txt", "empty")
    with _archive_with(tmp_path, "h.zip", inner.getvalue()) as archive:
        assert wrist_features(archive, "h.zip", None, cfg)["status"] == "corrupt_file"


def test_a_sensor_csv_with_the_wrong_separator_is_marked_corrupt(cfg, tmp_path):
    payload = signal("still").to_csv(sep=",", index=False).encode()
    with _archive_with(tmp_path, "h.csv", payload) as archive:
        assert wrist_features(archive, None, "h.csv", cfg)["status"] == "corrupt_file"


def test_an_empty_sensor_csv_is_marked_corrupt(cfg, tmp_path):
    with _archive_with(tmp_path, "h.csv", b"") as archive:
        assert wrist_features(archive, None, "h.csv", cfg)["status"] == "corrupt_file"


def test_a_sensor_csv_with_only_a_header_has_too_little_data(cfg, tmp_path):
    with _archive_with(tmp_path, "h.csv", b"Time;X;Y;Z;Photo;Temp\n") as archive:
        assert wrist_features(archive, None, "h.csv", cfg)["status"] == "insufficient_data"


def test_a_sensor_csv_cut_off_mid_row_still_gives_features(cfg, tmp_path):
    payload = csv_bytes(signal("moving"))
    with _archive_with(tmp_path, "h.csv", payload[: len(payload) - 20]) as archive:
        assert wrist_features(archive, None, "h.csv", cfg)["status"] == "ok"


# ---------- partial failures (the DAG's check_failures task) ----------

@pytest.fixture
def sent(monkeypatch):
    record = []
    monkeypatch.setattr(pipeline_steps, "notify", lambda subject, body, cfg=None: record.append(subject))
    return record


def manifest(n):
    return {"participants": [{"name": f"COPS-{i}.zip"} for i in range(1, n + 1)]}


def result(pid):
    return {"participant_id": pid, "table": f"{pid}.csv", "issues": f"{pid}.json"}


def test_one_failed_participant_does_not_stop_the_run(cfg, sent):
    results = [result(f"COPS-{i}") for i in range(1, 10)]  # COPS-10 failed: 1 of 10
    kept = pipeline_steps.step_check_failures(cfg, manifest(10), results)
    assert len(kept) == 9
    failed = read_json(cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")
    assert [f["participant_id"] for f in failed] == ["COPS-10"]
    assert sent == ["[COPS pipeline] 1 of 10 participants failed"]


def test_too_many_failed_participants_stop_the_run(cfg, sent):
    with pytest.raises(pipeline_steps.TooManyFailures, match="3 of 10"):
        pipeline_steps.step_check_failures(cfg, manifest(10), [result(f"COPS-{i}") for i in range(1, 8)])


def test_when_everyone_failed_the_run_stops(cfg, sent):
    with pytest.raises(pipeline_steps.TooManyFailures):
        pipeline_steps.step_check_failures(cfg, manifest(3), [None, None, None])


def test_no_failures_means_no_alert(cfg, sent):
    kept = pipeline_steps.step_check_failures(cfg, manifest(2), [result("COPS-1"), result("COPS-2")])
    assert len(kept) == 2 and sent == []
    assert read_json(cfg["resolved_paths"]["reports_dir"] / "failed_participants.json") == []


# ---------- configuration ----------

def test_a_missing_config_file_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_the_config_file_can_be_chosen_by_environment_variable(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(open(load_config.__globals__["DEFAULT_CONFIG_PATH"], encoding="utf-8").read()
                      .replace("sample_size: 5", "sample_size: 2"), encoding="utf-8")
    monkeypatch.setenv("COPS_CONFIG", str(custom))
    assert load_config()["data"]["sample_size"] == 2


def test_overrides_change_only_what_they_name(tmp_path):
    base = make_config(tmp_path)
    changed = make_config(tmp_path, download={"max_retries": 9})
    assert changed["download"]["max_retries"] == 9
    assert changed["download"]["timeout_seconds"] == base["download"]["timeout_seconds"]


# ---------- secrets ----------

def test_the_mail_password_never_reaches_the_logs(monkeypatch):
    for name, value in {"ALERT_EMAIL": "team@example.com", "SMTP_HOST": "smtp.example.com",
                        "SMTP_USER": "sender@example.com", "SMTP_PASSWORD": "s3cret-app-pw",
                        "SLACK_WEBHOOK_URL": "https://hooks.slack.example/T1/s3cret-token"}.items():
        monkeypatch.setenv(name, value)

    class RefusesLogin:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    def slack_down(*args, **kwargs):
        raise alerts.requests.ConnectionError("no route")

    monkeypatch.setattr(alerts.smtplib, "SMTP", RefusesLogin)
    monkeypatch.setattr(alerts.requests, "post", slack_down)
    # The pipeline's loggers don't pass records up, so listen on the alerts logger itself.
    logged = io.StringIO()
    handler = logging.StreamHandler(logged)
    alerts.logger.addHandler(handler)
    both_on = {"alerts": {"email_enabled": True, "slack_enabled": True}}
    try:
        alerts.notify("[COPS pipeline] Run failed", "details", both_on)
    finally:
        alerts.logger.removeHandler(handler)

    text = logged.getvalue()
    assert "Email alert failed" in text and "Slack alert failed" in text
    assert "s3cret" not in text
