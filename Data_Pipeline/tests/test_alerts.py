"""Alerts: sent when set up, skipped when not, and never able to crash the pipeline.

Email and Slack are faked, and every alert setting is cleared first, so no test
can send a real message, whatever is in the machine's environment.
"""

import smtplib

import pytest

from scripts import alerts

ALERT_SETTINGS = ["ALERT_EMAIL", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                  "SLACK_WEBHOOK_URL"]


@pytest.fixture(autouse=True)
def no_real_settings(monkeypatch):
    for name in ALERT_SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def smtp(monkeypatch):
    """A fake SMTP server that records what it's asked to do."""
    record = {"sent": [], "login": None, "tls": False, "server": None}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            record["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            record["tls"] = True

        def login(self, user, password):
            record["login"] = (user, password)

        def send_message(self, msg):
            record["sent"].append(msg)

    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    return record


@pytest.fixture
def slack(monkeypatch):
    posts = []

    class Reply:
        def raise_for_status(self):
            pass

    def post(url, json=None, timeout=None):
        posts.append((url, json))
        return Reply()

    monkeypatch.setattr(alerts.requests, "post", post)
    return posts


def set_email(monkeypatch, user="sender@example.com"):
    monkeypatch.setenv("ALERT_EMAIL", "team@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    if user:
        monkeypatch.setenv("SMTP_USER", user)
        monkeypatch.setenv("SMTP_PASSWORD", "app-password")


def test_nothing_is_sent_when_alerts_are_not_set_up(smtp, slack):
    assert alerts.send_email("subject", "body") is False
    assert alerts.send_slack("text") is False
    alerts.notify("subject", "body")  # and notify doesn't fail either
    assert smtp["sent"] == [] and slack == []


def test_an_email_alert_is_sent_securely(monkeypatch, smtp):
    set_email(monkeypatch)
    assert alerts.send_email("[COPS pipeline] 2 of 5 participants failed", "COPS-2: bad zip")
    msg = smtp["sent"][0]
    assert msg["To"] == "team@example.com" and msg["From"] == "sender@example.com"
    assert msg["Subject"] == "[COPS pipeline] 2 of 5 participants failed"
    assert "COPS-2: bad zip" in msg.get_content()
    assert smtp["server"] == ("smtp.example.com", 2525)
    assert smtp["tls"] and smtp["login"] == ("sender@example.com", "app-password")


def test_an_email_can_go_without_a_login(monkeypatch, smtp):
    set_email(monkeypatch, user=None)
    assert alerts.send_email("subject", "body")
    assert smtp["login"] is None
    assert smtp["sent"][0]["From"] == "team@example.com"


def test_a_failing_mail_server_does_not_raise(monkeypatch):
    set_email(monkeypatch)

    class Down:
        def __init__(self, *args, **kwargs):
            raise smtplib.SMTPConnectError(421, "unavailable")

    monkeypatch.setattr(alerts.smtplib, "SMTP", Down)
    assert alerts.send_email("subject", "body") is False


def test_a_slack_alert_is_posted(monkeypatch, slack):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")
    assert alerts.send_slack("*Critical anomalies*")
    assert slack == [("https://hooks.slack.example/T1", {"text": "*Critical anomalies*"})]


def test_a_failing_slack_webhook_does_not_raise(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")

    def broken(*args, **kwargs):
        raise alerts.requests.ConnectionError("no route")

    monkeypatch.setattr(alerts.requests, "post", broken)
    assert alerts.send_slack("text") is False


def test_notify_uses_only_the_channels_turned_on(monkeypatch, smtp, slack):
    set_email(monkeypatch)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")

    alerts.notify("subject", "body", {"alerts": {"email_enabled": False, "slack_enabled": True}})
    assert smtp["sent"] == [] and len(slack) == 1
    assert slack[0][1]["text"] == "*subject*\n```body```"

    alerts.notify("subject", "body", {"alerts": {"email_enabled": True, "slack_enabled": False}})
    assert len(smtp["sent"]) == 1 and len(slack) == 1


def test_an_airflow_task_failure_is_alerted_with_where_it_failed(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "notify", lambda subject, body, cfg=None: sent.append((subject, body)))

    class TaskInstance:
        task_id, dag_id, try_number = "process_participant", "cops_data_pipeline", 2
        log_url = "http://localhost:8080/log"

    alerts.airflow_failure_callback({"task_instance": TaskInstance(), "run_id": "manual__1",
                                     "exception": ValueError("No symptom diary")})

    subject, body = sent[0]
    assert subject == "[COPS pipeline] Task failed: process_participant"
    for part in ["cops_data_pipeline", "manual__1", "try 2", "No symptom diary", "http://localhost:8080/log"]:
        assert part in body
