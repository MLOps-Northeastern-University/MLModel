"""Alerts: sent when set up, skipped when not, and never able to crash the pipeline.

Email and Slack are faked, and every alert setting is cleared first, so no test
can send a real message, whatever is in the machine's environment.
"""

import smtplib

import pytest

from model_pipeline import alerts

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
    assert alerts.send_email("[Model pipeline] Bias gate failed", "recall gap 0.18 > 0.10")
    msg = smtp["sent"][0]
    assert msg["To"] == "team@example.com" and msg["From"] == "sender@example.com"
    assert msg["Subject"] == "[Model pipeline] Bias gate failed"
    assert "recall gap 0.18 > 0.10" in msg.get_content()
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
    assert alerts.send_slack("*Validation gate failed*")
    assert slack == [("https://hooks.slack.example/T1", {"text": "*Validation gate failed*"})]


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
