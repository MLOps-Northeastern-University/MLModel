"""Config, file helpers and alerts.

Email and Slack are faked, and every alert setting is cleared first, so no test
can send a real message, whatever is in the machine's environment.
"""

import smtplib

import numpy as np
import pytest

from scripts import utils
from tests.helpers import make_config

ALERT_SETTINGS = ["ALERT_EMAIL", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SLACK_WEBHOOK_URL"]
ON = {"alerts": {"email_enabled": True, "slack_enabled": True}}


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

    monkeypatch.setattr(utils.smtplib, "SMTP", FakeSMTP)
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

    monkeypatch.setattr(utils.requests, "post", post)
    return posts


def set_email(monkeypatch, user="sender@example.com"):
    monkeypatch.setenv("ALERT_EMAIL", "team@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    if user:
        monkeypatch.setenv("SMTP_USER", user)
        monkeypatch.setenv("SMTP_PASSWORD", "app-password")


# ---- config and files ----------------------------------------------------------------------

def test_config_resolves_and_creates_paths(tmp_path):
    cfg = make_config(tmp_path, data={"sample_size": 2})
    assert cfg["data"]["sample_size"] == 2 and cfg["data"]["mode"] == "list"  # override merged, rest kept
    assert cfg["resolved_paths"]["processed_dir"] == tmp_path / "data" / "processed"
    assert cfg["resolved_paths"]["processed_dir"].is_dir()


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        utils.load_config(tmp_path / "none.yaml")


def test_json_takes_numpy_values(tmp_path):
    values = {"n": np.int64(3), "x": np.float32(1.5), "gap": np.float32("nan"), "ok": np.bool_(True), "p": tmp_path}
    path = utils.write_json(values, tmp_path / "a.json")
    assert utils.read_json(path) == {"n": 3, "x": 1.5, "gap": None, "ok": True, "p": str(tmp_path)}
    with pytest.raises(TypeError):
        utils.write_json({"bad": object()}, tmp_path / "b.json")


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_text_files_in_either_encoding_are_read(tmp_path, encoding):
    path = tmp_path / "d.csv"
    path.write_bytes("drug;mg\nBenserazid ä;50\n".encode(encoding))
    assert utils.read_text_csv(path, sep=";")["drug"][0] == "Benserazid ä"


def test_doses_stay_text_when_read_back(tmp_path):
    import pandas as pd
    path = utils.write_table(pd.DataFrame({"dosage_mg": ["50", "50;50"]}), tmp_path / "t.csv")
    assert utils.read_table(path)["dosage_mg"].tolist() == ["50", "50;50"]


# ---- alerts --------------------------------------------------------------------------------

def test_nothing_is_sent_when_alerts_are_not_set_up(smtp, slack):
    assert utils.send_email("subject", "body") is False
    assert utils.send_slack("text") is False
    utils.notify("subject", "body", ON)
    assert smtp["sent"] == [] and slack == []


def test_an_email_alert_is_sent_securely(monkeypatch, smtp):
    set_email(monkeypatch)
    assert utils.send_email("[COPS pipeline] 2 of 5 participants failed", "COPS-2: bad zip")
    msg = smtp["sent"][0]
    assert msg["To"] == "team@example.com" and msg["From"] == "sender@example.com"
    assert "COPS-2: bad zip" in msg.get_content()
    assert smtp["server"] == ("smtp.example.com", 2525)
    assert smtp["tls"] and smtp["login"] == ("sender@example.com", "app-password")


def test_an_email_can_go_without_a_login(monkeypatch, smtp):
    set_email(monkeypatch, user=None)
    assert utils.send_email("subject", "body")
    assert smtp["login"] is None and smtp["sent"][0]["From"] == "team@example.com"


def test_a_failing_mail_server_does_not_raise(monkeypatch):
    set_email(monkeypatch)

    class Down:
        def __init__(self, *args, **kwargs):
            raise smtplib.SMTPConnectError(421, "unavailable")

    monkeypatch.setattr(utils.smtplib, "SMTP", Down)
    assert utils.send_email("subject", "body") is False


def test_a_slack_alert_is_posted(monkeypatch, slack):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")
    assert utils.send_slack("*Critical*")
    assert slack == [("https://hooks.slack.example/T1", {"text": "*Critical*"})]


def test_a_failing_slack_webhook_does_not_raise(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")

    def broken(*args, **kwargs):
        raise utils.requests.ConnectionError("no route")

    monkeypatch.setattr(utils.requests, "post", broken)
    assert utils.send_slack("text") is False


def test_notify_uses_only_the_channels_turned_on(monkeypatch, smtp, slack):
    set_email(monkeypatch)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/T1")
    utils.notify("subject", "body", {"alerts": {"email_enabled": False, "slack_enabled": True}})
    assert smtp["sent"] == [] and slack[0][1]["text"] == "*subject*\n```body```"
    utils.notify("subject", "body", {"alerts": {"email_enabled": True, "slack_enabled": False}})
    assert len(smtp["sent"]) == 1 and len(slack) == 1
