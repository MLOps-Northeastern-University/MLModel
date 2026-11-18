"""Email and Slack alerts for failed runs and failed gates.

Copied from the data pipeline, with the same environment variables, so one
set of GitHub secrets serves both. Credentials come from environment
variables, never from the code or config. Alert failures are logged but never
crash the pipeline, so a broken webhook cannot hide the original error.
"""

import os
import smtplib
from email.message import EmailMessage

import requests

from model_pipeline.utils.logging_setup import get_logger

logger = get_logger("alerts")


def send_email(subject, body):
    to_addr, host = os.environ.get("ALERT_EMAIL"), os.environ.get("SMTP_HOST")
    if not (to_addr and host):
        logger.info("Email alert skipped: ALERT_EMAIL or SMTP_HOST not set")
        return False
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, to_addr
    msg["From"] = os.environ.get("SMTP_USER", to_addr)
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)), timeout=30) as smtp:
            smtp.starttls()
            if os.environ.get("SMTP_USER"):
                smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
        logger.info("Email alert sent to %s", to_addr)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email alert failed: %s", exc)
        return False


def send_slack(text):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        logger.info("Slack alert skipped: SLACK_WEBHOOK_URL not set")
        return False
    try:
        requests.post(url, json={"text": text}, timeout=15).raise_for_status()
        logger.info("Slack alert sent")
        return True
    except requests.RequestException as exc:
        logger.error("Slack alert failed: %s", exc)
        return False


def notify(subject, body, cfg=None):
    """Send an alert through every enabled channel."""
    settings = (cfg or {}).get("alerts", {"email_enabled": True, "slack_enabled": True})
    logger.warning("ALERT: %s\n%s", subject, body)
    if settings.get("email_enabled"):
        send_email(subject, body)
    if settings.get("slack_enabled"):
        send_slack(f"*{subject}*\n```{body}```")
