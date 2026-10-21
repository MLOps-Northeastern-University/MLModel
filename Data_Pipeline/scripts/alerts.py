"""Email and Slack alerts for failures and data anomalies.

Credentials come from environment variables (see .env.example), never
from the code or config. Alert failures are logged but never crash the
pipeline, so a broken webhook cannot hide the original error.
"""

import os
import smtplib
from email.message import EmailMessage

import requests

from scripts.utils.logging_setup import get_logger

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


def airflow_failure_callback(context):
    """Airflow on_failure_callback: alert with the DAG run and failed task."""
    ti = context.get("task_instance")
    subject = f"[COPS pipeline] Task failed: {getattr(ti, 'task_id', 'unknown')}"
    body = (f"DAG: {getattr(ti, 'dag_id', '?')}\n"
            f"Run: {context.get('run_id', '?')}\n"
            f"Task: {getattr(ti, 'task_id', '?')} (try {getattr(ti, 'try_number', '?')})\n"
            f"Error: {context.get('exception')}\n"
            f"Log: {getattr(ti, 'log_url', '')}")
    notify(subject, body)
