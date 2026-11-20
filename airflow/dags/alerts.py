"""The alert both DAGs send when a task fails: email and Slack, from the same settings as the pipelines.

A step also alerts the team itself when it stops on purpose (a failed gate, too many failed
participants), with the reason. This one covers every failure, including a container that never
started, and links to the task's log.

    ALERT_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD   email (skipped unless the first two are set)
    SLACK_WEBHOOK_URL                                              Slack (skipped unless set)
"""

import logging
import os
import smtplib
from email.message import EmailMessage

import requests

logger = logging.getLogger("mednova.alerts")


def send_email(subject, body):
    to_addr, host = os.environ.get("ALERT_EMAIL"), os.environ.get("SMTP_HOST")
    if not (to_addr and host):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["To"], msg["From"] = subject, to_addr, os.environ.get("SMTP_USER", to_addr)
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)), timeout=30) as smtp:
            smtp.starttls()
            if os.environ.get("SMTP_USER"):
                smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email alert failed: %s", exc)
        return False


def send_slack(text):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    try:
        requests.post(url, json={"text": text}, timeout=15).raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Slack alert failed: %s", exc)
        return False


def task_failed(context):
    """on_failure_callback: which DAG, run and task failed, why, and where its log is."""
    ti = context.get("task_instance")
    subject = f"[MedNOVA {getattr(ti, 'dag_id', 'pipeline')}] Task failed: {getattr(ti, 'task_id', 'unknown')}"
    body = (f"DAG: {getattr(ti, 'dag_id', '?')}\n"
            f"Run: {context.get('run_id', '?')}\n"
            f"Task: {getattr(ti, 'task_id', '?')} (try {getattr(ti, 'try_number', '?')})\n"
            f"Error: {context.get('exception')}\n"
            f"Log: {getattr(ti, 'log_url', '')}")
    logger.warning("ALERT: %s\n%s", subject, body)
    send_email(subject, body)
    send_slack(f"*{subject}*\n```{body}```")
