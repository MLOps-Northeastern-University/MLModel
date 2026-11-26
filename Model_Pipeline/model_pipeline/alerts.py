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
    """Email the alert to ALERT_EMAIL (one address, or several separated by commas) through SMTP_HOST.

    SMTP_PORT defaults to 587, the STARTTLS port. With SMTP_USER set it logs in with SMTP_PASSWORD (in CI,
    an app password from the GitHub secrets) and sends as that account; without it, the alert is sent
    from ALERT_EMAIL itself, with no login.

    Returns:
        bool: True if the server accepted the message, False if email isn't set up or sending failed.
    """
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
            # STARTTLS before the login, so the password never crosses the network in clear. A server that doesn't
            # offer it raises an SMTPException here, and the alert is skipped rather than sent unencrypted.
            smtp.starttls()
            if os.environ.get("SMTP_USER"):
                smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
        logger.info("Email alert sent to %s", to_addr)
        return True
    # A refused login, a server that's down or a timeout (OSError covers connection and DNS errors) is logged
    # and returns False: an alert that can't be sent must never replace the error it was reporting.
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email alert failed: %s", exc)
        return False


def send_slack(text):
    """Post the alert to the Slack incoming webhook in SLACK_WEBHOOK_URL, if it is set.

    The team uses email, and the CI workflow doesn't set the webhook, so this is normally skipped with a
    log line.

    Returns:
        bool: True if Slack accepted the post, False if no webhook is set or the post failed.
    """
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        logger.info("Slack alert skipped: SLACK_WEBHOOK_URL not set")
        return False
    try:
        # raise_for_status turns an error reply (a revoked or mistyped webhook, say) into a RequestException,
        # so it is logged like a network failure instead of passing as sent.
        requests.post(url, json={"text": text}, timeout=15).raise_for_status()
        logger.info("Slack alert sent")
        return True
    except requests.RequestException as exc:
        logger.error("Slack alert failed: %s", exc)
        return False


def notify(subject, body, cfg=None):
    """Send an alert through every enabled channel."""
    # Without a config, or with one that has no alerts section, both channels are tried; each still needs
    # its environment variables, so nothing is sent unless those are set.
    settings = (cfg or {}).get("alerts", {"email_enabled": True, "slack_enabled": True})
    # Logged first, at WARNING, so the alert is in the run's logs (CI, Cloud Logging) even when no channel is
    # set up or sending fails.
    logger.warning("ALERT: %s\n%s", subject, body)
    if settings.get("email_enabled"):
        send_email(subject, body)
    if settings.get("slack_enabled"):
        # The subject in bold and the body in a code block, so stack traces and metric lists keep their spacing.
        send_slack(f"*{subject}*\n```{body}```")
