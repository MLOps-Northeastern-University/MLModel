"""Shared helpers: config, file I/O, logging and alerts.

Alert credentials come from environment variables (see .env.example), never from
the code or config. A failed alert is logged but never raises, so a broken webhook
can't hide the error it was reporting.
"""

import copy
import io
import json
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

# ---- config ------------------------------------------------------------------------------


def load_config(path=None, overrides=None):
    """Read the YAML config (path, else $COPS_CONFIG, else config/pipeline.yaml).

    overrides is a nested dict merged on top. cfg["resolved_paths"] holds the
    paths as absolute Paths, and their folders are created.
    """
    config_path = Path(path or os.environ.get("COPS_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    root = Path(cfg.get("project_root", PROJECT_ROOT))
    cfg["resolved_paths"] = {key: Path(value) if Path(value).is_absolute() else root / value
                             for key, value in cfg["paths"].items()}
    for key, value in cfg["resolved_paths"].items():
        (value.parent if key == "log_file" else value).mkdir(parents=True, exist_ok=True)
    return cfg


def _deep_merge(base, extra):
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

# ---- files -------------------------------------------------------------------------------


# Text even when every value looks like a number ("50" in one file, "50;50" in another).
TEXT_COLUMNS = {"dosage_mg": str, "medication": str}


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_table(path):
    return pd.read_csv(path, dtype=TEXT_COLUMNS)


def read_text_csv(source, **kwargs):
    """pd.read_csv for the dataset's text files: UTF-8 (with or without a BOM), else cp1252, else Latin-1."""
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), **kwargs)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)}")

# ---- logging -----------------------------------------------------------------------------


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def get_logger(name):
    """A logger writing to the console and logs/pipeline.log.

    The file isn't rotated: participants log from several processes at once, and
    rotating a file that several processes write to loses lines.
    """
    global _configured
    if not _configured:
        root = logging.getLogger("cops")
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(_FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
        path = PROJECT_ROOT / "logs" / "pipeline.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(formatter)
            root.addHandler(handler)
        except OSError as exc:  # a read-only file system shouldn't stop the pipeline
            root.warning("Could not open log file %s: %s", path, exc)
        root.propagate = False
        _configured = True
    return logging.getLogger(f"cops.{name}")


logger = get_logger("alerts")

# ---- alerts ------------------------------------------------------------------------------


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


def notify(subject, body, cfg):
    """Send an alert through every channel turned on in cfg["alerts"]."""
    settings = cfg.get("alerts", {})
    logger.warning("ALERT: %s\n%s", subject, body)
    if settings.get("email_enabled"):
        send_email(subject, body)
    if settings.get("slack_enabled"):
        send_slack(f"*{subject}*\n```{body}```")
