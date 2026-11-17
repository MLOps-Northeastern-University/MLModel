"""Data acquisition from the COPS project on OSF.

The pipeline never downloads the whole project at once. It lists the files
through the public OSF API, picks the participant zips to process, and
downloads them one at a time with retries and checksum verification.
"""

import hashlib
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from scripts.utils import get_logger, read_json, write_json

logger = get_logger("acquire")


class AcquisitionError(RuntimeError):
    """Raised when data cannot be listed or downloaded."""


def _get_json(session, url, cfg):
    """GET a JSON page from the OSF API, retrying with exponential backoff."""
    retries = cfg["download"]["max_retries"]
    backoff = cfg["download"]["backoff_seconds"]
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=cfg["download"]["timeout_seconds"])
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("OSF request failed (%s/%s) for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                raise AcquisitionError(f"Could not reach OSF at {url}") from exc
            time.sleep(backoff * 2 ** (attempt - 1))
    return None


def list_osf_files(cfg, session=None):
    """Recursively list every file in the OSF project's storage.

    Returns:
        list[dict]: One dict per file with name, path, size, URL and hashes.
    """
    session = session or requests.Session()
    project = cfg["data"]["osf_project_id"]
    start_url = f"{cfg['data']['osf_api_url']}/nodes/{project}/files/osfstorage/"
    files, folders = [], [start_url]

    while folders:
        url = folders.pop()
        while url:
            page = _get_json(session, url, cfg)
            for item in page.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") == "folder":
                    related = item["relationships"]["files"]["links"]["related"]["href"]
                    folders.append(related)
                else:
                    hashes = (attrs.get("extra") or {}).get("hashes") or {}
                    files.append(
                        {
                            "name": attrs.get("name"),
                            "path": attrs.get("materialized_path"),
                            "size": attrs.get("size"),
                            "download_url": item.get("links", {}).get("download"),
                            "sha256": hashes.get("sha256"),
                            "md5": hashes.get("md5"),
                        }
                    )
            url = (page.get("links") or {}).get("next")

    logger.info("Listed %d files in OSF project %s", len(files), project)
    return files


def choose_participants(entries, cfg, mode):
    """The participant zips to process, and the mode that chose them.

    data.participants, when set, names exactly who to process, whatever the mode.
    Otherwise "sample" takes the data.sample_size smallest zips (quick, and the same
    on every machine) and "full" takes all of them.
    """
    wanted = cfg["data"].get("participants")
    if wanted:
        names = {f"{p}.zip" for p in wanted}
        missing = names - {e["name"] for e in entries}
        if missing:
            raise AcquisitionError(f"Participants not found: {sorted(missing)}")
        return [e for e in entries if e["name"] in names], "list"
    entries = sorted(entries, key=lambda e: (e["size"] or 0, e["name"]))
    if mode == "sample":
        return entries[: cfg["data"]["sample_size"]], mode
    if mode != "full":
        raise ValueError(f"Unknown mode: {mode}")
    return entries, mode


def build_manifest(files, cfg, mode=None):
    """Select the participant zips and demographics file to process."""
    pattern = re.compile(cfg["files"]["participant_zip_pattern"])
    participants = [f for f in files if f["name"] and pattern.match(f["name"])]
    demographics = [f for f in files if f["name"] == cfg["files"]["demographics_file"]]

    if not participants:
        raise AcquisitionError("No participant zips found in the OSF listing")
    if not demographics:
        raise AcquisitionError("Demographics file not found in the OSF listing")

    participants, mode = choose_participants(participants, cfg, mode or cfg["data"]["mode"])
    manifest = {
        "mode": mode,
        "participants": sorted(participants, key=lambda f: _participant_number(f["name"])),
        "demographics": demographics[0],
    }
    total_gb = sum(f["size"] or 0 for f in participants) / 1e9
    logger.info("Manifest: %d participants (%.2f GB) in %s mode", len(participants), total_gb, mode)
    return manifest


def have_local_files(cfg):
    """True when data/raw/ already holds Demographics.csv and every data.participants zip."""
    wanted, raw = cfg["data"].get("participants"), cfg["resolved_paths"]["raw_dir"]
    names = [cfg["files"]["demographics_file"]] + [f"{p}.zip" for p in wanted or []]
    return bool(wanted) and all((raw / n).exists() for n in names)


def build_local_manifest(cfg, mode=None):
    """Build a manifest from the zips already in data/raw/."""
    folder = cfg["resolved_paths"]["raw_dir"]
    pattern = re.compile(cfg["files"]["participant_zip_pattern"])
    zips = [{"name": p.name, "local_path": str(p), "size": p.stat().st_size}
            for p in folder.glob("*.zip") if pattern.match(p.name)]
    zips, mode = choose_participants(zips, cfg, mode or cfg["data"]["mode"])
    demo_path = folder / cfg["files"]["demographics_file"]
    manifest = {
        "mode": mode,
        "participants": sorted(zips, key=lambda f: _participant_number(f["name"])),
        "demographics": {"name": demo_path.name, "local_path": str(demo_path)},
    }
    logger.info("Local manifest: %d participants from %s", len(zips), folder)
    return manifest


def fetch_manifest(cfg, mode=None, session=None):
    """Create the manifest and save it to disk.

    If data/raw/ already has every listed participant and Demographics.csv, they
    are used as they are, with no network. Otherwise OSF is listed, and each zip is
    downloaded later unless a valid copy is already in data/raw/.
    """
    if have_local_files(cfg):
        logger.info("All listed participants are already in %s: not downloading", cfg["resolved_paths"]["raw_dir"])
        manifest = build_local_manifest(cfg, mode)
    else:
        manifest = build_manifest(list_osf_files(cfg, session), cfg, mode)
    path = cfg["resolved_paths"]["raw_dir"] / "manifest.json"
    write_json(manifest, path)
    return manifest


def load_manifest(cfg):
    return read_json(cfg["resolved_paths"]["raw_dir"] / "manifest.json")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(entry, cfg, dest_dir=None, session=None):
    """Download one file, verify it and return the local path.

    Files listed with a local_path are copied or used in place instead.
    Downloads go to a temporary .part file, so a failed download never
    leaves a corrupt file behind.
    """
    dest_dir = Path(dest_dir or cfg["resolved_paths"]["raw_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    if entry.get("local_path"):
        source = Path(entry["local_path"])
        if not source.exists():
            raise AcquisitionError(f"Local file missing: {source}")
        return source

    dest = dest_dir / entry["name"]
    if dest.exists() and _is_valid(dest, entry):
        logger.info("Using cached %s", dest.name)
        return dest

    check_download_url(entry.get("download_url"), cfg)
    session = session or requests.Session()
    retries = cfg["download"]["max_retries"]
    backoff = cfg["download"]["backoff_seconds"]
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", entry["name"], attempt, retries)
            with session.get(
                entry["download_url"], stream=True, timeout=cfg["download"]["timeout_seconds"]
            ) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(cfg["download"]["chunk_size_bytes"]):
                        f.write(chunk)
            if not _is_valid(tmp, entry):
                raise AcquisitionError(f"Checksum or size mismatch for {entry['name']}")
            shutil.move(tmp, dest)
            logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except (requests.RequestException, AcquisitionError, OSError) as exc:
            logger.warning("Download failed for %s: %s", entry["name"], exc)
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise AcquisitionError(f"Giving up on {entry['name']}") from exc
            time.sleep(backoff * 2 ** (attempt - 1))
    return None


def check_download_url(url, cfg):
    """Refuse a download address that isn't https:// on an allowed domain.

    The address comes from the OSF listing, so this keeps a changed or
    tampered listing from sending the pipeline anywhere else.
    """
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower()
    allowed = [h.lower() for h in cfg["download"]["allowed_hosts"]]
    if parts.scheme != "https" or not any(host == h or host.endswith("." + h) for h in allowed):
        raise AcquisitionError(f"Refusing to download from {url!r}: not https:// on {', '.join(allowed)}")


def _is_valid(path, entry):
    """Check a file's size and, when OSF provides it, its SHA-256 hash."""
    if entry.get("size") and path.stat().st_size != entry["size"]:
        return False
    if entry.get("sha256") and _sha256(path) != entry["sha256"]:
        return False
    return True


def _participant_number(name):
    match = re.search(r"(\d+)", name or "")
    return int(match.group(1)) if match else 0
