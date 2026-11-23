"""Download every participant's raw zip (and Demographics.csv) into data/raw/, nothing else.

    python -m scripts.download_all            # checks the size and free space, then downloads
    python -m scripts.download_all --check    # only the size and free space, no download

For looking at the raw 100 Hz data. Zips already there and valid are skipped, so a stopped
run is resumed by running it again; a participant that fails is listed and retried next time.
The normal pipeline reuses these zips (keep data.delete_raw_after_processing false).
"""

import argparse
import shutil
import sys

from scripts import acquire
from scripts.utils import get_logger, load_config

logger = get_logger("download_all")
GB = 1e9
HEADROOM = 1.2   # want 20% more free space than the download needs


def free_gb(path):
    return shutil.disk_usage(path).free / GB


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download every COPS participant's raw zip")
    parser.add_argument("--check", action="store_true", help="only print the size and free space")
    parser.add_argument("--force", action="store_true", help="download even if free space looks too small")
    args = parser.parse_args(argv)

    cfg = load_config(overrides={"data": {"participants": None}})   # everyone, not the model team's list
    raw = cfg["resolved_paths"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    manifest = acquire.fetch_manifest(cfg, "full")
    people = manifest["participants"]

    have = {p.name for p in raw.glob("*.zip")}
    total = sum(e["size"] or 0 for e in people) / GB
    todo = sum(e["size"] or 0 for e in people if e["name"] not in have) / GB
    free = free_gb(raw)
    print(f"{len(people)} participants, {total:.1f} GB in all; {len(have)} zips already here, {todo:.1f} GB to go")
    print(f"Free space where data/raw is: {free:.1f} GB (want at least {todo * HEADROOM:.1f} GB)")
    if args.check:
        return 0
    if free < todo * HEADROOM and not args.force:
        print("Not enough free space: free some up, or rerun with --force")
        return 1

    acquire.download_file(manifest["demographics"], cfg)
    failed = []
    for i, entry in enumerate(people, 1):
        try:
            acquire.download_file(entry, cfg)
            print(f"{i}/{len(people)} {entry['name']} ok, {free_gb(raw):.1f} GB free", flush=True)
        except Exception as exc:   # one failure shouldn't stop the rest
            failed.append(entry["name"])
            print(f"{i}/{len(people)} {entry['name']} FAILED: {exc}", flush=True)
    print(f"Done: {len(people) - len(failed)} ok, {len(failed)} failed {failed}; {free_gb(raw):.1f} GB free")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
