"""Acquisition failure paths: retries, checksums, caching, OSF errors, zips already in data/raw.

No test touches the network: a fake session stands in for requests.
"""

import hashlib

import pytest

from data_pipeline import acquire
from tests.helpers import make_config, make_demographics, raw_dir

CONTENT = b"participant zip bytes " * 1000
SHA = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, download={"backoff_seconds": 0, "max_retries": 3})


def entry(**overrides):
    base = {"name": "COPS-1.zip", "download_url": "https://files.osf.io/v1/resources/5xvwn/providers/osfstorage/1",
            "size": len(CONTENT), "sha256": SHA}
    return {**base, **overrides}


class StreamResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


class ScriptedSession:
    """Answers each get() with the next scripted outcome: bytes, or an exception to raise."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, stream=False, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return StreamResponse(outcome)


class NoNetwork:
    def get(self, *args, **kwargs):
        raise AssertionError("should not download")


# ---------- download_file ----------

def test_download_is_verified_and_leaves_no_partial_file(cfg, tmp_path):
    session = ScriptedSession(CONTENT)
    path = acquire.download_file(entry(), cfg, tmp_path / "raw", session)
    assert path.read_bytes() == CONTENT
    assert not list((tmp_path / "raw").glob("*.part"))


def test_a_dropped_connection_is_retried(cfg, tmp_path):
    session = ScriptedSession(acquire.requests.ConnectionError("reset"), CONTENT)
    path = acquire.download_file(entry(), cfg, tmp_path / "raw", session)
    assert session.calls == 2
    assert path.read_bytes() == CONTENT


def test_a_corrupt_download_is_retried_then_refused(cfg, tmp_path):
    corrupt = CONTENT[:-1] + b"X"  # right size, wrong checksum
    session = ScriptedSession(corrupt, corrupt, corrupt)
    with pytest.raises(acquire.AcquisitionError, match="Giving up on COPS-1.zip"):
        acquire.download_file(entry(), cfg, tmp_path / "raw", session)
    assert session.calls == 3
    # Nothing corrupt is left behind to be used later.
    assert not list((tmp_path / "raw").iterdir())


def test_a_truncated_download_is_caught_by_its_size(cfg, tmp_path):
    session = ScriptedSession(CONTENT[:100], CONTENT)
    path = acquire.download_file(entry(sha256=None), cfg, tmp_path / "raw", session)
    assert session.calls == 2
    assert path.stat().st_size == len(CONTENT)


def test_a_valid_cached_file_is_reused_without_downloading(cfg, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "COPS-1.zip").write_bytes(CONTENT)
    assert acquire.download_file(entry(), cfg, raw, NoNetwork()).read_bytes() == CONTENT


def test_a_damaged_cached_file_is_downloaded_again(cfg, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "COPS-1.zip").write_bytes(b"half a file")
    session = ScriptedSession(CONTENT)
    assert acquire.download_file(entry(), cfg, raw, session).read_bytes() == CONTENT
    assert session.calls == 1


def test_a_missing_local_zip_is_reported(cfg, tmp_path):
    with pytest.raises(acquire.AcquisitionError, match="Local file missing"):
        acquire.download_file({"name": "COPS-1.zip", "local_path": str(tmp_path / "gone.zip")}, cfg)


# ---------- listing OSF ----------

class JsonResponse:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if self.payload is None:
            raise ValueError("not JSON")
        return self.payload


class PagedSession:
    """Two pages of one folder, linked by links.next, after one failed request."""

    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if len(self.calls) == 1:
            return JsonResponse(error=acquire.requests.HTTPError("503"))
        if url.endswith("page=2"):
            return JsonResponse({"data": [_file("COPS-2.zip", 20)], "links": {"next": None}})
        return JsonResponse({"data": [_file("Demographics.csv", 5), _file("COPS-1.zip", 10)],
                             "links": {"next": "https://osf.example/files?page=2"}})


def _file(name, size):
    return {"attributes": {"kind": "file", "name": name, "size": size,
                           "extra": {"hashes": {"sha256": "abc"}}},
            "links": {"download": f"https://osf.example/{name}"}}


def test_osf_listing_follows_pages_and_retries_a_failed_page(cfg):
    session = PagedSession()
    files = acquire.list_osf_files(cfg, session)
    assert sorted(f["name"] for f in files) == ["COPS-1.zip", "COPS-2.zip", "Demographics.csv"]
    assert files[0]["sha256"] == "abc"
    assert len(session.calls) == 3  # failed first page, its retry, then page 2


def test_a_reply_that_is_not_json_counts_as_a_failure(cfg):
    class NotJson:
        def get(self, url, timeout=None):
            return JsonResponse(payload=None)
    with pytest.raises(acquire.AcquisitionError, match="Could not reach OSF"):
        acquire.list_osf_files(cfg, NotJson())


# ---------- choosing participants ----------

def test_a_listing_without_participants_or_demographics_is_refused(cfg):
    demographics = {"name": "Demographics.csv", "size": 1}
    with pytest.raises(acquire.AcquisitionError, match="No participant zips"):
        acquire.build_manifest([demographics], cfg, "full")
    with pytest.raises(acquire.AcquisitionError, match="Demographics file not found"):
        acquire.build_manifest([{"name": "COPS-1.zip", "size": 1}], cfg, "full")


def test_an_unknown_mode_is_refused(cfg):
    files = [{"name": "COPS-1.zip", "size": 1}, {"name": "Demographics.csv", "size": 1}]
    with pytest.raises(ValueError, match="Unknown mode"):
        acquire.build_manifest(files, cfg, "everything")


def test_listed_participants_already_in_raw_are_used_without_osf(tmp_path):
    folder = raw_dir(tmp_path)
    for name in ["COPS-1.zip", "COPS-2.zip", "COPS-3.zip"]:
        (folder / name).write_bytes(b"x")
    make_demographics(folder, ["COPS-1", "COPS-2", "COPS-3"])
    cfg = make_config(tmp_path, data={"mode": "list", "participants": ["COPS-3", "COPS-2"]})
    manifest = acquire.fetch_manifest(cfg, session=NoNetwork())
    assert [p["name"] for p in manifest["participants"]] == ["COPS-2.zip", "COPS-3.zip"]
    assert manifest["mode"] == "list" and manifest["demographics"]["name"] == "Demographics.csv"
    assert acquire.load_manifest(cfg) == manifest  # saved for the next steps


@pytest.mark.parametrize("participants, files", [
    (["COPS-1", "COPS-2"], ["COPS-1.zip", "Demographics.csv"]),   # a listed zip is missing
    (["COPS-1"], ["COPS-1.zip"]),                                  # Demographics.csv is missing
    ([], ["COPS-1.zip", "Demographics.csv"]),                      # no list: OSF says who there is
])
def test_otherwise_osf_is_listed(tmp_path, participants, files):
    for name in files:
        (raw_dir(tmp_path) / name).write_bytes(b"x")
    cfg = make_config(tmp_path, data={"participants": participants})
    assert not acquire.have_local_files(cfg, "list")


def test_zips_in_raw_are_used_without_osf_only_in_list_mode(tmp_path):
    folder = raw_dir(tmp_path)
    (folder / "COPS-1.zip").write_bytes(b"x")
    make_demographics(folder, ["COPS-1"])
    cfg = make_config(tmp_path, data={"participants": ["COPS-1"]})
    assert acquire.have_local_files(cfg, "list")
    assert not acquire.have_local_files(cfg, "full")              # full means everyone OSF has


# ---------- safety ----------

@pytest.mark.parametrize("url", [
    "http://files.osf.io/v1/file",           # not https
    "https://evil.example/COPS-1.zip",        # another site
    "https://osf.io.evil.example/x",          # osf.io only as part of another domain
    "https://notosf.io/x",                    # a look-alike domain
    "file:///etc/passwd",                     # a local file
    "",                                       # no address at all
])
def test_downloads_only_from_osf_over_https(cfg, tmp_path, url):
    with pytest.raises(acquire.AcquisitionError, match="Refusing to download"):
        acquire.download_file(entry(download_url=url), cfg, tmp_path / "raw", NoNetwork())


@pytest.mark.parametrize("url", ["https://osf.io/download/abc/", "https://files.osf.io/v1/resources/x"])
def test_osf_addresses_are_accepted(cfg, url):
    acquire.check_download_url(url, cfg)  # no error


@pytest.mark.parametrize("name", ["../../COPS-1.zip", "COPS-1.zip/../../evil.zip", "COPS-1.zip.exe",
                                  r"C:\COPS-1.zip", "COPS-x.zip"])
def test_odd_file_names_in_the_listing_are_never_chosen(cfg, name):
    files = [{"name": name, "size": 1}, {"name": "COPS-2.zip", "size": 2},
             {"name": "Demographics.csv", "size": 1}]
    manifest = acquire.build_manifest(files, cfg, "full")
    assert [p["name"] for p in manifest["participants"]] == ["COPS-2.zip"]


# ---------- the manifest from a nested OSF listing ----------

class FolderSession:
    """A root listing with Demographics.csv and a Data folder holding four participants."""

    def get(self, url, timeout=None, stream=False):
        if url.endswith("/files/osfstorage/"):
            return JsonResponse({"data": [
                {"attributes": {"kind": "file", "name": "Demographics.csv", "size": 3600, "extra": {}},
                 "links": {"download": "https://osf.example/demo"}},
                {"attributes": {"kind": "folder", "name": "Data"},
                 "relationships": {"files": {"links": {"related": {"href": "https://osf.example/data/"}}}}},
            ], "links": {"next": None}})
        sizes = [(1, 900), (2, 100), (3, 500), (4, 300)]
        return JsonResponse({"data": [_file(f"COPS-{i}.zip", size) for i, size in sizes], "links": {"next": None}})


def test_sample_mode_picks_the_smallest_from_nested_folders(cfg):
    cfg["data"]["sample_size"] = 2
    manifest = acquire.build_manifest(acquire.list_osf_files(cfg, FolderSession()), cfg, "sample")
    assert [p["name"] for p in manifest["participants"]] == ["COPS-2.zip", "COPS-4.zip"]
    assert manifest["demographics"]["name"] == "Demographics.csv"


def test_full_mode_takes_everyone(cfg):
    files = acquire.list_osf_files(cfg, FolderSession())
    assert len(acquire.build_manifest(files, cfg, "full")["participants"]) == 4


def test_unreachable_osf_raises(cfg):
    class Broken:
        def get(self, *a, **k):
            raise acquire.requests.ConnectionError("down")
    with pytest.raises(acquire.AcquisitionError):
        acquire.list_osf_files(cfg, Broken())


# ---------- a named list of participants ----------

def test_list_mode_takes_exactly_the_named_participants(cfg):
    cfg["data"]["participants"] = ["COPS-3", "COPS-1"]
    manifest = acquire.build_manifest(acquire.list_osf_files(cfg, FolderSession()), cfg, "list")
    assert [p["name"] for p in manifest["participants"]] == ["COPS-1.zip", "COPS-3.zip"]
    assert manifest["mode"] == "list"


def test_the_named_list_is_ignored_in_sample_and_full_mode(cfg):
    cfg["data"]["participants"] = ["COPS-3"]
    cfg["data"]["sample_size"] = 2
    files = acquire.list_osf_files(cfg, FolderSession())
    sampled = acquire.build_manifest(files, cfg, "sample")["participants"]
    assert [p["name"] for p in sampled] == ["COPS-2.zip", "COPS-4.zip"]
    assert len(acquire.build_manifest(files, cfg, "full")["participants"]) == 4


@pytest.mark.parametrize("mode", ["sample", "full"])
def test_excluded_participants_are_never_chosen(cfg, mode):
    cfg["data"]["exclude"] = ["COPS-2"]                            # the smallest, so sample would take it
    cfg["data"]["sample_size"] = 2
    manifest = acquire.build_manifest(acquire.list_osf_files(cfg, FolderSession()), cfg, mode)
    assert "COPS-2.zip" not in [p["name"] for p in manifest["participants"]]


def test_a_participant_both_listed_and_excluded_is_refused(cfg):
    cfg["data"].update(participants=["COPS-2"], exclude=["COPS-2"])
    with pytest.raises(acquire.AcquisitionError, match="listed and excluded"):
        acquire.build_manifest(acquire.list_osf_files(cfg, FolderSession()), cfg, "list")


def test_a_named_participant_who_isnt_there_is_reported(cfg):
    cfg["data"]["participants"] = ["COPS-1", "COPS-99"]
    with pytest.raises(acquire.AcquisitionError, match="COPS-99"):
        acquire.build_manifest(acquire.list_osf_files(cfg, FolderSession()), cfg, "list")
