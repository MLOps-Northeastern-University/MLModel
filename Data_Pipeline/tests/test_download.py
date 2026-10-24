"""Acquisition failure paths: retries, checksums, caching, OSF errors, local zips.

No test touches the network: a fake session stands in for requests.
"""

import hashlib

import pytest

from scripts import acquire
from tests.helpers import make_config, make_demographics, make_participant_zip

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


def test_local_zips_sample_mode_takes_the_smallest(tmp_path):
    folder = tmp_path / "zips"
    folder.mkdir()
    # The manifest only looks at names and sizes, so plain files of known sizes will do.
    for name, size in [("COPS-1.zip", 900), ("COPS-2.zip", 100), ("COPS-3.zip", 500), ("notes.zip", 1)]:
        (folder / name).write_bytes(b"x" * size)
    make_demographics(folder, ["COPS-1", "COPS-2", "COPS-3"])

    cfg = make_config(tmp_path, data={"local_zip_dir": str(folder), "sample_size": 2})
    manifest = acquire.fetch_manifest(cfg, "sample")

    # The two smallest participant zips, in participant order; other zips are ignored.
    assert [p["name"] for p in manifest["participants"]] == ["COPS-2.zip", "COPS-3.zip"]
    assert manifest["demographics"]["name"] == "Demographics.csv"
    assert acquire.load_manifest(cfg) == manifest  # saved for the next steps


def test_local_zips_without_demographics_still_run(tmp_path):
    folder = tmp_path / "zips"
    folder.mkdir()
    make_participant_zip(folder, "COPS-1")
    cfg = make_config(tmp_path, data={"local_zip_dir": str(folder)})
    assert acquire.build_local_manifest(cfg, "full")["demographics"] is None


def test_an_empty_local_folder_is_reported(tmp_path):
    (tmp_path / "zips").mkdir()
    cfg = make_config(tmp_path, data={"local_zip_dir": str(tmp_path / "zips")})
    with pytest.raises(acquire.AcquisitionError, match="No participant zips found"):
        acquire.build_local_manifest(cfg, "full")


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
