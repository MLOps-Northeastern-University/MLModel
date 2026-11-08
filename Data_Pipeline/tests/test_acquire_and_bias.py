import pandas as pd
import pytest

from scripts import acquire
from scripts.bias import run_bias_analysis
from tests.helpers import make_config


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, download={"backoff_seconds": 0})


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Mimics two pages of the OSF API: a root listing and a Data folder."""

    def get(self, url, timeout=None, stream=False):
        if url.endswith("/files/osfstorage/"):
            return FakeResponse({"data": [
                {"attributes": {"kind": "file", "name": "Demographics.csv", "size": 3600, "extra": {}},
                 "links": {"download": "https://osf.example/demo"}},
                {"attributes": {"kind": "folder", "name": "Data"},
                 "relationships": {"files": {"links": {"related": {"href": "https://osf.example/data/"}}}}},
            ], "links": {"next": None}})
        return FakeResponse({"data": [
            {"attributes": {"kind": "file", "name": f"COPS-{i}.zip", "size": size, "extra": {}},
             "links": {"download": f"https://osf.example/{i}"}}
            for i, size in [(1, 900), (2, 100), (3, 500), (4, 300)]
        ], "links": {"next": None}})


def test_manifest_picks_smallest_in_sample_mode(cfg):
    cfg["data"]["sample_size"] = 2
    files = acquire.list_osf_files(cfg, FakeSession())
    manifest = acquire.build_manifest(files, cfg, "sample")
    assert [p["name"] for p in manifest["participants"]] == ["COPS-2.zip", "COPS-4.zip"]
    assert manifest["demographics"]["name"] == "Demographics.csv"


def test_full_mode_takes_everyone(cfg):
    files = acquire.list_osf_files(cfg, FakeSession())
    assert len(acquire.build_manifest(files, cfg, "full")["participants"]) == 4


def test_unreachable_osf_raises(cfg):
    class Broken:
        def get(self, *a, **k):
            raise acquire.requests.ConnectionError("down")
    with pytest.raises(acquire.AcquisitionError):
        acquire.list_osf_files(cfg, Broken())


def test_bias_report_runs(cfg):
    rows = []
    for p in range(6):
        for h in range(30):
            rows.append({"participant_id": f"COPS-{p}", "hour_index": h,
                         "kinesia_class": "off" if h % 3 == 0 else "on",
                         "sex": "M" if p % 2 else "F",
                         "W_move_mean": 0.01 if h % 3 == 0 else 0.08, "W_tremor_mean": 0.02})
    report = run_bias_analysis(pd.DataFrame(rows), cfg)
    assert "sex" in report["data_level"]
    assert "slices" in report["model_level"]
    assert (cfg["resolved_paths"]["reports_dir"] / "bias_report.md").exists()
