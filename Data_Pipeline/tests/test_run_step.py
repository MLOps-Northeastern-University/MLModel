"""Each step run on its own, one command at a time, as the Airflow DAG runs them: the same dataset as
the whole pipeline, with every hand-off through files."""

import json

import pandas as pd

from scripts import pipeline, run_step
from scripts.utils import read_table
from tests.helpers import make_local_run


def command(cfg, monkeypatch, *argv):
    monkeypatch.setattr(run_step, "load_config", lambda path=None: cfg)
    return run_step.main(list(argv))


def run_all(cfg, monkeypatch, capsys):
    """manifest, patients, each participant, then the rest, as the DAG does. {step: exit code}."""
    codes = {"manifest": command(cfg, monkeypatch, "manifest")}
    names = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    codes["patients"] = command(cfg, monkeypatch, "patients")
    codes["participants"] = [command(cfg, monkeypatch, "participant", name) for name in names]
    for step in ("check_failures", "build", "checks"):
        codes[step] = command(cfg, monkeypatch, step)
    return codes


def processed(cfg, name):
    return read_table(cfg["resolved_paths"]["processed_dir"] / name)


def test_the_steps_one_by_one_build_the_same_dataset_as_the_whole_run(tmp_path, monkeypatch, capsys):
    whole = make_local_run(tmp_path / "whole")
    assert pipeline.run(whole) == 0
    steps = make_local_run(tmp_path / "steps")
    codes = run_all(steps, monkeypatch, capsys)
    assert codes == {"manifest": 0, "patients": 0, "participants": [0, 0], "check_failures": 0, "build": 0,
                     "checks": 0}
    for name in ("hours.csv", "events.csv", "patients.csv"):
        pd.testing.assert_frame_equal(processed(steps, name), processed(whole, name))


def test_manifest_prints_the_participants_for_the_dag(tmp_path, monkeypatch, capsys):
    cfg = make_local_run(tmp_path)
    assert command(cfg, monkeypatch, "manifest") == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == ["COPS-1.zip", "COPS-2.zip"]


def test_a_participant_left_from_an_earlier_run_is_not_counted(tmp_path, monkeypatch, capsys):
    cfg = make_local_run(tmp_path)
    run_all(cfg, monkeypatch, capsys)                              # an earlier run: both participants worked
    assert command(cfg, monkeypatch, "manifest") == 0
    assert not (cfg["resolved_paths"]["interim_dir"] / "COPS-2").exists()
    command(cfg, monkeypatch, "patients")
    command(cfg, monkeypatch, "participant", "COPS-1.zip")       # COPS-2 fails this time: never runs
    assert command(cfg, monkeypatch, "check_failures") == 1      # 1 of 2 failed: more than allowed


def test_a_participant_not_in_the_manifest_is_refused(tmp_path, monkeypatch):
    cfg = make_local_run(tmp_path)
    command(cfg, monkeypatch, "manifest")
    command(cfg, monkeypatch, "patients")
    assert command(cfg, monkeypatch, "participant", "COPS-77.zip") == 1
