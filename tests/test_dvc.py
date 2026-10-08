"""Runs the real DVC pipeline in a scratch git repo. Skipped when the dvc CLI isn't installed."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT

pytestmark = [pytest.mark.dvc, pytest.mark.skipif(shutil.which("dvc") is None, reason="pip install '.[dvc]'")]


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_repro_push_pull_roundtrip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("src", "seeds", "baselines", "dvc.yaml", "params.yaml", "gate.yaml", "dvc.lock"):
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, repo / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        elif src.exists():
            shutil.copy(src, repo / name)
    sh("git", "init", "-q", cwd=repo)
    sh("dvc", "init", "-q", cwd=repo)
    remote = tmp_path / "remote"
    sh("dvc", "remote", "add", "-d", "local", str(remote), cwd=repo)
    sh("dvc", "repro", "testsets", cwd=repo)
    lock = (repo / "dvc.lock").read_text()
    assert "data/testsets" in lock
    # Second repro is a no-op: outputs match the lock file.
    assert "didn't change" in sh("dvc", "repro", "testsets", cwd=repo) or "up to date" in sh("dvc", "status", "testsets", cwd=repo)
    test_file = repo / "data" / "testsets" / "intent_v2.jsonl"
    digest = hashlib.sha256(test_file.read_bytes()).hexdigest()
    sh("dvc", "push", cwd=repo)
    shutil.rmtree(repo / "data" / "testsets")
    sh("dvc", "pull", "testsets", cwd=repo)
    assert hashlib.sha256(test_file.read_bytes()).hexdigest() == digest


def test_committed_lock_matches_generator(tmp_path):
    """dvc.lock in the repo pins the same test-set hash the generator produces today."""
    import yaml

    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())
    pinned = {o["path"]: o["md5"] for o in lock["stages"]["testsets"]["outs"]}
    repo = tmp_path / "r"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", ".dvc", "data", "runs", "reports", "__pycache__", ".venv", "*.egg-info"))
    sh("git", "init", "-q", cwd=repo)
    sh("dvc", "init", "-q", cwd=repo)
    sh("dvc", "repro", "testsets", cwd=repo)
    fresh = yaml.safe_load((repo / "dvc.lock").read_text())
    assert {o["path"]: o["md5"] for o in fresh["stages"]["testsets"]["outs"]} == pinned
