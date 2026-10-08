import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A copy of the repo's config + seeds in a temp dir, with fewer resamples for speed."""
    for name in ("seeds", "baselines"):
        shutil.copytree(ROOT / name, tmp_path / name)
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    params["bootstrap"]["n_resamples"] = 200
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    shutil.copy(ROOT / "gate.yaml", tmp_path / "gate.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def evaluated(workspace):
    from llm_eval_lab.cli import main

    assert main(["make-testsets"]) == 0
    assert main(["predict"]) == 0
    assert main(["evaluate"]) == 0
    return workspace
