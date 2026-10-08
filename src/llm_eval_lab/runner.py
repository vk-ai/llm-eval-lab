"""Pipeline steps used by the CLI and by dvc.yaml stages."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import LABELS
from .datasets import read_jsonl, sha256_file
from .metrics import bootstrap_ci, classification_report
from .mt import evaluate_mt
from .stats import mcnemar, permutation_test
from .systems import get_system


def load_params(path: str | Path = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def testset_path(data_dir: Path, version: str) -> Path:
    return Path(data_dir) / f"intent_{version}.jsonl"


def predict(data_dir: Path, runs_dir: Path, version: str, systems: list[str]) -> dict[str, Path]:
    rows = read_jsonl(testset_path(data_dir, version))
    train = read_jsonl(Path(data_dir) / "intent_train.jsonl")
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in systems:
        system = get_system(name, train)
        preds = system.predict(rows)
        path = runs_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in preds))
        out[name] = path
    return out


def load_preds(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


def system_metrics(rows: list[dict], preds: dict[str, dict], n_resamples: int, seed: int) -> dict:
    gold = [r["label"] for r in rows]
    pred = [preds[r["id"]]["pred"] for r in rows]
    rep = classification_report(gold, pred, LABELS)
    rep["ci95"] = {
        m: bootstrap_ci(gold, pred, LABELS, m, n_resamples=n_resamples, seed=seed).to_dict()
        for m in ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    }
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        by_kind.setdefault(r.get("kind", "all"), []).append(int(preds[r["id"]]["pred"] == r["label"]))
    rep["accuracy_by_slice"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())}
    return rep


def compare(rows: list[dict], preds_a: dict, preds_b: dict, n_resamples: int, seed: int) -> dict:
    gold = [r["label"] for r in rows]
    pa = [preds_a[r["id"]]["pred"] for r in rows]
    pb = [preds_b[r["id"]]["pred"] for r in rows]
    return {
        "mcnemar": mcnemar([x == g for x, g in zip(pa, gold)], [x == g for x, g in zip(pb, gold)]).to_dict(),
        "permutation": {
            m: permutation_test(gold, pa, pb, LABELS, m, n_resamples=n_resamples, seed=seed).to_dict()
            for m in ("accuracy", "macro_f1")
        },
    }


def evaluate(data_dir: Path, runs_dir: Path, params: dict) -> dict:
    version = params["testset"]["version"]
    path = testset_path(data_dir, version)
    rows = read_jsonl(path)
    nb = params.get("bootstrap", {})
    n_res, seed = int(nb.get("n_resamples", 1000)), int(nb.get("seed", 0))
    preds = {s: load_preds(Path(runs_dir) / f"{s}.jsonl") for s in params["systems"]}
    base, cand = params["compare"]["baseline"], params["compare"]["candidate"]
    mt_rows = read_jsonl(Path(data_dir) / "mt_en_de_v1.jsonl")
    return {
        "testset": {"name": path.name, "version": version, "sha256": sha256_file(path), "n": len(rows)},
        "systems": {s: system_metrics(rows, p, n_res, seed) for s, p in preds.items()},
        "comparison": {"baseline": base, "candidate": cand, **compare(rows, preds[base], preds[cand], n_res, seed)},
        "mt": evaluate_mt(mt_rows, n_resamples=n_res, seed=seed, use_comet=bool(params.get("mt", {}).get("comet", False))),
    }


def flat_metrics(report: dict) -> dict:
    """Small, flat file for `dvc metrics show/diff`."""
    out = {}
    for s, m in report["systems"].items():
        out[s] = {"accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
    for s, m in report["mt"]["systems"].items():
        out[f"mt_{s}"] = {"bleu": m["bleu"], "chrf_pp": m["chrf++"]}
    out["comparison"] = {
        "mcnemar_p": report["comparison"]["mcnemar"]["p_exact"],
        "macro_f1_delta": report["comparison"]["permutation"]["macro_f1"]["delta"],
        "macro_f1_p": report["comparison"]["permutation"]["macro_f1"]["p_value"],
    }
    return out
