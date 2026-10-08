"""CI regression gate: compare the candidate against a stored baseline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .datasets import read_jsonl
from .stats import mcnemar


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        head = "## Eval gate: " + ("PASSED" if self.passed else "FAILED")
        lines = [head, "", "| Check | Baseline | Current | Rule | Result |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c['check']} | {c.get('baseline', '')} | {c.get('current', '')} | {c['rule']} | {'ok' if c['ok'] else '**FAIL**'} |")
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def make_baseline(report: dict, runs_dir: Path, system: str) -> dict:
    preds = {r["id"]: r["pred"] for r in read_jsonl(Path(runs_dir) / f"{system}.jsonl")}
    m = report["systems"][system]
    return {
        "system": system,
        "testset": report["testset"],
        "metrics": {"accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)},
        "mt": {s: v for s, v in report["mt"]["systems"].items()},
        "predictions": dict(sorted(preds.items())),
    }


def run_gate(report: dict, runs_dir: Path, baseline: dict, rules: dict, data_dir: Path) -> GateResult:
    res = GateResult(True)
    cand = report["comparison"]["candidate"]
    cur = report["systems"][cand]
    same_testset = baseline["testset"]["sha256"] == report["testset"]["sha256"]
    if not same_testset:
        msg = (
            f"test set changed: baseline {baseline['testset']['name']} {baseline['testset']['sha256'][:12]} "
            f"vs current {report['testset']['name']} {report['testset']['sha256'][:12]}"
        )
        if rules.get("allow_testset_change", False):
            res.notes.append(msg + " (allowed; paired test skipped, refresh the baseline)")
        else:
            res.passed = False
            res.checks.append({"check": "test set hash", "rule": "must match baseline", "ok": False,
                               "baseline": baseline["testset"]["sha256"][:12], "current": report["testset"]["sha256"][:12]})
            res.notes.append(msg + ". Re-baseline deliberately with `llm-eval baseline` in a reviewed PR.")
    for metric, max_drop in (rules.get("max_drop") or {}).items():
        b, c = baseline["metrics"][metric], round(cur[metric], 4)
        ok = c >= b - max_drop
        res.checks.append({"check": f"{metric} drop", "baseline": b, "current": c, "rule": f">= baseline - {max_drop}", "ok": ok})
        res.passed &= ok
    for metric, floor in (rules.get("min") or {}).items():
        c = round(cur[metric], 4)
        ok = c >= floor
        res.checks.append({"check": f"{metric} floor", "current": c, "rule": f">= {floor}", "ok": ok})
        res.passed &= ok
    sig = rules.get("significance")
    if sig and same_testset:
        rows = read_jsonl(Path(data_dir) / baseline["testset"]["name"])
        cur_preds = {r["id"]: r["pred"] for r in read_jsonl(Path(runs_dir) / f"{cand}.jsonl")}
        ca = [baseline["predictions"][r["id"]] == r["label"] for r in rows]
        cb = [cur_preds[r["id"]] == r["label"] for r in rows]
        mc = mcnemar(ca, cb)
        worse = mc.b > mc.c
        ok = not (worse and mc.p_exact < sig.get("alpha", 0.05))
        res.checks.append({
            "check": "McNemar vs baseline",
            "baseline": f"b={mc.b}",
            "current": f"c={mc.c}",
            "rule": f"not significantly worse (p={mc.p_exact:.3g}, alpha={sig.get('alpha', 0.05)})",
            "ok": ok,
        })
        res.passed &= ok
    for system, limits in (rules.get("mt_max_drop") or {}).items():
        for metric, max_drop in limits.items():
            b = baseline["mt"][system][metric]
            c = report["mt"]["systems"][system][metric]
            ok = c >= b - max_drop
            res.checks.append({"check": f"MT {system} {metric} drop", "baseline": b, "current": c, "rule": f">= baseline - {max_drop}", "ok": ok})
            res.passed &= ok
    return res


def load_rules(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
