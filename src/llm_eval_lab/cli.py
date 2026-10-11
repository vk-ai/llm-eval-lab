"""`llm-eval` command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import LABELS
from . import calibration as calib
from .cards import model_card, system_card
from .datasets import build_all, read_jsonl
from .gate import load_json, load_rules, make_baseline, run_gate
from .review import ReviewQueue, simulate_rater
from .runner import evaluate, flat_metrics, load_params, load_preds, predict, testset_path


def _write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def cmd_make_testsets(a) -> int:
    p = load_params(a.params)
    m = build_all(Path(a.out), p["testset"]["version"], int(p["testset"]["seed"]), Path(a.seeds))
    print(json.dumps(m, indent=2))
    return 0


def cmd_predict(a) -> int:
    p = load_params(a.params)
    for name, path in predict(Path(a.data), Path(a.runs), p["testset"]["version"], p["systems"]).items():
        print(f"{name}: {path}")
    return 0


def cmd_evaluate(a) -> int:
    p = load_params(a.params)
    report = evaluate(Path(a.data), Path(a.runs), p)
    _write_json(a.out, report)
    _write_json(a.metrics, flat_metrics(report))
    for s, m in report["systems"].items():
        ci = m["ci95"]["macro_f1"]
        print(f"{s:10s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]")
    c = report["comparison"]
    print(f"McNemar {c['baseline']} vs {c['candidate']}: b={c['mcnemar']['b']} c={c['mcnemar']['c']} p={c['mcnemar']['p_exact']:.4g}")
    return 0


def cmd_card(a) -> int:
    report = load_json(a.report)
    review = None
    if a.review_db and Path(a.review_db).exists():
        review = ReviewQueue(a.review_db).stats()
    system = a.system or report["comparison"]["candidate"]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    when = os.environ.get("CARD_DATE")
    (Path(a.out_dir) / "model_card.md").write_text(model_card(report, system, review, generated_at=when))
    (Path(a.out_dir) / "system_card.md").write_text(system_card(report))
    print(f"wrote {a.out_dir}/model_card.md and system_card.md")
    return 0


def cmd_baseline(a) -> int:
    report = load_json(a.report)
    system = a.system or report["comparison"]["candidate"]
    _write_json(a.out, make_baseline(report, Path(a.runs), system))
    print(f"baseline for {system} written to {a.out}")
    return 0


def cmd_gate(a) -> int:
    report = load_json(a.report)
    if a.candidate:
        # Evaluate a different candidate against the same baseline without touching params.yaml.
        p = load_params(a.params)
        p["systems"] = sorted(set(p["systems"]) | {a.candidate})
        p["compare"]["candidate"] = a.candidate
        a.runs = tempfile.mkdtemp(prefix="llm-eval-gate-")  # don't touch the DVC-tracked runs/ dir
        predict(Path(a.data), Path(a.runs), p["testset"]["version"], p["systems"])
        report = evaluate(Path(a.data), Path(a.runs), p)
    rules = load_rules(a.rules)
    baseline = load_json(rules["baseline"])
    result = run_gate(report, Path(a.runs), baseline, rules, Path(a.data))
    md = result.markdown()
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(md)
    return 0 if result.passed else 1


def cmd_review(a) -> int:
    q = ReviewQueue(a.db, LABELS)
    if a.action == "enqueue":
        p = load_params(a.params)
        rows = read_jsonl(testset_path(Path(a.data), p["testset"]["version"]))
        base, cand = p["compare"]["baseline"], p["compare"]["candidate"]
        n = q.enqueue_from_predictions(rows, load_preds(Path(a.runs) / f"{base}.jsonl"), load_preds(Path(a.runs) / f"{cand}.jsonl"),
                                       confidence_below=a.confidence_below, audit_rate=a.audit_rate)
        print(f"enqueued {n} items")
    elif a.action == "next":
        print(json.dumps(q.next_item(a.rater)))
    elif a.action == "label":
        q.submit(a.item, a.rater, a.label)
    elif a.action == "adjudicate":
        q.adjudicate(a.item, a.label, a.rater, a.note or "")
    elif a.action == "simulate":
        # Two simulated reviewers label everything. For demos only.
        gold = {r[0]: r[1] for r in q.db.execute("SELECT id, gold FROM items")}
        for name, acc in (("alice", a.alice), ("bob", a.bob)):
            rater = simulate_rater(name, acc, LABELS, seed=0)
            while (item := q.next_item(name)) is not None:
                q.submit(item["id"], name, rater(gold[item["id"]]))
        print(json.dumps(q.stats(), indent=2))
    elif a.action == "pending":
        print(json.dumps(q.needs_adjudication(), indent=2))
    elif a.action == "corrections":
        print("".join(json.dumps(c) + "\n" for c in q.corrections()), end="")
    else:
        print(json.dumps(q.stats(), indent=2))
    return 0


def cmd_calibrate(a) -> int:
    """Judge-vs-human agreement against the human-human ceiling, as a gate (exit 1 on fail)."""
    p = load_params(a.params)
    judge_preds = load_preds(Path(a.runs) / f"{a.judge}.jsonl")
    if a.simulate_sample:
        # Random (not disagreement-biased) calibration sample labelled by SIMULATED raters.
        import random

        rows = read_jsonl(testset_path(Path(a.data), p["testset"]["version"]))
        rows = random.Random(a.seed).sample(rows, min(a.simulate_sample, len(rows)))
        q = ReviewQueue(":memory:", LABELS, raters_per_item=len(a.rater_accuracy))
        for r in rows:
            q.enqueue(r["id"], r["text"], r["label"], ["calibration"])
        gold = {r[0]: r[1] for r in q.db.execute("SELECT id, gold FROM items")}
        for spec in a.rater_accuracy:
            name, acc = spec.split("=")
            rater = simulate_rater(name, float(acc), LABELS, seed=a.seed)
            while (item := q.next_item(name)) is not None:
                q.submit(item["id"], name, rater(gold[item["id"]]))
    else:
        q = ReviewQueue(a.db, LABELS)
    judge, humans, ids = calib.from_queue(q, judge_preds)
    cal = calib.calibrate(judge, humans, n_resamples=a.resamples, seed=a.seed)
    rules = (load_rules(a.rules).get("judge_calibration") or {})
    ok, checks = calib.gate(cal, rules)
    out = {"judge": a.judge, **cal.to_dict(), "passed": ok, "checks": checks,
           "simulated_raters": bool(a.simulate_sample)}
    _write_json(a.out, out)
    ci = out["ci95"]
    md = [f"## Judge calibration ({a.judge}, n={cal.n}): " + ("PASSED" if ok else "FAILED"), "",
          f"- judge-human kappa {out['judge_human_kappa']} [{ci['judge_human'][0]}, {ci['judge_human'][1]}] ({out['band']})",
          f"- human-human kappa {out['human_human_kappa']} [{ci['human_human'][0]}, {ci['human_human'][1]}] (ceiling)",
          f"- gap {out['gap']:+} [{ci['gap'][0]:+}, {ci['gap'][1]:+}]", "",
          "| Check | Baseline | Current | Rule | Result |", "|---|---|---|---|---|"]
    md += [f"| {c['check']} | {c.get('baseline', '')} | {c['current']} | {c['rule']} | {'ok' if c['ok'] else '**FAIL**'} |" for c in checks]
    if a.simulate_sample:
        md.append("\n_Raters are simulated (seeded random accuracy)._")
    text = "\n".join(md) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llm-eval")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="data/testsets")
    ap.add_argument("--runs", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-testsets", help="build versioned test sets (DVC stage)")
    s.add_argument("--out", default="data/testsets")
    s.add_argument("--seeds", default="seeds")
    s.set_defaults(fn=cmd_make_testsets)

    s = sub.add_parser("predict", help="run every system in params.yaml on the test set")
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("evaluate", help="metrics, CIs, paired tests, MT scores")
    s.add_argument("--out", default="reports/eval.json")
    s.add_argument("--metrics", default="reports/metrics.json")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("card", help="write model_card.md and system_card.md")
    s.add_argument("--report", default="reports/eval.json")
    s.add_argument("--system")
    s.add_argument("--review-db", default="reports/review.sqlite")
    s.add_argument("--out-dir", default="reports")
    s.set_defaults(fn=cmd_card)

    s = sub.add_parser("baseline", help="store the current candidate as the gate baseline")
    s.add_argument("--report", default="reports/eval.json")
    s.add_argument("--system")
    s.add_argument("--out", default="baselines/baseline.json")
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("gate", help="fail (exit 1) on regressions vs the baseline")
    s.add_argument("--report", default="reports/eval.json")
    s.add_argument("--rules", default="gate.yaml")
    s.add_argument("--candidate", help="evaluate this system instead of params.compare.candidate")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("review", help="human-review queue")
    s.add_argument("action", choices=["enqueue", "next", "label", "adjudicate", "simulate", "pending", "corrections", "stats"])
    s.add_argument("--db", default="reports/review.sqlite")
    s.add_argument("--rater", default="me")
    s.add_argument("--item", type=int)
    s.add_argument("--label", choices=LABELS)
    s.add_argument("--note")
    s.add_argument("--confidence-below", type=float, default=0.6)
    s.add_argument("--audit-rate", type=float, default=0.05)
    s.add_argument("--alice", type=float, default=0.92)
    s.add_argument("--bob", type=float, default=0.85)
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("calibrate", help="judge-vs-human agreement vs the human-human ceiling (gate)")
    s.add_argument("--judge", default="nb-v1", help="system whose predictions act as the judge")
    s.add_argument("--db", default="reports/review.sqlite")
    s.add_argument("--rules", default="gate.yaml")
    s.add_argument("--out", default="reports/calibration.json")
    s.add_argument("--simulate-sample", type=int, default=0,
                   help="draw N random test items and have simulated raters label them")
    s.add_argument("--rater-accuracy", nargs="+", default=["alice=0.92", "bob=0.85", "carol=0.88"])
    s.add_argument("--resamples", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
