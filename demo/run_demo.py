"""Offline end-to-end demo. Runs in a temp copy so it never touches DVC outputs in the repo."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_eval_lab.cli import main  # noqa: E402
from llm_eval_lab.datasets import build_all  # noqa: E402
from llm_eval_lab.gate import load_json  # noqa: E402


def quiet(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


def section(title: str) -> None:
    print(f"\n== {title}")


def run() -> None:
    work = Path(tempfile.mkdtemp(prefix="llm-eval-demo-"))
    for name in ("seeds", "baselines"):
        shutil.copytree(ROOT / name, work / name)
    for name in ("params.yaml", "gate.yaml"):
        shutil.copy(ROOT / name, work / name)
    os.chdir(work)
    os.environ.setdefault("CARD_DATE", "demo run")

    section("1. versioned test sets (same seed -> same bytes; v2 adds 80 hard tickets)")
    for v in ("v1", "v2"):
        m = build_all(work / "tmp" / v, v, 7, work / "seeds")
        f = m["files"][f"intent_{v}.jsonl"]
        print(f"   intent_{v}.jsonl n={f['n']:3d} sha256={f['sha256'][:16]} labels={f['labels']}")
    quiet("make-testsets")
    quiet("predict")

    section("2. per-system metrics with 95% bootstrap CIs (1000 resamples)")
    quiet("evaluate")
    r = load_json("reports/eval.json")
    for s, m in r["systems"].items():
        ci = m["ci95"]
        print(f"   {s:9s} accuracy {ci['accuracy']['point']:.3f} [{ci['accuracy']['low']:.3f}, {ci['accuracy']['high']:.3f}]"
              f"  macro-F1 {ci['macro_f1']['point']:.3f} [{ci['macro_f1']['low']:.3f}, {ci['macro_f1']['high']:.3f}]"
              f"  by slice {m['accuracy_by_slice']}")

    section("3. paired A/B tests on the same 320 tickets")
    c = r["comparison"]
    mc = c["mcnemar"]
    print(f"   McNemar: {c['baseline']} right & {c['candidate']} wrong = {mc['b']}, reverse = {mc['c']}, exact p = {mc['p_exact']:.3g}")
    for metric, p in c["permutation"].items():
        print(f"   permutation {metric}: delta {p['delta']:+.3f} 95% CI [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] p = {p['p_value']:.4f}")

    section("4. MT (EN->DE, 30 hand-written sentences): BLEU and chrF++ via sacrebleu")
    for s, v in r["mt"]["systems"].items():
        print(f"   {s}: BLEU {v['bleu']:.2f}  chrF++ {v['chrf++']:.2f}")
    for metric, p in r["mt"]["paired"].items():
        print(f"   paired bootstrap {metric}: sys_b - sys_a = {p['delta']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]")
    print(f"   signature: {r['mt']['signatures']['chrf++']}")

    section("5. human-review queue (two SIMULATED raters: alice 92%, bob 85% accurate)")
    print("  ", quiet("review", "enqueue", "--audit-rate", "0.05")[1].strip())
    stats = json.loads(quiet("review", "simulate")[1])
    print(f"   reasons {stats['reasons']}")
    for p in stats["pairwise_kappa"]:
        print(f"   Cohen's kappa {p['raters'][0]} vs {p['raters'][1]} = {p['kappa']:.3f} ({p['band']}, n={p['n']})")
    print(f"   rater accuracy vs gold {stats['rater_accuracy_vs_gold']}, needing adjudication: {stats['needs_adjudication']}")
    pending = json.loads(quiet("review", "pending")[1])
    gold = {row[0]: row[1] for row in __import__("sqlite3").connect("reports/review.sqlite").execute("SELECT id, gold FROM items")}
    for item in pending:  # a lead adjudicates; here the lead simply picks the gold label
        quiet("review", "adjudicate", "--item", str(item["id"]), "--rater", "lead", "--label", gold[item["id"]])
    print(f"   adjudicated {len(pending)}; corrections to gold: {len(quiet('review', 'corrections')[1].splitlines())}")

    section("6. CI gate vs stored baseline (baselines/baseline.json = nb-v1)")
    code, out = quiet("gate")
    print(f"   candidate nb-v1  -> exit {code} ({out.splitlines()[0][3:]})")
    code, out = quiet("gate", "--candidate", "nb-small")
    print(f"   candidate nb-small -> exit {code} ({out.splitlines()[0][3:]})")
    for line in out.splitlines():
        if "FAIL" in line and line.startswith("|"):
            print("     " + line)

    section("7. model and system cards")
    quiet("card")
    card = Path("reports/model_card.md").read_text().splitlines()
    print("   reports/model_card.md:", len(card), "lines;", "reports/system_card.md written")
    for line in card:
        if line.startswith(("| Accuracy", "| Macro F1", "- McNemar", "- Cohen")):
            print("   " + line)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    run()
