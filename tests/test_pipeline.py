"""End-to-end through the CLI: test sets -> predictions -> eval -> baseline -> gate -> cards -> review."""
import json
from pathlib import Path

from llm_eval_lab.cli import main
from llm_eval_lab.gate import load_json, run_gate


def test_evaluate_report_shape(evaluated):
    report = load_json("reports/eval.json")
    assert report["testset"]["n"] == 320 and report["testset"]["version"] == "v2"
    nb, rules = report["systems"]["nb-v1"], report["systems"]["rules-v1"]
    assert nb["macro_f1"] > rules["macro_f1"]
    ci = nb["ci95"]["macro_f1"]
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert rules["accuracy_by_slice"]["hard"] < rules["accuracy_by_slice"]["easy"]
    comp = report["comparison"]
    assert comp["mcnemar"]["c"] > comp["mcnemar"]["b"] and comp["mcnemar"]["p_exact"] < 0.001
    flat = json.loads(Path("reports/metrics.json").read_text())
    assert set(flat) >= {"nb-v1", "rules-v1", "mt_sys_a", "comparison"}


def test_committed_baseline_matches_fresh_run(evaluated):
    """The baseline in the repo was produced from this exact test set; CI relies on that."""
    report = load_json("reports/eval.json")
    baseline = load_json("baselines/baseline.json")
    assert baseline["testset"]["sha256"] == report["testset"]["sha256"]
    assert main(["gate"]) == 0


def test_gate_fails_for_weaker_candidate(evaluated, capsys):
    assert main(["gate", "--candidate", "nb-small"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "McNemar" in out
    assert not Path("runs/nb-small.jsonl").exists()  # DVC-tracked runs/ untouched


def test_gate_writes_github_summary(evaluated, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main(["gate"])
    assert "Eval gate: PASSED" in summary.read_text()


def test_gate_blocks_silent_testset_change(evaluated):
    report = load_json("reports/eval.json")
    baseline = load_json("baselines/baseline.json")
    baseline["testset"]["sha256"] = "0" * 64
    rules = {"max_drop": {"macro_f1": 0.02}}
    res = run_gate(report, Path("runs"), baseline, rules, Path("data/testsets"))
    assert not res.passed and "test set changed" in res.notes[0]
    rules["allow_testset_change"] = True
    rules["significance"] = {"alpha": 0.05}
    res = run_gate(report, Path("runs"), baseline, rules, Path("data/testsets"))
    assert res.passed and "paired test skipped" in res.notes[0]


def test_gate_tolerates_small_drop_but_not_large(evaluated):
    report = load_json("reports/eval.json")
    baseline = load_json("baselines/baseline.json")
    baseline["metrics"]["macro_f1"] += 0.01
    assert run_gate(report, Path("runs"), baseline, {"max_drop": {"macro_f1": 0.02}}, Path("data/testsets")).passed
    baseline["metrics"]["macro_f1"] += 0.05
    assert not run_gate(report, Path("runs"), baseline, {"max_drop": {"macro_f1": 0.02}}, Path("data/testsets")).passed


def test_cards_and_review_loop(evaluated, capsys):
    assert main(["review", "enqueue", "--audit-rate", "0.05"]) == 0
    assert main(["review", "simulate"]) == 0
    stats = json.loads(capsys.readouterr().out.split("enqueued")[1].split("\n", 1)[1])
    assert stats["pairwise_kappa"] and stats["fully_labelled"] == stats["items"] > 0
    assert main(["review", "corrections"]) == 0
    assert main(["card"]) == 0
    card = Path("reports/model_card.md").read_text()
    for section in ("## Model details", "## Intended use", "## Evaluation data", "95% bootstrap CI", "McNemar", "Cohen's kappa", "## Caveats"):
        assert section in card
    system = Path("reports/system_card.md").read_text()
    assert "chrF++" in system and "BLEU" in system and "not run" in system


def test_review_cli_label_and_adjudicate(evaluated, capsys):
    main(["review", "enqueue", "--audit-rate", "0"])
    capsys.readouterr()
    main(["review", "next", "--rater", "me"])
    item = json.loads(capsys.readouterr().out)
    main(["review", "label", "--item", str(item["id"]), "--rater", "me", "--label", "bug"])
    main(["review", "label", "--item", str(item["id"]), "--rater", "you", "--label", "billing"])
    main(["review", "pending"])
    pending = json.loads(capsys.readouterr().out)
    assert pending[0]["id"] == item["id"]
    main(["review", "adjudicate", "--item", str(item["id"]), "--rater", "lead", "--label", "bug"])
    main(["review", "stats"])
    assert json.loads(capsys.readouterr().out)["adjudicated"] == 1
