import json

import pytest

from llm_eval_lab import LABELS
from llm_eval_lab.agreement import cohens_kappa
from llm_eval_lab.calibration import calibrate, from_queue, gate, simulate
from llm_eval_lab.review import ReviewQueue

RULES = {"min_items": 150, "min_kappa_lower": 0.6, "max_gap": 0.05}


def test_hand_example_matches_cohens_kappa():
    judge = ["a", "a", "b", "b", "a", "b"]
    h = {"x": ["a", "a", "b", "a", "a", "b"], "y": ["a", "b", "b", "b", "a", "b"]}
    cal = calibrate(judge, h, n_resamples=50)
    assert cal.judge_human == pytest.approx((cohens_kappa(judge, h["x"]) + cohens_kappa(judge, h["y"])) / 2)
    assert cal.human_human == pytest.approx(cohens_kappa(h["x"], h["y"]))
    assert cal.gap == pytest.approx(cal.judge_human - cal.human_human)
    for k, (lo, hi) in cal.ci.items():
        assert lo <= hi


def test_needs_two_raters_and_aligned_items():
    with pytest.raises(ValueError):
        calibrate(["a", "b"], {"x": ["a", "b"]})
    with pytest.raises(ValueError):
        calibrate(["a", "b"], {"x": ["a", "b"], "y": ["a"]})


def test_ci_shrinks_with_sample_size():
    """The r/LangChain point: N=50 gives a wide CI, N=400 a much tighter one."""
    widths = {}
    for n in (50, 400):
        j, h, _ = simulate(n, LABELS, 0.85, {"a": 0.9, "b": 0.9}, seed=1)
        lo, hi = calibrate(j, h, n_resamples=300).ci["judge_human"]
        widths[n] = hi - lo
    assert widths[50] > 2 * widths[400]
    assert widths[50] > 0.15


def test_gate_passes_judge_as_good_as_humans():
    j, h, _ = simulate(300, LABELS, 0.92, {"a": 0.9, "b": 0.9, "c": 0.9}, seed=2)
    ok, checks = gate(calibrate(j, h, n_resamples=300), RULES)
    assert ok, checks


def test_gate_fails_judge_below_human_ceiling():
    j, h, _ = simulate(300, LABELS, 0.75, {"a": 0.93, "b": 0.93}, seed=3)
    cal = calibrate(j, h, n_resamples=300)
    ok, checks = gate(cal, RULES)
    assert not ok and cal.gap < -0.05
    assert {c["check"]: c["ok"] for c in checks}["judge vs human ceiling"] is False


def test_gate_fails_on_too_few_items_even_if_kappa_looks_good():
    j, h, _ = simulate(40, LABELS, 0.95, {"a": 0.95, "b": 0.95}, seed=4)
    ok, checks = gate(calibrate(j, h, n_resamples=300), RULES)
    assert not ok and not checks[0]["ok"]


def test_from_queue_aligns_complete_items_only():
    q = ReviewQueue(":memory:", LABELS, raters_per_item=2)
    for i, lab in enumerate(["bug", "billing", "account"]):
        q.enqueue(f"e{i}", "t", lab, ["audit"])
    q.submit(1, "alice", "bug"); q.submit(1, "bob", "bug")
    q.submit(2, "alice", "billing"); q.submit(2, "bob", "account")
    q.submit(3, "alice", "account")  # bob missing -> excluded
    judge, humans, ids = from_queue(q, {"e0": {"pred": "bug"}, "e1": {"pred": "billing"}, "e2": {"pred": "account"}})
    assert ids == ["e0", "e1"] and judge == ["bug", "billing"]
    assert humans == {"alice": ["bug", "billing"], "bob": ["bug", "account"]}


def test_cli_calibrate_end_to_end(evaluated):
    from llm_eval_lab.cli import main

    assert main(["calibrate", "--judge", "nb-v1", "--simulate-sample", "320", "--resamples", "200"]) == 0
    out = json.loads((evaluated / "reports" / "calibration.json").read_text())
    assert out["passed"] and out["n"] == 320 and out["simulated_raters"]
    assert out["raters"] == ["alice", "bob", "carol"]
    # the weak keyword system must not pass as a judge
    assert main(["calibrate", "--judge", "rules-v1", "--simulate-sample", "320", "--resamples", "200"]) == 1
    # and a 50-item calibration set is rejected as too small
    assert main(["calibrate", "--judge", "nb-v1", "--simulate-sample", "50", "--resamples", "200"]) == 1
