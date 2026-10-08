import math

import pytest

from llm_eval_lab.metrics import bootstrap_ci, classification_report

C = ("a", "b", "c")


def test_hand_computed_report():
    y = ["a", "a", "a", "b", "b", "c"]
    p = ["a", "a", "b", "b", "c", "c"]
    r = classification_report(y, p, C)
    # a: P=2/2 R=2/3 ; b: P=1/2 R=1/2 ; c: P=1/2 R=1/1
    assert r["per_class"]["a"]["precision"] == 1.0
    assert r["per_class"]["a"]["recall"] == pytest.approx(2 / 3)
    assert r["per_class"]["a"]["f1"] == pytest.approx(0.8)
    assert r["per_class"]["b"]["f1"] == pytest.approx(0.5)
    assert r["per_class"]["c"]["f1"] == pytest.approx(2 / 3)
    assert r["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert r["accuracy"] == r["micro_f1"] == pytest.approx(4 / 6)
    assert r["per_class"]["a"]["support"] == 3


def test_absent_class_scores_zero_not_nan():
    r = classification_report(["a", "a"], ["a", "b"], C)
    assert r["per_class"]["c"]["f1"] == 0.0
    assert not any(math.isnan(v["f1"]) for v in r["per_class"].values())


def test_perfect_predictions_have_degenerate_ci():
    y = ["a", "b", "c"] * 20
    ci = bootstrap_ci(y, y, C, "macro_f1", n_resamples=200)
    assert ci.point == ci.low == ci.high == 1.0


def test_ci_contains_point_and_shrinks_with_n():
    small_y = (["a"] * 8 + ["b"] * 8 + ["c"] * 8)
    small_p = (["a"] * 6 + ["b"] * 2 + ["b"] * 6 + ["c"] * 2 + ["c"] * 6 + ["a"] * 2)
    ci_small = bootstrap_ci(small_y, small_p, C, "accuracy", n_resamples=400, seed=1)
    ci_big = bootstrap_ci(small_y * 10, small_p * 10, C, "accuracy", n_resamples=400, seed=1)
    assert ci_small.low <= ci_small.point <= ci_small.high
    assert ci_big.point == pytest.approx(ci_small.point)
    assert (ci_big.high - ci_big.low) < (ci_small.high - ci_small.low) / 2


def test_ci_is_seeded():
    y = ["a", "b", "c", "a", "b"] * 5
    p = ["a", "b", "b", "a", "c"] * 5
    assert bootstrap_ci(y, p, C, seed=3) == bootstrap_ci(y, p, C, seed=3)
