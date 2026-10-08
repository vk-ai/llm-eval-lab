import pytest

from llm_eval_lab.agreement import cohens_kappa, interpret_kappa, kappa_ci


def test_textbook_kappa():
    # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> po 0.7, pe 0.5, kappa 0.4
    r1 = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(r1, r2) == pytest.approx(0.4)
    assert interpret_kappa(0.4) == "fair"


def test_perfect_and_systematic_disagreement():
    assert cohens_kappa(list("abcab"), list("abcab")) == 1.0
    assert cohens_kappa(list("abab"), list("baba")) < 0


def test_constant_raters():
    assert cohens_kappa(["a"] * 5, ["a"] * 5) == 1.0


def test_errors_and_bands():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.85) == "almost perfect"


def test_kappa_ci_brackets_point():
    r1 = list("aabbccabca") * 4
    r2 = list("aabbcbabcc") * 4
    lo, hi = kappa_ci(r1, r2, n_resamples=300)
    assert lo <= cohens_kappa(r1, r2) <= hi
