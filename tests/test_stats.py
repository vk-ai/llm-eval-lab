import pytest

from llm_eval_lab.stats import mcnemar, paired_bootstrap_corpus, permutation_test

C = ("x", "y")


def test_mcnemar_hand_values():
    a = [True] * 10 + [False] * 2 + [True] * 30 + [False] * 5
    b = [False] * 10 + [True] * 2 + [True] * 30 + [False] * 5
    r = mcnemar(a, b)
    assert (r.b, r.c) == (10, 2)
    assert r.p_exact == pytest.approx(2 * (1 + 12 + 66) / 4096)  # 0.03857
    assert r.chi2_cc == pytest.approx(49 / 12)
    assert r.p_chi2_cc == pytest.approx(0.04331, abs=1e-4)


def test_mcnemar_no_discordant_pairs():
    r = mcnemar([True, False], [True, False])
    assert r.p_exact == 1.0 and r.b == r.c == 0


def test_mcnemar_needs_pairs():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_permutation_identical_systems():
    y = ["x", "y"] * 20
    p = ["x", "x"] * 20
    r = permutation_test(y, p, p, C, "accuracy", n_resamples=200)
    assert r.delta == 0 and r.p_value == 1.0


def test_permutation_detects_real_difference_and_ci_excludes_zero():
    y = ["x", "y"] * 50
    weak = ["x"] * 100            # 50% accuracy
    strong = list(y[:90]) + ["x"] * 10  # 95% accuracy
    r = permutation_test(y, weak, strong, C, "accuracy", n_resamples=500)
    assert r.delta == pytest.approx(0.45)
    assert r.p_value < 0.01
    assert r.ci_low > 0


def test_permutation_small_difference_not_significant():
    y = ["x", "y"] * 15
    a = list(y)
    b = list(y)
    b[0] = "y"  # one extra error
    r = permutation_test(y, a, b, C, "accuracy", n_resamples=500)
    assert r.p_value > 0.5


def test_paired_bootstrap_corpus_direction():
    refs = ["a b c d"] * 20
    good = ["a b c d"] * 20
    bad = ["a x y d"] * 20

    def score(h, r):
        return sum(x == y for x, y in zip(h, r)) / len(r)

    r = paired_bootstrap_corpus(score, bad, good, refs, n_resamples=200)
    assert r.delta == 1.0 and r.p_value == 0.0
