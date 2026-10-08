"""Paired significance tests for comparing two systems on the same examples."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .metrics import _metric_fn, encode


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


@dataclass
class McNemarResult:
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    p_exact: float
    chi2_cc: float
    p_chi2_cc: float

    def to_dict(self) -> dict:
        return {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in asdict(self).items()}


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> McNemarResult:
    """McNemar's test on the discordant pairs.

    ``p_exact`` is the two-sided exact binomial test (recommended when b + c < 25);
    ``p_chi2_cc`` is the chi-squared approximation with continuity correction.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired test needs the same examples for both systems")
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0, 0.0, 1.0)
    p_exact = min(1.0, 2 * _binom_cdf(min(b, c), n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi2 with 1 dof
    return McNemarResult(b, c, p_exact, chi2, p_chi2)


@dataclass
class PairedResult:
    metric: str
    a: float
    b: float
    delta: float  # b - a
    p_value: float
    ci_low: float
    ci_high: float
    n_resamples: int

    def to_dict(self) -> dict:
        return {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in asdict(self).items()}


def permutation_test(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    classes: tuple[str, ...],
    metric: str = "macro_f1",
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedResult:
    """Paired approximate randomisation test on any corpus-level metric.

    Under H0 the two systems are exchangeable per example, so we randomly swap
    their predictions per example and count how often |delta| is at least the
    observed one. Also returns a paired-bootstrap 95% CI for delta.
    """
    t = encode(y_true, classes)
    a = encode(pred_a, classes)
    b = encode(pred_b, classes)
    fn = _metric_fn(metric, len(classes))
    observed = fn(t, b) - fn(t, a)
    rng = np.random.default_rng(seed)
    n = len(t)
    hits = 0
    for _ in range(n_resamples):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        if abs(fn(t, pb) - fn(t, pa)) >= abs(observed) - 1e-12:
            hits += 1
    p = (hits + 1) / (n_resamples + 1)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = fn(t[idx], b[idx]) - fn(t[idx], a[idx])
    return PairedResult(metric, fn(t, a), fn(t, b), observed, p, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), n_resamples)


def paired_bootstrap_corpus(
    score_fn,
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    n_resamples: int = 1000,
    seed: int = 0,
) -> PairedResult:
    """Paired bootstrap for corpus-level scores such as BLEU or chrF++ (Koehn, 2004).

    p is the share of resamples where B does not beat A (one-sided, B > A),
    doubled and capped at 1 for a two-sided reading.
    """
    rng = np.random.default_rng(seed)
    n = len(refs)
    a0, b0 = score_fn(hyps_a, refs), score_fn(hyps_b, refs)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        ra = [refs[j] for j in idx]
        deltas[i] = score_fn([hyps_b[j] for j in idx], ra) - score_fn([hyps_a[j] for j in idx], ra)
    obs = b0 - a0
    if obs >= 0:
        p = 2 * float(np.mean(deltas <= 0))
    else:
        p = 2 * float(np.mean(deltas >= 0))
    return PairedResult("corpus", a0, b0, obs, min(1.0, p), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), n_resamples)
