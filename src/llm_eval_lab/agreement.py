"""Inter-rater agreement."""
from __future__ import annotations

from collections import Counter

import numpy as np


def cohens_kappa(r1: list[str], r2: list[str]) -> float:
    """Cohen's kappa for two raters on the same items: (p_o - p_e) / (1 - p_e)."""
    if len(r1) != len(r2):
        raise ValueError("raters must label the same items")
    n = len(r1)
    if n == 0:
        raise ValueError("no items")
    po = sum(a == b for a, b in zip(r1, r2)) / n
    c1, c2 = Counter(r1), Counter(r2)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_ci(r1: list[str], r2: list[str], n_resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(r1)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        vals.append(cohens_kappa([r1[i] for i in idx], [r2[i] for i in idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands. A convention, not a law."""
    if k < 0:
        return "poor"
    for limit, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")):
        if k <= limit:
            return name
    return "almost perfect"
