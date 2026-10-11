"""Judge calibration: does the automated judge agree with humans as well as humans agree with each other?

Following the advice in r/LLMDevs ("measure human-human agreement first, compare
judge-human agreement against that baseline") and the r/LangChain sample-size
thread (kappa 0.6 on N=50 has a CI of roughly [0.4, 0.8]):

* ``human_human``  mean pairwise Cohen's kappa between human raters (the ceiling)
* ``judge_human``  mean Cohen's kappa between the judge and each human
* ``gap``          judge_human - human_human

All three get 95% bootstrap CIs from the *same* item resamples, so the gap CI is
paired. The gate fails when there are too few items, when the judge-human CI
lower bound is under a floor, or when the judge trails the human ceiling by more
than ``max_gap``.

In this repo the "judge" is a toy classifier (e.g. ``nb-v1``) and the humans
are simulated. The maths is the same for an LLM judge and real reviewers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .agreement import cohens_kappa, interpret_kappa


@dataclass
class Calibration:
    n: int
    raters: list[str]
    judge_human: float
    human_human: float
    gap: float
    ci: dict[str, tuple[float, float]]
    per_rater: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        r4 = lambda x: round(float(x), 4)  # noqa: E731
        return {
            "n": self.n,
            "raters": self.raters,
            "judge_human_kappa": r4(self.judge_human),
            "human_human_kappa": r4(self.human_human),
            "gap": r4(self.gap),
            "ci95": {k: [r4(a), r4(b)] for k, (a, b) in self.ci.items()},
            "judge_vs_each_rater": {k: r4(v) for k, v in self.per_rater.items()},
            "band": interpret_kappa(self.judge_human),
        }


def _scores(judge: list[str], humans: dict[str, list[str]], idx) -> tuple[float, float]:
    names = sorted(humans)
    j = [judge[i] for i in idx]
    h = {r: [humans[r][i] for i in idx] for r in names}
    jh = float(np.mean([cohens_kappa(j, h[r]) for r in names]))
    hh = float(np.mean([cohens_kappa(h[a], h[b]) for a, b in combinations(names, 2)]))
    return jh, hh


def calibrate(judge: list[str], humans: dict[str, list[str]], n_resamples: int = 1000, seed: int = 0) -> Calibration:
    """``judge[i]`` and ``humans[rater][i]`` are labels for the same item i."""
    if len(humans) < 2:
        raise ValueError("need at least 2 human raters to measure the human-human ceiling")
    n = len(judge)
    if n < 2 or any(len(v) != n for v in humans.values()):
        raise ValueError("judge and every rater must label the same >= 2 items")
    full = range(n)
    jh, hh = _scores(judge, humans, full)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_resamples, 3))
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        a, c = _scores(judge, humans, idx)
        boot[b] = (a, c, a - c)
    q = lambda col: (float(np.quantile(boot[:, col], 0.025)), float(np.quantile(boot[:, col], 0.975)))  # noqa: E731
    return Calibration(
        n=n,
        raters=sorted(humans),
        judge_human=jh,
        human_human=hh,
        gap=jh - hh,
        ci={"judge_human": q(0), "human_human": q(1), "gap": q(2)},
        per_rater={r: cohens_kappa(judge, humans[r]) for r in sorted(humans)},
    )


def from_queue(queue, judge_preds: dict[str, dict], raters: list[str] | None = None) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Pull items every chosen rater labelled from a ReviewQueue, aligned with judge predictions."""
    by_ex: dict[str, dict[str, str]] = {}
    for r in queue.db.execute("SELECT i.example_id, l.rater, l.label FROM labels l JOIN items i ON i.id = l.item_id"):
        by_ex.setdefault(r["example_id"], {})[r["rater"]] = r["label"]
    raters = raters or sorted({x for v in by_ex.values() for x in v})
    ids = sorted(ex for ex, v in by_ex.items() if all(x in v for x in raters) and ex in judge_preds)
    judge = [judge_preds[ex]["pred"] for ex in ids]
    humans = {x: [by_ex[ex][x] for ex in ids] for x in raters}
    return judge, humans, ids


def gate(cal: Calibration, rules: dict) -> tuple[bool, list[dict]]:
    """Rules (gate.yaml ``judge_calibration``): min_items, min_kappa_lower, max_gap."""
    checks = []
    lo = cal.ci["judge_human"][0]
    if "min_items" in rules:
        checks.append({"check": "calibration items", "current": cal.n, "rule": f">= {rules['min_items']}",
                       "ok": cal.n >= rules["min_items"]})
    if "min_kappa_lower" in rules:
        checks.append({"check": "judge-human kappa (95% CI low)", "current": round(lo, 4),
                       "rule": f">= {rules['min_kappa_lower']}", "ok": lo >= rules["min_kappa_lower"]})
    if "max_gap" in rules:
        checks.append({"check": "judge vs human ceiling", "baseline": round(cal.human_human, 4),
                       "current": round(cal.judge_human, 4), "rule": f"gap >= -{rules['max_gap']}",
                       "ok": cal.gap >= -rules["max_gap"]})
    return all(c["ok"] for c in checks), checks


def simulate(n: int, labels: tuple[str, ...], judge_accuracy: float, rater_accuracy: dict[str, float],
             seed: int = 0) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Synthetic calibration set (gold, judge, raters) for tests and the sample-size demo."""
    rng = random.Random(seed)
    gold = [rng.choice(labels) for _ in range(n)]

    def noisy(acc: float, g: str) -> str:
        return g if rng.random() < acc else rng.choice([x for x in labels if x != g])

    judge = [noisy(judge_accuracy, g) for g in gold]
    humans = {r: [noisy(a, g) for g in gold] for r, a in rater_accuracy.items()}
    return judge, humans, gold
