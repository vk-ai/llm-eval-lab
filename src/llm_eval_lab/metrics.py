"""Classification metrics with bootstrap confidence intervals (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def encode(labels: list[str], classes: tuple[str, ...]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[x] for x in labels], dtype=np.int64)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """k x k confusion matrix, rows = true, cols = predicted."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def prf_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(true_pos > 0, tp / true_pos, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    n = cm.sum()
    micro = tp.sum() / n if n else 0.0  # micro P = micro R = micro F1 = accuracy for single-label multiclass
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_pos,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(micro),
        "accuracy": float(micro),
    }


def classification_report(y_true: list[str], y_pred: list[str], classes: tuple[str, ...]) -> dict:
    k = len(classes)
    r = prf_from_confusion(confusion(encode(y_true, classes), encode(y_pred, classes), k))
    return {
        "accuracy": r["accuracy"],
        "macro_precision": r["macro_precision"],
        "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"],
        "micro_f1": r["micro_f1"],
        "per_class": {
            c: {
                "precision": float(r["precision"][i]),
                "recall": float(r["recall"][i]),
                "f1": float(r["f1"][i]),
                "support": int(r["support"][i]),
            }
            for i, c in enumerate(classes)
        },
    }


def _metric_fn(name: str, k: int):
    def fn(t: np.ndarray, p: np.ndarray) -> float:
        return prf_from_confusion(confusion(t, p, k))[name]

    return fn


@dataclass(frozen=True)
class CI:
    point: float
    low: float
    high: float
    level: float = 0.95

    def to_dict(self) -> dict:
        return {"point": round(self.point, 4), "low": round(self.low, 4), "high": round(self.high, 4), "level": self.level}


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    classes: tuple[str, ...],
    metric: str = "macro_f1",
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI: resample examples with replacement, recompute the metric."""
    t = encode(y_true, classes)
    p = encode(y_pred, classes)
    fn = _metric_fn(metric, len(classes))
    rng = np.random.default_rng(seed)
    n = len(t)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[b] = fn(t[idx], p[idx])
    alpha = (1 - level) / 2
    return CI(fn(t, p), float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)), level)
