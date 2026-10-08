"""Systems under test.

These are deliberately small, deterministic classifiers so the statistics in
this repo can run offline in CI. They stand in for "model A" and "model B" in
an A/B comparison. ``OpenAICompatSystem`` lets you point the same harness at a
real chat model, but it is not exercised against a live API in this repo.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from . import LABELS

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Prediction:
    id: str
    label: str
    confidence: float

    def to_dict(self) -> dict:
        return {"id": self.id, "pred": self.label, "confidence": round(self.confidence, 4)}


class RulesSystem:
    """Keyword rules: the kind of baseline a team ships first."""

    name = "rules-v1"
    KEYWORDS = {
        "billing": {"charged", "refund", "invoice", "bill", "payment", "charge", "price", "subscription"},
        "bug": {"crash", "crashes", "error", "broken", "fails", "bug", "blank", "stopped", "update"},
        "account": {"login", "log", "password", "account", "locked", "email", "authentication", "delete"},
        "shipping": {"order", "package", "parcel", "delivery", "delivered", "shipment", "tracking", "arrived"},
    }

    def predict(self, rows: list[dict]) -> list[Prediction]:
        out = []
        for row in rows:
            toks = set(tokenize(row["text"]))
            scores = {lab: len(toks & kws) for lab, kws in self.KEYWORDS.items()}
            best = max(LABELS, key=lambda lab: (scores[lab], -LABELS.index(lab)))
            total = sum(scores.values())
            conf = scores[best] / total if total else 0.25
            out.append(Prediction(row["id"], best if total else "bug", conf))
        return out


class NaiveBayesSystem:
    """Multinomial naive Bayes with Laplace smoothing on unigrams + bigrams."""

    def __init__(self, name: str = "nb-v1", alpha: float = 1.0, max_train: int | None = None):
        self.name = name
        self.alpha = alpha
        self.max_train = max_train
        self.priors: dict[str, float] = {}
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    @staticmethod
    def features(text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def fit(self, rows: list[dict]) -> "NaiveBayesSystem":
        rows = rows[: self.max_train] if self.max_train else rows
        by_label: dict[str, Counter] = defaultdict(Counter)
        n_label: Counter = Counter()
        for row in rows:
            feats = self.features(row["text"])
            by_label[row["label"]].update(feats)
            n_label[row["label"]] += 1
            self.vocab.update(feats)
        n = sum(n_label.values())
        self.priors = {lab: math.log((n_label[lab] + 1) / (n + len(LABELS))) for lab in LABELS}
        self.counts = {lab: by_label[lab] for lab in LABELS}
        self.totals = {lab: sum(by_label[lab].values()) for lab in LABELS}
        return self

    def predict(self, rows: list[dict]) -> list[Prediction]:
        v = len(self.vocab) or 1
        out = []
        for row in rows:
            feats = [f for f in self.features(row["text"]) if f in self.vocab]
            logp = {}
            for lab in LABELS:
                denom = self.totals[lab] + self.alpha * v
                logp[lab] = self.priors[lab] + sum(math.log((self.counts[lab][f] + self.alpha) / denom) for f in feats)
            m = max(logp.values())
            z = sum(math.exp(x - m) for x in logp.values())
            best = max(LABELS, key=lambda lab: logp[lab])
            out.append(Prediction(row["id"], best, math.exp(logp[best] - m) / z))
        return out


class OpenAICompatSystem:
    """Zero-shot classification through any OpenAI-compatible /chat/completions endpoint.

    Uses only the standard library so it adds no dependency. Not run in CI.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0):
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def _ask(self, text: str) -> str:
        import urllib.request

        prompt = (
            "Classify the customer-support ticket into exactly one of: "
            + ", ".join(LABELS)
            + ". Answer with the label only.\n\nTicket: "
            + text
        )
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-configured URL
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    @staticmethod
    def parse(answer: str) -> str:
        low = answer.strip().lower()
        for lab in LABELS:
            if re.search(rf"\b{lab}\b", low):
                return lab
        return "bug"

    def predict(self, rows: list[dict]) -> list[Prediction]:
        return [Prediction(r["id"], self.parse(self._ask(r["text"])), 1.0) for r in rows]


def get_system(name: str, train_rows: list[dict] | None = None):
    if name == "rules-v1":
        return RulesSystem()
    if name == "nb-v1":
        return NaiveBayesSystem("nb-v1").fit(train_rows or [])
    if name == "nb-small":  # deliberately under-trained candidate, used to show the gate failing
        return NaiveBayesSystem("nb-small", max_train=24).fit(train_rows or [])
    if name.startswith("openai:"):
        return OpenAICompatSystem(name.split(":", 1)[1])
    raise ValueError(f"unknown system {name!r}")
