"""Human-review queue backed by SQLite.

Items are enqueued when two systems disagree, when the candidate's confidence
is low, or by a seeded random audit sample. Each item is labelled
independently by ``raters_per_item`` reviewers (who never see each other's
labels or the model output), agreement is measured with Cohen's kappa, rater
disagreements go to adjudication, and adjudicated labels that contradict the
test-set gold become corrections for the next test-set version.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .agreement import cohens_kappa, interpret_kappa

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY,
  example_id TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  gold TEXT NOT NULL,
  reasons TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
  item_id INTEGER NOT NULL REFERENCES items(id),
  rater TEXT NOT NULL,
  label TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (item_id, rater)
);
CREATE TABLE IF NOT EXISTS adjudications (
  item_id INTEGER PRIMARY KEY REFERENCES items(id),
  label TEXT NOT NULL,
  adjudicator TEXT NOT NULL,
  note TEXT,
  created_at REAL NOT NULL
);
"""


class ReviewQueue:
    def __init__(self, path: str | Path = ":memory:", labels: tuple[str, ...] | None = None, raters_per_item: int = 2):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.labels = labels
        self.raters_per_item = raters_per_item

    # -- intake ---------------------------------------------------------------
    def enqueue_from_predictions(
        self,
        rows: list[dict],
        preds_a: dict[str, dict],
        preds_b: dict[str, dict],
        confidence_below: float = 0.6,
        audit_rate: float = 0.05,
        seed: int = 0,
    ) -> int:
        rng = random.Random(seed)
        added = 0
        for row in rows:
            reasons = []
            a, b = preds_a.get(row["id"]), preds_b.get(row["id"])
            if a and b and a["pred"] != b["pred"]:
                reasons.append("disagreement")
            if b and b.get("confidence", 1.0) < confidence_below:
                reasons.append("low_confidence")
            if rng.random() < audit_rate:
                reasons.append("audit")
            if reasons:
                added += self.enqueue(row["id"], row["text"], row["label"], reasons)
        return added

    def enqueue(self, example_id: str, text: str, gold: str, reasons: list[str]) -> int:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO items(example_id, text, gold, reasons, created_at) VALUES (?,?,?,?,?)",
            (example_id, text, gold, json.dumps(reasons), time.time()),
        )
        self.db.commit()
        return cur.rowcount

    # -- labelling ------------------------------------------------------------
    def next_item(self, rater: str) -> dict | None:
        """Next item this rater hasn't labelled that still needs labels. Gold and model output are hidden."""
        row = self.db.execute(
            """
            SELECT i.id, i.example_id, i.text FROM items i
            WHERE i.id NOT IN (SELECT item_id FROM labels WHERE rater = ?)
              AND (SELECT COUNT(*) FROM labels l WHERE l.item_id = i.id) < ?
            ORDER BY i.id LIMIT 1
            """,
            (rater, self.raters_per_item),
        ).fetchone()
        return dict(row) if row else None

    def submit(self, item_id: int, rater: str, label: str) -> None:
        if self.labels and label not in self.labels:
            raise ValueError(f"label {label!r} not in {self.labels}")
        n = self.db.execute("SELECT COUNT(*) FROM labels WHERE item_id = ? AND rater != ?", (item_id, rater)).fetchone()[0]
        if n >= self.raters_per_item:
            raise ValueError(f"item {item_id} already has {n} labels")
        self.db.execute(
            "INSERT OR REPLACE INTO labels(item_id, rater, label, created_at) VALUES (?,?,?,?)",
            (item_id, rater, label, time.time()),
        )
        self.db.commit()

    def needs_adjudication(self) -> list[dict]:
        rows = self.db.execute(
            """
            SELECT i.id, i.example_id, i.text, GROUP_CONCAT(l.rater || '=' || l.label, ', ') AS votes
            FROM items i JOIN labels l ON l.item_id = i.id
            WHERE i.id NOT IN (SELECT item_id FROM adjudications)
            GROUP BY i.id HAVING COUNT(*) >= ? AND COUNT(DISTINCT l.label) > 1
            ORDER BY i.id
            """,
            (self.raters_per_item,),
        ).fetchall()
        return [dict(r) for r in rows]

    def adjudicate(self, item_id: int, label: str, adjudicator: str, note: str = "") -> None:
        if self.labels and label not in self.labels:
            raise ValueError(f"label {label!r} not in {self.labels}")
        self.db.execute(
            "INSERT OR REPLACE INTO adjudications(item_id, label, adjudicator, note, created_at) VALUES (?,?,?,?,?)",
            (item_id, label, adjudicator, note, time.time()),
        )
        self.db.commit()

    def final_labels(self) -> dict[str, str]:
        """example_id -> agreed label (unanimous raters) or adjudicated label."""
        out = {}
        for r in self.db.execute(
            """
            SELECT i.example_id, a.label AS adj, GROUP_CONCAT(DISTINCT l.label) AS ls, COUNT(l.label) AS n
            FROM items i LEFT JOIN labels l ON l.item_id = i.id LEFT JOIN adjudications a ON a.item_id = i.id
            GROUP BY i.id
            """
        ):
            if r["adj"]:
                out[r["example_id"]] = r["adj"]
            elif r["n"] >= self.raters_per_item and r["ls"] and "," not in r["ls"]:
                out[r["example_id"]] = r["ls"]
        return out

    def corrections(self) -> list[dict]:
        """Final labels that disagree with the test-set gold, ready for seeds/corrections.jsonl."""
        gold = {r["example_id"]: r["gold"] for r in self.db.execute("SELECT example_id, gold FROM items")}
        return [
            {"id": ex, "label": lab, "was": gold[ex]}
            for ex, lab in sorted(self.final_labels().items())
            if lab != gold[ex]
        ]

    # -- reporting ------------------------------------------------------------
    def stats(self) -> dict:
        by_item: dict[int, dict[str, str]] = defaultdict(dict)
        for r in self.db.execute("SELECT item_id, rater, label FROM labels"):
            by_item[r["item_id"]][r["rater"]] = r["label"]
        gold = {r["id"]: r["gold"] for r in self.db.execute("SELECT id, gold FROM items")}
        raters = sorted({rater for v in by_item.values() for rater in v})
        pairs = []
        for r1, r2 in combinations(raters, 2):
            shared = [i for i, v in by_item.items() if r1 in v and r2 in v]
            if len(shared) >= 2:
                k = cohens_kappa([by_item[i][r1] for i in shared], [by_item[i][r2] for i in shared])
                pairs.append({"raters": [r1, r2], "n": len(shared), "kappa": round(k, 4), "band": interpret_kappa(k)})
        vs_gold = {}
        for rater in raters:
            items = [i for i, v in by_item.items() if rater in v]
            if items:
                vs_gold[rater] = round(sum(by_item[i][rater] == gold[i] for i in items) / len(items), 4)
        total = self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        done = sum(1 for v in by_item.values() if len(v) >= self.raters_per_item)
        reasons: dict[str, int] = defaultdict(int)
        for r in self.db.execute("SELECT reasons FROM items"):
            for reason in json.loads(r["reasons"]):
                reasons[reason] += 1
        return {
            "items": total,
            "fully_labelled": done,
            "reasons": dict(sorted(reasons.items())),
            "pairwise_kappa": pairs,
            "rater_accuracy_vs_gold": vs_gold,
            "needs_adjudication": len(self.needs_adjudication()),
            "adjudicated": self.db.execute("SELECT COUNT(*) FROM adjudications").fetchone()[0],
        }


def simulate_rater(name: str, accuracy: float, labels: tuple[str, ...], seed: int):
    """A simulated reviewer for demos and tests: right with probability ``accuracy``."""
    rng = random.Random(f"{name}-{seed}")

    def label(gold: str) -> str:
        if rng.random() < accuracy:
            return gold
        return rng.choice([x for x in labels if x != gold])

    return label
