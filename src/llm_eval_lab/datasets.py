"""Deterministic, versioned test-set builder.

The intent data is synthetic: tickets are generated from templates with a
seeded RNG, so a given (version, seed) always produces byte-identical files.
DVC pins the resulting hashes in ``dvc.lock``; CI re-runs the stage and
checks that nothing changed.

Versions
--------
v1  240 "easy" tickets (keyword-bearing templates).
v2  v1 + 80 "hard" tickets (paraphrases without the obvious keywords and
    mixed-intent tickets labelled with their primary intent), plus any
    adjudicated label corrections from ``seeds/corrections.jsonl``.

The MT set is hand-written (``seeds/mt_en_de.tsv``) and copied into the test
set directory with an ID per line.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import LABELS

EASY = {
    "billing": [
        "I was charged twice for my {plan} subscription this month",
        "Please refund the {amount} charge on my card",
        "My invoice shows the wrong amount for {month}",
        "Why did my bill go up to {amount}?",
        "I need a copy of the invoice for {month}",
        "The payment for my {plan} plan failed but money left my account",
    ],
    "bug": [
        "The app crashes when I open the {screen} screen",
        "I get error {code} every time I try to save",
        "The {screen} page is broken and shows a blank screen",
        "Export to CSV fails with error {code}",
        "The search button does nothing on the {screen} page",
        "Notifications stopped working after the last update",
    ],
    "account": [
        "I can't log in to my account, it says wrong password",
        "How do I reset my password?",
        "Please delete my account and all my data",
        "I want to change the email address on my account",
        "My account got locked after too many login attempts",
        "How do I enable two-factor authentication on my account?",
    ],
    "shipping": [
        "My order {order} hasn't arrived yet",
        "Where is my package? Tracking for {order} hasn't updated",
        "The parcel was delivered to the wrong address",
        "Can I change the delivery address for order {order}?",
        "My shipment {order} arrived damaged",
        "How long does delivery to {city} usually take?",
    ],
}

# No obvious keywords, or two intents in one ticket (label = primary intent).
HARD = {
    "billing": [
        "I don't recognise the {amount} that left my account yesterday",
        "Can you explain this line item from {month}? It doesn't match what I agreed to",
        "You took money from me after I cancelled, and the app also froze when I tried to check",
        "Switching from {plan} to the cheaper option, will I get anything back for unused days?",
    ],
    "bug": [
        "Nothing happens when I tap the {screen} tab, I've reinstalled twice",
        "The numbers on the {screen} dashboard are clearly wrong since Monday",
        "Since the update the app logs me out whenever I open {screen}",
        "Uploading a photo just spins forever",
    ],
    "account": [
        "I no longer have access to the phone number linked to my profile",
        "Someone else may have got into my profile, please help",
        "I'd like my personal data removed under GDPR",
        "The verification code never reaches me so I can't get in",
    ],
    "shipping": [
        "It's been two weeks and still nothing at my door for {order}",
        "The courier says they tried but nobody came to {city}",
        "Half the items in {order} were missing from the box",
        "I was charged for express but {order} came by standard post",
    ],
}

FILL = {
    "plan": ["Pro", "Basic", "Team", "Enterprise"],
    "amount": ["$19.99", "$49", "EUR 120", "$7.50", "GBP 30"],
    "month": ["March", "April", "May", "June", "July"],
    "screen": ["settings", "reports", "profile", "billing history", "inbox"],
    "code": ["E1042", "500", "TIMEOUT-7", "ERR_SYNC", "403"],
    "order": ["#A1023", "#B7781", "#C3310", "#D9054", "#E1209"],
    "city": ["Bangalore", "Berlin", "Leeds", "Austin", "Lyon"],
}

PREFIX = ["", "", "", "Hi, ", "Hello team, ", "Urgent: ", "Quick question - "]
SUFFIX = ["", "", "", " Thanks.", " Please help.", " This is the second time I'm asking."]


def _render(rng: random.Random, template: str) -> str:
    out = template
    for key, values in FILL.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(values), 1)
    return f"{rng.choice(PREFIX)}{out}{rng.choice(SUFFIX)}"


def _make(rng: random.Random, pool: dict[str, list[str]], per_label: int, kind: str, start: int) -> list[dict]:
    rows = []
    i = start
    for label in LABELS:
        for _ in range(per_label):
            rows.append({"id": f"t{i:04d}", "text": _render(rng, rng.choice(pool[label])), "label": label, "kind": kind})
            i += 1
    return rows


def build_intent(version: str, seed: int, corrections: list[dict] | None = None) -> list[dict]:
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown test-set version {version!r}")
    rng = random.Random(seed)
    rows = _make(rng, EASY, 60, "easy", 0)
    if version == "v2":
        rows += _make(rng, HARD, 20, "hard", len(rows))
        fixes = {c["id"]: c["label"] for c in (corrections or [])}
        for row in rows:
            if row["id"] in fixes:
                row["label"] = fixes[row["id"]]
                row["corrected"] = True
    rng.shuffle(rows)
    return rows


def build_train(seed: int) -> list[dict]:
    """Training data for the naive-Bayes system. Disjoint RNG stream from the test set."""
    rng = random.Random(seed + 1000)
    rows = _make(rng, EASY, 40, "easy", 5000)
    rows += _make(rng, HARD, 6, "hard", 5000 + len(rows))
    rng.shuffle(rows)
    return rows


def load_mt_seed(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        src, ref, sys_a, sys_b = line.split("\t")
        rows.append({"id": f"mt{len(rows):03d}", "src": src, "ref": ref, "sys_a": sys_a, "sys_b": sys_b})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_all(out_dir: Path, version: str, seed: int, seeds_dir: Path) -> dict:
    out_dir = Path(out_dir)
    corrections_path = Path(seeds_dir) / "corrections.jsonl"
    corrections = read_jsonl(corrections_path) if corrections_path.exists() else []
    intent = build_intent(version, seed, corrections)
    files = {
        f"intent_{version}.jsonl": intent,
        "intent_train.jsonl": build_train(seed),
        "mt_en_de_v1.jsonl": load_mt_seed(Path(seeds_dir) / "mt_en_de.tsv"),
    }
    manifest = {"version": version, "seed": seed, "files": {}}
    for name, rows in files.items():
        digest = write_jsonl(out_dir / name, rows)
        entry = {"sha256": digest, "n": len(rows)}
        if name.startswith("intent_"):
            counts: dict[str, int] = {}
            for r in rows:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            entry["labels"] = dict(sorted(counts.items()))
        manifest["files"][name] = entry
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
