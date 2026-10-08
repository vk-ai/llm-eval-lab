import json

from llm_eval_lab.datasets import build_all, build_intent, build_train, load_mt_seed, read_jsonl
from conftest import ROOT


def test_same_seed_same_bytes(tmp_path):
    m1 = build_all(tmp_path / "a", "v2", 7, ROOT / "seeds")
    m2 = build_all(tmp_path / "b", "v2", 7, ROOT / "seeds")
    assert m1 == m2
    assert (tmp_path / "a" / "intent_v2.jsonl").read_bytes() == (tmp_path / "b" / "intent_v2.jsonl").read_bytes()


def test_different_seed_changes_hash(tmp_path):
    m1 = build_all(tmp_path / "a", "v2", 7, ROOT / "seeds")
    m2 = build_all(tmp_path / "b", "v2", 8, ROOT / "seeds")
    assert m1["files"]["intent_v2.jsonl"]["sha256"] != m2["files"]["intent_v2.jsonl"]["sha256"]


def test_versions_and_balance():
    v1, v2 = build_intent("v1", 7), build_intent("v2", 7)
    assert len(v1) == 240 and len(v2) == 320
    assert {r["kind"] for r in v1} == {"easy"}
    assert sum(r["kind"] == "hard" for r in v2) == 80
    for rows in (v1, v2):
        counts = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        assert len(set(counts.values())) == 1  # balanced classes


def test_v1_is_prefix_content_of_v2():
    """v2 keeps every v1 ticket (same RNG stream), so v1 results stay comparable on the easy slice."""
    v1 = {(r["id"], r["text"], r["label"]) for r in build_intent("v1", 7)}
    v2 = {(r["id"], r["text"], r["label"]) for r in build_intent("v2", 7) if r["kind"] == "easy"}
    assert v1 == v2


def test_corrections_apply_only_to_v2():
    rows = build_intent("v2", 7)
    target = rows[0]
    new_label = "shipping" if target["label"] != "shipping" else "billing"
    fixed = {r["id"]: r for r in build_intent("v2", 7, [{"id": target["id"], "label": new_label}])}
    assert fixed[target["id"]]["label"] == new_label and fixed[target["id"]]["corrected"]
    assert all("corrected" not in r for r in build_intent("v1", 7, [{"id": target["id"], "label": new_label}]))


def test_train_disjoint_from_test():
    test_ids = {r["id"] for r in build_intent("v2", 7)}
    assert not test_ids & {r["id"] for r in build_train(7)}


def test_unknown_version():
    import pytest

    with pytest.raises(ValueError):
        build_intent("v9", 7)


def test_mt_seed_and_manifest(tmp_path):
    rows = load_mt_seed(ROOT / "seeds" / "mt_en_de.tsv")
    assert len(rows) == 30 and set(rows[0]) == {"id", "src", "ref", "sys_a", "sys_b"}
    build_all(tmp_path, "v2", 7, ROOT / "seeds")
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["files"]["intent_v2.jsonl"]["n"] == len(read_jsonl(tmp_path / "intent_v2.jsonl"))
