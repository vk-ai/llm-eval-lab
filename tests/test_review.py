import pytest

from llm_eval_lab import LABELS
from llm_eval_lab.review import ReviewQueue, simulate_rater

ROWS = [
    {"id": "e1", "text": "refund please", "label": "billing"},
    {"id": "e2", "text": "app crashed", "label": "bug"},
    {"id": "e3", "text": "where is my parcel", "label": "shipping"},
    {"id": "e4", "text": "reset password", "label": "account"},
]
A = {"e1": {"pred": "billing"}, "e2": {"pred": "bug"}, "e3": {"pred": "shipping"}, "e4": {"pred": "account"}}
B = {
    "e1": {"pred": "billing", "confidence": 0.95},
    "e2": {"pred": "account", "confidence": 0.9},   # disagreement
    "e3": {"pred": "shipping", "confidence": 0.4},  # low confidence
    "e4": {"pred": "account", "confidence": 0.99},
}


def queue():
    q = ReviewQueue(":memory:", LABELS)
    q.enqueue_from_predictions(ROWS, A, B, confidence_below=0.6, audit_rate=0.0)
    return q


def test_enqueue_reasons_and_idempotence():
    q = queue()
    s = q.stats()
    assert s["items"] == 2 and s["reasons"] == {"disagreement": 1, "low_confidence": 1}
    assert q.enqueue_from_predictions(ROWS, A, B, audit_rate=0.0) == 0  # no duplicates


def test_audit_sample_is_seeded():
    q1, q2 = ReviewQueue(), ReviewQueue()
    n1 = q1.enqueue_from_predictions(ROWS * 1, A, A, audit_rate=0.5, seed=4)
    n2 = q2.enqueue_from_predictions(ROWS * 1, A, A, audit_rate=0.5, seed=4)
    assert n1 == n2


def test_next_item_hides_gold_and_model_output():
    item = queue().next_item("alice")
    assert set(item) == {"id", "example_id", "text"}


def test_each_item_gets_two_independent_labels_then_closes():
    q = queue()
    first = q.next_item("alice")
    q.submit(first["id"], "alice", "bug")
    assert q.next_item("alice")["id"] != first["id"]  # alice moves on
    assert q.next_item("bob")["id"] == first["id"]    # bob still sees it
    q.submit(first["id"], "bob", "bug")
    assert q.next_item("carol")["id"] != first["id"]  # two labels -> closed
    with pytest.raises(ValueError):
        q.submit(first["id"], "carol", "bug")


def test_invalid_label_rejected():
    q = queue()
    with pytest.raises(ValueError):
        q.submit(1, "alice", "spam")
    with pytest.raises(ValueError):
        q.adjudicate(1, "spam", "lead")


def test_disagreement_goes_to_adjudication_and_becomes_correction():
    q = queue()
    ids = {r[0]: r[1] for r in q.db.execute("SELECT example_id, id FROM items")}
    q.submit(ids["e2"], "alice", "bug")
    q.submit(ids["e2"], "bob", "account")
    q.submit(ids["e3"], "alice", "shipping")
    q.submit(ids["e3"], "bob", "shipping")
    pending = q.needs_adjudication()
    assert [p["example_id"] for p in pending] == ["e2"]
    assert q.final_labels() == {"e3": "shipping"}
    q.adjudicate(ids["e2"], "account", "lead", "login crash is an account issue")
    assert q.needs_adjudication() == []
    assert q.corrections() == [{"id": "e2", "label": "account", "was": "bug"}]


def test_kappa_from_simulated_raters():
    q = ReviewQueue(":memory:", LABELS)
    rows = [{"id": f"x{i}", "text": f"t{i}", "label": LABELS[i % 4]} for i in range(200)]
    for r in rows:
        q.enqueue(r["id"], r["text"], r["label"], ["audit"])
    gold = {r[0]: r[1] for r in q.db.execute("SELECT id, gold FROM items")}
    for name, acc in (("alice", 0.95), ("bob", 0.9)):
        rate = simulate_rater(name, acc, LABELS, seed=1)
        while (item := q.next_item(name)) is not None:
            q.submit(item["id"], name, rate(gold[item["id"]]))
    s = q.stats()
    pair = s["pairwise_kappa"][0]
    assert pair["n"] == 200
    assert 0.7 < pair["kappa"] < 0.95
    assert s["rater_accuracy_vs_gold"]["alice"] > s["rater_accuracy_vs_gold"]["bob"] - 0.05
    assert s["fully_labelled"] == 200


def test_persists_to_file(tmp_path):
    db = tmp_path / "q.sqlite"
    q = ReviewQueue(db, LABELS)
    q.enqueue("e1", "t", "bug", ["audit"])
    assert ReviewQueue(db, LABELS).stats()["items"] == 1
