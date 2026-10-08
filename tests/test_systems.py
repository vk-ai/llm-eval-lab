import pytest

from llm_eval_lab.datasets import build_intent, build_train
from llm_eval_lab.systems import NaiveBayesSystem, OpenAICompatSystem, RulesSystem, get_system


def test_rules_keywords():
    p = RulesSystem().predict([{"id": "1", "text": "I was charged twice, refund please"}])[0]
    assert p.label == "billing" and p.confidence == 1.0


def test_rules_no_keyword_falls_back_low_confidence():
    p = RulesSystem().predict([{"id": "1", "text": "hmm"}])[0]
    assert p.confidence == 0.25


def test_nb_learns_hard_examples_rules_miss():
    rows = [r for r in build_intent("v2", 7) if r["kind"] == "hard"]
    nb = NaiveBayesSystem().fit(build_train(7))
    acc_nb = sum(p.label == r["label"] for p, r in zip(nb.predict(rows), rows)) / len(rows)
    acc_rules = sum(p.label == r["label"] for p, r in zip(RulesSystem().predict(rows), rows)) / len(rows)
    assert acc_nb > acc_rules + 0.3


def test_nb_confidence_is_a_probability():
    nb = NaiveBayesSystem().fit(build_train(7))
    for p in nb.predict(build_intent("v1", 7)[:20]):
        assert 0.25 <= p.confidence <= 1.0


def test_openai_parse_and_registry():
    assert OpenAICompatSystem.parse(" Billing.") == "billing"
    assert OpenAICompatSystem.parse("no idea") == "bug"
    assert get_system("openai:gpt-x").name == "openai:gpt-x"
    with pytest.raises(ValueError):
        get_system("nope")
