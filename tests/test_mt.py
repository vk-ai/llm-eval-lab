import importlib.util

import pytest

from llm_eval_lab.datasets import load_mt_seed
from llm_eval_lab.mt import CometUnavailable, bleu, chrf_pp, comet, evaluate_mt
from conftest import ROOT


def test_identity_scores_100():
    refs = ["Das ist ein Test.", "Noch ein Satz hier."]
    assert bleu(refs, refs) == pytest.approx(100.0)
    assert chrf_pp(refs, refs) == pytest.approx(100.0)


def test_chrf_pp_rewards_partial_matches_more_than_bleu():
    refs = ["Die Lieferung dauert in der Regel drei bis fünf Werktage."]
    hyp = ["Lieferung nimmt normalerweise drei bis fünf Arbeitstage."]
    assert 0 < bleu(hyp, refs) < chrf_pp(hyp, refs) < 100


def test_fluent_system_beats_literal_system():
    rows = load_mt_seed(ROOT / "seeds" / "mt_en_de.tsv")
    r = evaluate_mt(rows, n_resamples=200)
    a, b = r["systems"]["sys_a"], r["systems"]["sys_b"]
    assert a["bleu"] > b["bleu"] and a["chrf++"] > b["chrf++"]
    assert r["paired"]["chrf++"]["ci_high"] < 0  # sys_b - sys_a is negative with 95% CI
    assert "nrefs:1" in r["signatures"]["bleu"] and "nc:6|nw:2" in r["signatures"]["chrf++"]


@pytest.mark.skipif(importlib.util.find_spec("comet") is not None, reason="COMET installed; covered by the comet marker")
def test_comet_unavailable_is_explicit():
    with pytest.raises(CometUnavailable):
        comet(["a"], ["b"], ["c"])
    r = evaluate_mt(load_mt_seed(ROOT / "seeds" / "mt_en_de.tsv")[:3], n_resamples=10, use_comet=True)
    assert r["systems"]["sys_a"]["comet"] is None and "not installed" in r["comet_note"]


@pytest.mark.comet
@pytest.mark.skipif(importlib.util.find_spec("comet") is None, reason="pip install '.[comet]' to run COMET")
def test_comet_ranks_fluent_system_higher():  # pragma: no cover - optional
    rows = load_mt_seed(ROOT / "seeds" / "mt_en_de.tsv")
    r = evaluate_mt(rows, n_resamples=10, use_comet=True)
    assert r["systems"]["sys_a"]["comet"] > r["systems"]["sys_b"]["comet"]
