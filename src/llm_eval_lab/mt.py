"""MT metrics: BLEU and chrF++ via sacrebleu; COMET via unbabel-comet if installed."""
from __future__ import annotations

from sacrebleu.metrics import BLEU, CHRF

from .stats import paired_bootstrap_corpus

_BLEU = BLEU()
_CHRFPP = CHRF(word_order=2)  # word_order=2 is chrF++


def bleu(hyps: list[str], refs: list[str]) -> float:
    return _BLEU.corpus_score(hyps, [refs]).score


def chrf_pp(hyps: list[str], refs: list[str]) -> float:
    return _CHRFPP.corpus_score(hyps, [refs]).score


def signatures() -> dict:
    return {"bleu": str(_BLEU.get_signature()), "chrf++": str(_CHRFPP.get_signature())}


class CometUnavailable(RuntimeError):
    pass


def comet(srcs: list[str], hyps: list[str], refs: list[str], model: str = "Unbabel/wmt22-comet-da", gpus: int = 0) -> float:
    """System-level COMET. Needs ``pip install .[comet]`` and downloads the model (~2.3 GB) on first use."""
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise CometUnavailable("unbabel-comet is not installed; pip install '.[comet]'") from exc
    ckpt = load_from_checkpoint(download_model(model))  # pragma: no cover
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]  # pragma: no cover
    return float(ckpt.predict(data, batch_size=16, gpus=gpus).system_score)  # pragma: no cover


def evaluate_mt(rows: list[dict], systems=("sys_a", "sys_b"), n_resamples: int = 1000, seed: int = 0, use_comet: bool = False) -> dict:
    refs = [r["ref"] for r in rows]
    srcs = [r["src"] for r in rows]
    out: dict = {"n": len(rows), "systems": {}}
    for s in systems:
        hyps = [r[s] for r in rows]
        entry = {"bleu": round(bleu(hyps, refs), 2), "chrf++": round(chrf_pp(hyps, refs), 2)}
        if use_comet:
            try:
                entry["comet"] = round(comet(srcs, hyps, refs), 4)
            except CometUnavailable as exc:
                entry["comet"] = None
                out["comet_note"] = str(exc)
        out["systems"][s] = entry
    if len(systems) == 2:
        a, b = ([r[s] for r in rows] for s in systems)
        out["paired"] = {
            "bleu": paired_bootstrap_corpus(bleu, a, b, refs, n_resamples, seed).to_dict(),
            "chrf++": paired_bootstrap_corpus(chrf_pp, a, b, refs, n_resamples, seed).to_dict(),
        }
    out["signatures"] = signatures()  # sacrebleu only knows the reference count after scoring
    return out
