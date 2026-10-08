# llm-eval-lab

A small lab for doing evaluation properly: versioned test sets with DVC,
precision/recall/F1 with bootstrap confidence intervals, paired significance
tests for A/B comparisons, a human-review queue with inter-rater agreement,
MT metrics (BLEU, chrF++, optional COMET), auto-generated model and system
cards, and a CI gate that fails the build when a candidate regresses against a
stored baseline.

> **Learning project.** I built this to practise evaluation methodology end to
> end. It is not used in production anywhere. The "models" are two small
> deterministic classifiers and the data is synthetic, so the statistics are
> real but the systems are toys. See [What is real vs stubbed](#what-is-real-vs-stubbed).

## What it does

```
seeds/ + params.yaml ──dvc repro──► data/testsets/ (intent_v2.jsonl, MANIFEST.json; hashes pinned in dvc.lock)
                                         │
                       llm-eval predict  ▼
                       runs/rules-v1.jsonl, runs/nb-v1.jsonl
                                         │
                      llm-eval evaluate  ▼
   reports/eval.json: P/R/F1 + 95% bootstrap CIs, McNemar, permutation test, BLEU/chrF++ (+ paired bootstrap)
          │                     │                         │
   llm-eval gate          llm-eval card            llm-eval review enqueue/next/label/adjudicate
   vs baselines/          model_card.md            SQLite queue → Cohen's kappa → corrections
   baseline.json          system_card.md                  → seeds/corrections.jsonl → test set v3
   (exit 1 on regression)
```

| Capability | How |
|---|---|
| Versioned test sets | `dvc.yaml` stages `testsets → predict → evaluate`; `dvc.lock` pins the md5 of every output. The generator is seeded, so CI re-runs it and checks the hash is unchanged. `params.yaml` selects the version (`v1`: 240 easy tickets; `v2`: + 80 hard tickets + adjudicated corrections). A local DVC remote (`/tmp/llm-eval-lab-dvc-remote`) shows push/pull. |
| P/R/F1 with CIs | Per-class and macro precision/recall/F1, accuracy and per-slice accuracy, each with a percentile bootstrap 95% CI (1,000 resamples, seeded). numpy only. |
| Significance tests | McNemar (exact binomial and chi-squared with continuity correction) on paired correctness; paired approximate-randomisation (permutation) test on any corpus metric, with a paired-bootstrap CI for the delta. |
| Human review | SQLite queue fed by system disagreement, low confidence and a seeded random audit sample. Reviewers don't see gold or model output; each item gets 2 independent labels; Cohen's kappa (with Landis & Koch bands) per rater pair; disagreements go to adjudication; adjudicated labels that contradict gold become test-set corrections. |
| MT metrics | BLEU and chrF++ (`CHRF(word_order=2)`) from sacrebleu with their signatures recorded; paired bootstrap resampling (Koehn 2004) for system comparisons; COMET (`Unbabel/wmt22-comet-da`) behind the optional `[comet]` extra. |
| Model / system cards | `llm-eval card` renders Markdown cards (Mitchell et al. 2019 sections) from `reports/eval.json`, including CIs, the paired comparison and reviewer agreement. |
| CI gate | `gate.yaml`: max absolute drop per metric, hard floors, "not significantly worse" (McNemar vs the baseline's stored per-example predictions) and MT drops. Refuses to compare against a baseline built on a different test-set hash unless explicitly allowed. Writes a Markdown table to `$GITHUB_STEP_SUMMARY`. |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,dvc]"     # drop ,dvc if you only want the library
pytest -q                       # 52 passed, 1 skipped (COMET) with DVC installed
python demo/run_demo.py         # offline end-to-end demo (output below)

dvc repro                       # rebuild test sets -> predictions -> reports
dvc metrics show
llm-eval gate                   # exit 0 = no regression vs baselines/baseline.json
llm-eval gate --candidate nb-small   # a deliberately weaker candidate -> exit 1
llm-eval card                   # reports/model_card.md, reports/system_card.md
```

Test-set versions live in git history: `dvc.lock` pins the hash, so
`git checkout <rev> && dvc checkout` (or `dvc pull` from the remote) restores
the exact files any past result was computed on.

Reviewing by hand:

```bash
llm-eval review enqueue                       # from the two systems' disagreements, low confidence, 5% audit
llm-eval review next --rater alice            # {"id": 3, "example_id": "t0123", "text": "..."}
llm-eval review label --item 3 --rater alice --label billing
llm-eval review pending                       # items where raters disagree
llm-eval review adjudicate --item 3 --rater lead --label billing --note "refund request"
llm-eval review stats                         # kappa per rater pair, accuracy vs gold
llm-eval review corrections >> seeds/corrections.jsonl   # feeds the next test-set version
```

To run the same harness against a real model, add `openai:<model>` to
`systems` in `params.yaml` and set `OPENAI_BASE_URL` / `OPENAI_API_KEY`
(any OpenAI-compatible server). That path is not exercised in CI.

## Demo run

`python demo/run_demo.py` (about 18 s on the 8-core build machine):

```text
== 1. versioned test sets (same seed -> same bytes; v2 adds 80 hard tickets)
   intent_v1.jsonl n=240 sha256=86f3d8f4bc4fe9fa labels={'account': 60, 'billing': 60, 'bug': 60, 'shipping': 60}
   intent_v2.jsonl n=320 sha256=e9723cd867350951 labels={'account': 80, 'billing': 80, 'bug': 80, 'shipping': 80}

== 2. per-system metrics with 95% bootstrap CIs (1000 resamples)
   nb-v1     accuracy 0.963 [0.938, 0.981]  macro-F1 0.962 [0.937, 0.981]  by slice {'easy': 1.0, 'hard': 0.85}
   rules-v1  accuracy 0.791 [0.744, 0.834]  macro-F1 0.795 [0.750, 0.835]  by slice {'easy': 0.9708, 'hard': 0.25}

== 3. paired A/B tests on the same 320 tickets
   McNemar: rules-v1 right & nb-v1 wrong = 6, reverse = 61, exact p = 1.49e-12
   permutation accuracy: delta +0.172 95% CI [+0.128, +0.216] p = 0.0010
   permutation macro_f1: delta +0.168 95% CI [+0.125, +0.214] p = 0.0010

== 4. MT (EN->DE, 30 hand-written sentences): BLEU and chrF++ via sacrebleu
   sys_a: BLEU 84.31  chrF++ 90.81
   sys_b: BLEU 26.78  chrF++ 57.20
   paired bootstrap bleu: sys_b - sys_a = -57.53 [-71.36, -43.51]
   paired bootstrap chrf++: sys_b - sys_a = -33.61 [-41.46, -25.62]
   signature: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0

== 5. human-review queue (two SIMULATED raters: alice 92%, bob 85% accurate)
   enqueued 88 items
   reasons {'audit': 20, 'disagreement': 71, 'low_confidence': 6}
   Cohen's kappa alice vs bob = 0.686 (substantial, n=88)
   rater accuracy vs gold {'alice': 0.9205, 'bob': 0.8523}, needing adjudication: 20
   adjudicated 20; corrections to gold: 0

== 6. CI gate vs stored baseline (baselines/baseline.json = nb-v1)
   candidate nb-v1  -> exit 0 (Eval gate: PASSED)
   candidate nb-small -> exit 1 (Eval gate: FAILED)
     | macro_f1 drop | 0.9624 | 0.7042 | >= baseline - 0.02 | **FAIL** |
     | accuracy drop | 0.9625 | 0.7281 | >= baseline - 0.02 | **FAIL** |
     | macro_f1 floor |  | 0.7042 | >= 0.8 | **FAIL** |
     | McNemar vs baseline | b=80 | c=5 | not significantly worse (p=1.81e-18, alpha=0.05) | **FAIL** |

== 7. model and system cards
   reports/model_card.md: 47 lines; reports/system_card.md written
   | Accuracy | 0.963 [0.938, 0.981] |
   | Macro F1 | 0.962 [0.937, 0.981] |
   - McNemar: rules-v1 right / nb-v1 wrong = 6, the reverse = 61, exact p = 1.494e-12.
   - Cohen's kappa alice vs bob: 0.686 (substantial, n=88).
```

Notes on reading it:
- The permutation p-values of 0.0010 are the floor for 1,000 permutations
  (`(0 + 1) / (1000 + 1)`): no permutation reached the observed delta.
- The rules system is fine on easy tickets (97%) and collapses on the hard
  slice (25%); the macro numbers alone hide that, which is why the report keeps
  per-slice accuracy.
- The review simulation finds no corrections because the simulated raters are
  scored against the generator's own gold. With real reviewers this is where
  label errors in the test set would surface.

## Tests

`pytest -q` gives **52 passed, 1 skipped** in about 25 seconds with the `dvc`
extra installed (the 2 DVC tests skip without it; the COMET test skips unless
`[comet]` is installed).

- `test_metrics.py`: hand-computed per-class P/R/F1, absent classes score 0 (not NaN), CIs bracket the point estimate, shrink with more data and are seeded.
- `test_stats.py`: McNemar against hand-computed exact and chi-squared p-values, permutation test on identical / clearly different / barely different systems, paired bootstrap direction.
- `test_agreement.py`: the textbook kappa example (p_o 0.7, p_e 0.5, κ 0.4), perfect and systematic disagreement, CI.
- `test_review.py`: enqueue reasons and idempotence, blind labelling, two-label cap, adjudication → corrections, kappa from simulated raters, persistence.
- `test_mt.py`: identity = 100, chrF++ vs BLEU on a partial match, fluent system beats literal one with a CI below zero, explicit error when COMET is missing.
- `test_datasets.py`, `test_systems.py`: determinism, v1 ⊂ v2, corrections only in v2, train/test disjoint.
- `test_pipeline.py`: CLI end to end, the committed baseline matches a fresh run, the gate fails for `nb-small`, blocks a silent test-set change, writes the GitHub summary.
- `test_dvc.py`: real `dvc repro`, no-op second repro, push to a local remote, delete, pull, same hash; and the committed `dvc.lock` hash equals what the generator produces today.

CI (`.github/workflows/ci.yml`) runs the tests and demo on Python 3.11 and
3.12, then a second job that runs `dvc repro` from scratch, posts
`dvc metrics diff` and the gate table to the job summary, fails on a
regression, and uploads the reports and cards as an artifact.

## What is real vs stubbed

**Real (runs and is tested here):**
- All statistics: bootstrap CIs, McNemar, permutation tests, paired bootstrap, Cohen's kappa. Checked against hand-computed values.
- BLEU and chrF++ from sacrebleu 2.x, with signatures recorded.
- DVC: the pipeline, lock file, metrics and a push/pull round trip to a local remote, run in tests and CI.
- The SQLite review queue, the gate (including the failing case) and the card generator.

**Stubbed, synthetic or not verified:**
- **The systems are toys.** `rules-v1` is keyword matching and `nb-v1` is a hand-written naive Bayes. Neither is an LLM. The `openai:` adapter is unit-tested for parsing only, never against a live API.
- **The intent data is synthetic,** generated from templates. Template-generated examples are not independent, so the CIs are optimistic.
- **The MT set is 30 hand-written EN→DE sentences.** The two "systems" are hand-written variants (fluent vs literal), not MT output. Real MT evaluation needs far more data.
- **COMET is not run in CI.** I ran it once locally on CPU (`unbabel-comet` 2.2.7, `Unbabel/wmt22-comet-da`, about 55 s including model load): sys_a **0.972**, sys_b **0.910**, the same ranking as BLEU and chrF++. `unbabel-comet` 2.2.7 also needed `setuptools<80` installed for an old `pkg_resources` import. The `comet` pytest marker covers it when the extra is installed.
- **Reviewers are simulated** in the demo and tests (seeded random accuracy). There is no web UI; reviewing is through the CLI.
- **The DVC remote is a local directory.** An S3/GCS remote is a one-line `dvc remote add` change that I haven't tested.
- **The MT gate check is trivial today,** because the MT hypotheses are fixed files. It guards against scorer or tokenizer changes, not model changes.

## Layout

```
src/llm_eval_lab/  datasets.py · systems.py · metrics.py · stats.py · agreement.py · review.py
                   mt.py · runner.py · gate.py · cards.py · cli.py
seeds/             MT sentences, review corrections (inputs to the testsets stage)
baselines/         baseline.json (metrics + per-example predictions for the paired gate test)
dvc.yaml, dvc.lock, params.yaml, gate.yaml
reports/metrics.json   committed so `dvc metrics diff` has something to compare against
tests/             52 tests (+1 COMET, optional)
demo/run_demo.py
```

## License

MIT
