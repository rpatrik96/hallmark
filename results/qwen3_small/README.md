# Qwen3 dense sweep: a parameter-count comparison that does not separate

Eight scored results, 4B/8B/14B/32B on `dev_public` and `test_public`, served
through the HuggingFace/Featherless router (`--provider huggingface`). The runs
are from 2026-08-11; this directory is the scoring of them.

CLAUDE.md describes the dense series as "a controlled parameter-count sweep
within one family." It does not currently support that reading.

| model | split | n | DR | FPR | F1 | ECE |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B  | dev  | 1,119 | .975 | **.940** | .704 | .395 |
| Qwen3-8B  | dev  | 1,119 | .977 | **.949** | .703 | .389 |
| Qwen3-14B | dev  | 1,119 | .990 | **.979** | .703 | .377 |
| Qwen3-32B | dev  | 1,119 | .975 | **.949** | .702 | .392 |
| Qwen3-4B  | test |   831 | .977 | **.913** | .773 | .309 |
| Qwen3-8B  | test |   831 | .983 | **.929** | .773 | .301 |
| Qwen3-14B | test |   831 | .994 | **.965** | .772 | .289 |
| Qwen3-32B | test |   831 | .988 | **.936** | .775 | .302 |

Every model flags 95-98% of entries as hallucinated. The true rate on
`dev_public` is 54% (606 of 1,119), so an FPR of .94-.98 means nearly every
valid entry is flagged too. The high detection rates are an artifact of
answering HALLUCINATED almost everywhere and carry no signal; F1 sits at .70
(dev) and .77 (test) for all four sizes, which is close to what a
flag-everything constant predictor scores on these splits.

There is no monotonic trend in size. 14B is the *worst* on both splits
(FPR .979 / .965) and 4B the least bad (.940 / .913); 8B and 32B are tied on
dev. The ordering is not a parameter-count effect.

The records are not malformed. A sampled 4B response carries a coherent
`reason`, a `predicted_hallucination_type` of `plausible_fabrication`, one API
call and 3.6 s wall clock -- so this is not the truncated-JSON failure the
`_NO_THINK_MODELS` guard addresses (CLAUDE.md, *Thinking-mode guard*). The
models answer; they answer HALLUCINATED. The open question is whether the
prompt pushes them there or whether models this size cannot confirm a real
paper from parametric knowledge and treat "cannot confirm" as fabricated.
Distinguishing those needs a control -- the same prompt against entries the
model is known to have memorized -- not another sweep.

ECE of .29-.40 says the confidences are badly calibrated on top of it.

These are reported as a negative result. They are not release artifacts and are
deliberately not in `data/v1.2/baseline_results/`, which `hallmark
validate-results` treats as the released set.

## Reproducing

Raw per-entry predictions live in `results/checkpoints/llm_hf_qwen3_*/`, which
is gitignored (`.gitignore:117`) -- that is why these runs were invisible to git
for a month. Scoring makes no model calls:

```
uv run hallmark evaluate --split dev_public \
    --predictions results/checkpoints/llm_hf_qwen3_4b_dev_public/huggingface_qwen3-4b.jsonl \
    --tool-name llm_hf_qwen3_4b \
    --output results/qwen3_small/llm_hf_qwen3_4b_dev_public.json --strict
```

The 32B checkpoints carry retry duplicates (1 on dev, 2 on test): a first
attempt returning UNCERTAIN at confidence .5 and a retry returning HALLUCINATED
at .95. The retry is authoritative, so the files above were scored with
last-write-wins applied. `load_predictions` does not deduplicate, so scoring the
32B checkpoints directly double-counts those entries.
