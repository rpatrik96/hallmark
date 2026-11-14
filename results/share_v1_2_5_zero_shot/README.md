# Zero-shot LLM per-entry predictions on v1.2.5

One file per model and split (`<tool>_<split>_predictions.jsonl`), for the 12 zero-shot
LLM baselines on `dev_public` (1,119) and `test_public` (825).

- Already on v1.2.5 membership: the six year-only test_public entries
  (`results/relabel_recheck/v1_2_5_exclusion_log.json`) are dropped.
- Predictions carry no gold label. Score against `data/v1.2/{dev,test}_public.jsonl` on this
  branch, which has the seven v1.2.5 relabels (`results/relabel_recheck/v1_2_5_label_log.json`).
- Rescoring each file reproduces the released v1.2.5 DR / FPR / F1 / ECE exactly
  (`matches_released_v1_2_5` in `manifest.json`, which also records source path, eval mode and counts).
- `[Error fallback]` rows are kept, as in the released numbers: deepseek_r1 test 177 / dev 18,
  gemini_pro test 77 / dev 37, gemini_flash dev 13, mistral dev 13, others 1-2.
- Zero-shot models do not resolve DOIs, so the v1.2.5 arXiv DOI-suffix fix does not affect them.
