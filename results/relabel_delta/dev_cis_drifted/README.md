# Dev_public bootstrap CIs — Sonnet 4.6 / Opus 4.7 (DRIFTED re-run, do NOT ship)

These augmented result JSONs carry stratified bootstrap 95% CIs (seed=42, 10k resamples)
computed from the **full 1119-entry per-entry predictions** regenerated on 2026-05-30
(TASK B), via `scripts/compute_bootstrap_ci.py` → `compute_persisted_cis()`.

**Why these are not shipped to the paper.** The 2026-05-30 re-run does NOT reproduce the
paper's published Sonnet/Opus dev_public aggregates. The OpenRouter Anthropic endpoint
(`anthropic/claude-{sonnet-4.6,opus-4.7}`) drifted since the original 2026-05-04 run
(commit `493cbb3`), which used the identical path / prompt / temp=0 but whose per-entry
predictions were never persisted. Evidence:

* Controlled replay of the committed temporal-supplement entries (identical inputs, identical
  path) shows only **90.0 %** (Opus) / **75.0 %** (Sonnet) label agreement vs the 2026-05-04
  predictions — see `../endpoint_drift_probe/drift_summary.json`. Sonnet now emits UNCERTAIN
  labels the original never produced.
* On dev_public the drift compounds: re-run DR/FPR = **.916/.165** (Sonnet, published .781/.127),
  **.909/.162** (Opus, published .752/.072) — 13.5 pp / 15.7 pp max deviation, far beyond the
  "few pp" delta-eval sanity bar.

The CIs here therefore surround the **drifted** point estimates and are inconsistent with the
paper's published values (which the delta-eval and all of manifest §11 — PPV table, Pareto,
takeaways — are built on). They are persisted for reproducibility and as drift evidence, not
as a replacement for the published numbers.

**Net effect on the TODOs.** The app:bootstrap dev-CI TODO and the tab:pertype_full F5 dev-column
TODO for Sonnet/Opus cannot be resolved with a faithful number, because the originals are
unpersisted and the endpoint no longer reproduces them. They are resolved by a **documented
limitation** instead (see `../todo_sonnet_opus_dev.json` → `meta.finding`, and the appendix prose
that already states these two tools are summary-only on dev_public).

Source artifacts:
* per-entry predictions: `results/llm_openrouter_claude_{sonnet_4_6,opus_4_7}_dev_public_predictions.jsonl`
* scorer: `results/relabel_delta/score_sonnet_opus_dev.py`
* drift probe: `results/relabel_delta/endpoint_drift_probe/`
