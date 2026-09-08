# Archived runs

Runs kept for the record. Nothing here scores a current benchmark split, so the
freshness gate (`scripts/check_results_freshness.py --results-dir results`) and
the `hallmark leaderboard` glob both skip this directory. The canonical released
results are in `data/v1.2/baseline_results/`.

A file belongs here when it is a real artifact that no staleness check can judge:
a CI sample, a smoke run, a probe report, or a run superseded by a later one. A
file that *should* score the current split and does not belongs in the gate,
where its staleness is the finding.

| File | Why it is here |
|---|---|
| `temporal_probe_*.json` | Temporal-robustness probe reports over a 60-entry post-cutoff probe set, one per model. They carry `probe_metrics` / `full_baseline` rather than the `EvaluationResult` fields, name no split, and are the input to `fig_temporal_robustness` in `scripts/generate_figures.py`. |

`scripts/probe_temporal_robustness.py` writes new probe reports here.

Three sibling directories stay outside this archive, and outside the gate's
non-recursive glob: `results/superseded_pre_relabel/` (results replaced by a
re-run under the current labels, with its own README), `results/failed_runs/`
(runs quarantined because the wrapper fell back to all-VALID), and
`results/checkpoints/` (resumable run state, not results).
