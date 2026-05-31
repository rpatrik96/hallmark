# HALLMARK A5 -- Human Inter-Annotator-Agreement Study

This kit lets you run a true human inter-annotator-agreement (IAA) study on the
132 most contestable HALLMARK entries: the real-world hallucination incidents
plus the real papers recovered by the systematic relabel pass. The automated
multi-rater kappa in `../kappa_results.json` is a **reliability proxy**, not
human IAA -- this kit produces the human version.

## Substrate

- `annotation_entries.csv` -- 132 blinded entries (metadata only; no gold
  labels, no provenance).
- Gold labels live in `../substrate_gold.jsonl` and must be kept hidden from
  annotators until scoring.

## Protocol (recommended)

1. Recruit >= 3 annotators familiar with ML literature.
2. Give each annotator `RUBRIC.md` and a fresh copy of `annotation_template.csv`
   renamed `annotator_<name>.csv`. Do NOT share `substrate_gold.jsonl`.
3. Annotators label independently (no discussion) -- this is critical for a
   valid IAA estimate.
4. Collect the filled CSVs back into this directory.

## Scoring

Reuse the kappa functions in `scripts/a5_compute_kappa.py`:

- pairwise **Cohen's kappa** between each pair of annotators,
- **Fleiss' kappa** across all annotators,
- each annotator vs the benchmark gold label (`substrate_gold.jsonl`),
- a **hallucination-type agreement** rate on entries all annotators call
  HALLUCINATED.

Interpret with the Landis & Koch (1977) scale: <=0.20 slight, <=0.40 fair,
<=0.60 moderate, <=0.80 substantial, >0.80 almost perfect.

## Honest framing for the paper

Report human IAA as the primary reliability evidence and the automated
multi-rater kappa as a scalable proxy that anticipated it. State plainly that
LLM raters share systematic blind spots (e.g. recency, popularity bias), so
their agreement upper-bounds neither human agreement nor ground-truth quality.
