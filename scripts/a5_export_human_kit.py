#!/usr/bin/env python3
"""A5: export a ready-to-use HUMAN inter-annotator-agreement kit.

The automated multi-rater kappa is a reliability *proxy*. This kit lets the
user run the real thing with human annotators: it bundles the blinded entries,
an annotation rubric, a per-annotator CSV template, and instructions. Drop the
filled CSVs back next to this kit and feed them to ``a5_score_human_iaa.py``
(or reuse the kappa functions in ``a5_compute_kappa.py``).

Outputs (under ``results/ablations/a5_kappa/human_annotation_kit/``):
  * ``annotation_entries.csv``  -- blinded entries, one row per citation
  * ``annotation_template.csv`` -- empty template for one annotator to fill
  * ``RUBRIC.md``               -- decision rubric + hallucination-type catalog
  * ``INSTRUCTIONS.md``         -- how to run the study + how to score it
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "ablations" / "a5_kappa"
KIT = OUT / "human_annotation_kit"

HALLUCINATION_TYPES = [
    ("fabricated_doi", "DOI does not resolve / is invented."),
    ("nonexistent_venue", "Venue or journal does not exist."),
    ("placeholder_authors", "Authors are placeholders (e.g. 'Author1', bare 'et al.')."),
    ("future_date", "Year is in the future relative to plausible publication."),
    ("chimeric_title", "Title splices fragments from multiple real works."),
    ("wrong_venue", "Real paper cited at the wrong venue."),
    ("swapped_authors", "Authors swapped or mismatched against the real paper."),
    ("preprint_as_published", "arXiv preprint cited as published in a venue."),
    ("hybrid_fabrication", "Real DOI but authors/title don't match the DOI target."),
    ("near_miss_title", "Title differs from a real paper by small but meaningful edits."),
    ("plausible_fabrication", "Entirely fabricated yet plausible-sounding paper."),
    ("merged_citation", "Metadata combined from two real papers."),
    ("partial_author_list", "Real paper but author list is incomplete."),
    ("arxiv_version_mismatch", "arXiv version cited as a different version / as published."),
]


def _to_bibtex(entry: dict) -> str:
    if entry.get("raw_bibtex"):
        return str(entry["raw_bibtex"])
    lines = [f"@{entry['bibtex_type']}{{{entry['bibtex_key']},"]
    for key, value in sorted(entry["fields"].items()):
        lines.append(f"  {key} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    KIT.mkdir(parents=True, exist_ok=True)
    blinded = [
        json.loads(line)
        for line in (OUT / "substrate_blinded.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # 1. annotation_entries.csv -- human-readable view of each blinded entry.
    fields_seen = ["title", "author", "year", "booktitle", "journal", "venue", "doi", "url"]
    with (KIT / "annotation_entries.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bibtex_key", "bibtex_type", *fields_seen, "other_fields", "raw_bibtex"])
        for e in blinded:
            f = e["fields"]
            other = {k: v for k, v in f.items() if k not in fields_seen}
            w.writerow(
                [
                    e["bibtex_key"],
                    e["bibtex_type"],
                    *[f.get(k, "") for k in fields_seen],
                    json.dumps(other, ensure_ascii=False) if other else "",
                    _to_bibtex(e),
                ]
            )

    # 2. annotation_template.csv -- one row per entry for a single annotator.
    with (KIT / "annotation_template.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "bibtex_key",
                "title",
                "label",  # VALID | HALLUCINATED | UNCERTAIN
                "hallucination_type",  # one of the catalog (blank if VALID/UNCERTAIN)
                "confidence",  # 0.0-1.0
                "notes",
            ]
        )
        for e in blinded:
            w.writerow([e["bibtex_key"], e["fields"].get("title", ""), "", "", "", ""])

    # 3. RUBRIC.md
    type_lines = "\n".join(f"| `{t}` | {d} |" for t, d in HALLUCINATION_TYPES)
    rubric = f"""# HALLMARK A5 -- Human Annotation Rubric

You are an independent annotator. For each citation in `annotation_entries.csv`,
decide whether it is a **VALID** real publication or a **HALLUCINATED**
(fabricated or corrupted) reference, using **only the metadata shown**. Do not
consult the original dataset labels. You may use external search engines and
databases (Google Scholar, DBLP, Semantic Scholar, CrossRef) -- in fact you
should, exactly as a careful reviewer would when checking a bibliography.

## Decision

- **VALID** -- the paper exists and the metadata (title, authors, year, venue,
  DOI) is consistent with the real record.
- **HALLUCINATED** -- the paper does not exist, OR it exists but the metadata is
  corrupted (wrong authors, wrong venue, spliced title, mismatched DOI, etc.).
- **UNCERTAIN** -- you cannot resolve the entry with reasonable effort. Use
  sparingly; prefer a decision when the evidence supports one.

Record a **confidence** in `[0, 1]` and a short **note** with your reasoning
(e.g. "found on DBLP, authors match" or "title is a splice of two CVPR papers").

## Hallucination-type catalog (fill only when label = HALLUCINATED)

| type | definition |
|------|------------|
{type_lines}

## Filling the template

Copy `annotation_template.csv` to `annotator_<yourname>.csv` and fill one row
per entry. Keep the `bibtex_key` column untouched so rows can be matched across
annotators.
"""
    (KIT / "RUBRIC.md").write_text(rubric)

    # 4. INSTRUCTIONS.md
    n = len(blinded)
    instructions = f"""# HALLMARK A5 -- Human Inter-Annotator-Agreement Study

This kit lets you run a true human inter-annotator-agreement (IAA) study on the
{n} most contestable HALLMARK entries: the real-world hallucination incidents
plus the real papers recovered by the systematic relabel pass. The automated
multi-rater kappa in `../kappa_results.json` is a **reliability proxy**, not
human IAA -- this kit produces the human version.

## Substrate

- `annotation_entries.csv` -- {n} blinded entries (metadata only; no gold
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
"""
    (KIT / "INSTRUCTIONS.md").write_text(instructions)

    print(f"Human annotation kit written -> {KIT}")
    for p in sorted(KIT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
