from __future__ import annotations

import copy

from hallmark.dataset.schema import BenchmarkEntry


def _clone_entry(entry: BenchmarkEntry) -> BenchmarkEntry:
    """Deep clone a BenchmarkEntry.

    Strips the ``url`` field from cloned entries to prevent a trivial
    shortcut: hallucinated entries that inherit the source paper's DBLP
    URL allow tools to detect mismatches by simply resolving the URL
    and comparing returned metadata.
    """
    fields = copy.deepcopy(entry.fields)
    # Remove URL to prevent metadata-comparison shortcut
    fields.pop("url", None)
    return BenchmarkEntry(
        bibtex_key=entry.bibtex_key,
        bibtex_type=entry.bibtex_type,
        fields=fields,
        label=entry.label,
        hallucination_type=entry.hallucination_type,
        difficulty_tier=entry.difficulty_tier,
        explanation=entry.explanation,
        generation_method=entry.generation_method,
        source_conference=entry.source_conference,
        publication_date=entry.publication_date,
        added_to_benchmark=entry.added_to_benchmark,
        subtests=copy.deepcopy(entry.subtests),
        raw_bibtex=None,
        source=entry.source,
    )


def is_preprint_source(entry: BenchmarkEntry) -> bool:
    """Check if an entry looks like an arXiv preprint (suitable for preprint_as_published)."""
    has_eprint = bool(entry.fields.get("eprint"))
    has_conference_doi = bool(entry.fields.get("doi"))
    # A preprint either has an eprint field, or lacks a conference DOI
    # Entries with both DOI and booktitle are conference papers, not preprints
    if has_eprint:
        return True
    return not has_conference_doi and entry.bibtex_type in ("misc", "article")
