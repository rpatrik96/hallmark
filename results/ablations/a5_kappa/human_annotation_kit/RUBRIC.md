# HALLMARK A5 -- Human Annotation Rubric

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
| `fabricated_doi` | DOI does not resolve / is invented. |
| `nonexistent_venue` | Venue or journal does not exist. |
| `placeholder_authors` | Authors are placeholders (e.g. 'Author1', bare 'et al.'). |
| `future_date` | Year is in the future relative to plausible publication. |
| `chimeric_title` | Title splices fragments from multiple real works. |
| `wrong_venue` | Real paper cited at the wrong venue. |
| `swapped_authors` | Authors swapped or mismatched against the real paper. |
| `preprint_as_published` | arXiv preprint cited as published in a venue. |
| `hybrid_fabrication` | Real DOI but authors/title don't match the DOI target. |
| `near_miss_title` | Title differs from a real paper by small but meaningful edits. |
| `plausible_fabrication` | Entirely fabricated yet plausible-sounding paper. |
| `merged_citation` | Metadata combined from two real papers. |
| `partial_author_list` | Real paper but author list is incomplete. |
| `arxiv_version_mismatch` | arXiv version cited as a different version / as published. |

## Filling the template

Copy `annotation_template.csv` to `annotator_<yourname>.csv` and fill one row
per entry. Keep the `bibtex_key` column untouched so rows can be matched across
annotators.
