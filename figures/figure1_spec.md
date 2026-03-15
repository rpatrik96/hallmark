# Figure 1: HALLMARK Overview — Specification

Three-panel horizontal layout (`\subfigure` or `minipage`, full `\textwidth`).

## Panel (a): "The Problem"

**Goal:** Show a concrete side-by-side comparison of a VALID and a HALLUCINATED BibTeX entry, with sub-test annotations highlighting what changed.

**Layout:**
- Two BibTeX entries side by side (monospace font, light background boxes)
- Left: VALID entry with green check marks on each field
- Right: HALLUCINATED entry (e.g., `chimeric_title` type) with:
  - Red X on the mutated field(s) (e.g., title)
  - Green checks on unchanged fields
  - A callout or bracket labeling the hallucination type

**Design notes:**
- Use `\texttt{}` or `listings` for BibTeX formatting
- Color-code: green (#2CA02C) for valid fields, red (#D62728) for hallucinated fields
- Keep entries short (4-5 fields: author, title, booktitle, year, doi)
- Use a real example from the dataset if possible

**Suggested example pair:**
- VALID: A real NeurIPS 2022 paper entry
- HALLUCINATED: Same entry with the title replaced by a chimeric title (spliced from two real papers)

## Panel (b): "The Taxonomy"

**Goal:** Visualize the 3-tier hallucination taxonomy with type counts and a difficulty gradient.

**Layout:**
- Three stacked horizontal blocks, one per tier
- Each block contains the type names in that tier
- Vertical arrow on the right side labeled "Difficulty"

**Tier contents:**
- **Tier 1 — Easy** (green block, 4 types): `fabricated_doi`, `nonexistent_venue`, `placeholder_authors`, `future_date`
- **Tier 2 — Medium** (amber block, 5 types): `chimeric_title`, `wrong_venue`, `author_mismatch`, `preprint_as_published`, `hybrid_fabrication`
- **Tier 3 — Hard** (red block, 2+1 types): `near_miss_title`, `plausible_fabrication`; stress-test: `arxiv_version_mismatch`

**Design notes:**
- Block colors: Tier 1 = muted green (#A8D5A2), Tier 2 = muted amber (#F5D98E), Tier 3 = muted red (#F5A8A8)
- Each type name in small sans-serif inside its block
- Include count badges (e.g., "4 types") on the right edge of each block
- Difficulty arrow: gradient from green to red, labeled "Easy" at top and "Hard" at bottom
- TikZ `\draw` with `\node` for each block is the natural implementation

## Panel (c): "The Gap"

**Goal:** Show the DR vs FPR scatter plot (the figure from `generate_dr_fpr_scatter.py`).

**Layout:**
- Directly embed the PDF from `paper/figures/dr_fpr_scatter.pdf`
- No modifications needed; the scatter plot is self-contained

**Caption note for panel (c):**
- "LLMs achieve high detection rates but at the cost of elevated false positive rates. API tools cluster near the origin with low coverage."

## Overall Figure Properties

- **Width:** Full `\textwidth` (~6.5 inches for NeurIPS single-column, ~7 inches for camera-ready)
- **Panel widths:** (a) 30%, (b) 30%, (c) 40% — adjust to balance visual weight
- **Vertical alignment:** All panels top-aligned
- **Caption:** "Overview of HALLMARK. (a) A valid BibTeX entry and its hallucinated counterpart with sub-test annotations. (b) The three-tier hallucination taxonomy with 11 main types ordered by detection difficulty. (c) Detection Rate vs. False Positive Rate for all evaluated tools; LLMs (circles) trade precision for recall while API tools (squares) have limited coverage."
- **Label:** `\label{fig:overview}`
- **Referenced as:** `\cref{fig:overview}`

## Implementation Plan

1. Panels (a) and (b) should be implemented in TikZ for precise control
2. Panel (c) is a `\includegraphics` of the pre-generated scatter plot PDF
3. Use `\subcaption` package for sub-figure labels
4. Ensure all text is rendered at readable size (minimum 7pt after scaling)
5. Test at both single-column and double-column widths
