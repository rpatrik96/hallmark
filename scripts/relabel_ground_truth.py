#!/usr/bin/env python3
"""Systematic, deterministic, multi-source ground-truth relabeller for HALLMARK.

SUPERSEDES ``scripts/patch_mislabels.py``
=========================================
``patch_mislabels.py`` carried a hand-curated 32-key allow-list of confirmed
mislabels (four "batches") that were verified one at a time. That approach was
*opportunistic*: it only fixed entries someone happened to notice, it could not
recompute sub-tests from the real paper, and it left every flipped VALID entry
carrying stale ``hallucination_type`` / ``difficulty_tier`` / False sub-tests.

This script replaces it with ONE deterministic pass that re-verifies EVERY
``HALLUCINATED`` entry against multiple bibliographic sources and decides each
label from source evidence alone (no LLM, no allow-list). ``patch_mislabels.py``
has now been removed from the tree; this module is the sole relabeller.

ROOT CAUSE this fixes
---------------------
The original auto-labeller resolved DOIs through CrossRef only. CrossRef does
*not* index arXiv DataCite DOIs (``10.48550/arXiv.*``) and rarely indexes the
NeurIPS/ICLR/ICML proceedings, so real arXiv papers were declared "DOI does not
resolve -> no CrossRef title match -> HALLUCINATED/plausible_fabrication" with
``title_exists=false`` / ``authors_match=false``. The LLM "fabrication" generator
(``generate_llm_hallucinations.py`` hardcodes ``label=HALLUCINATED``) also
emitted some genuinely real papers. The combined effect: dozens of real landmark
papers (OPT, PaLM, Flamingo, Make-A-Video, Linformer, SCAFFOLD, FedProx, ...)
remain labeled HALLUCINATED.

Resolution strategy (deterministic, source-based)
--------------------------------------------------
For each HALLUCINATED entry:
  1. If ``fields.doi`` is an arXiv DataCite DOI (``10.48550/arXiv.<ID>``): fetch
     the real arXiv record for ``<ID>`` (arXiv Atom API, with DataCite as a
     cross-check). arXiv DOIs *do* resolve even though CrossRef ignores them.
  2. Otherwise: title-search arXiv, then OpenAlex, then CrossRef; take the best
     candidate by normalized-title SequenceMatcher ratio.
Compare the entry's title (normalized SequenceMatcher) and authors (last-name
set overlap / Jaccard) against the matched real paper, and apply the decision
rules below.

Every resolution is cached to a JSON file so re-runs are fully offline and
deterministic. Requests are throttled; timeouts/errors mark an entry UNRESOLVED
and never crash the run.

Decision rules (thresholds are deterministic; evidence recorded per entry)
--------------------------------------------------------------------------
TITLE_MATCH = normalized SequenceMatcher ratio >= 0.90
AUTHOR_MATCH = last-name set overlap (|intersection| / min(|a|,|b|)) >= 0.5
YEAR_OK = |entry_year - real_year| <= 1  (arXiv preprint vs proceedings year)

1. FLIP -> VALID
   TITLE_MATCH and AUTHOR_MATCH and YEAR_OK (when both years present).
   The citation is actually correct. Set label=VALID; drop hallucination_type
   and difficulty_tier (schema's VALID convention — the loader's ``from_dict``
   also strips them, and ``__post_init__`` rejects a VALID entry that keeps a
   hallucination_type); recompute all 6 sub-tests from the real paper.

2. RE-TYPE (keep HALLUCINATED, correct the type)
   TITLE_MATCH but the metadata is deliberately wrong:
     * authors wrong (AUTHOR_MATCH false)        -> swapped_authors
     * year shifted into the future              -> future_date
     * year shifted but not future (|d|>1)       -> arxiv_version_mismatch
   Fix hallucination_type + difficulty_tier + sub-tests to reflect the TRUE
   defect; keep label=HALLUCINATED.

3. KEEP plausible_fabrication
   No real paper matches the title on any source (best title sim < 0.90).
   Genuine fabrication. Keep label=HALLUCINATED; ensure sub-tests are correct
   (title_exists=false is now justified).

4. UNRESOLVED
   All sources timed out / errored (no candidate retrieved at all). Do not
   change; flag for manual review.

Also cleans VALID entries that carry the leftover inconsistency from the prior
32-key flips: a retained ``hallucination_type`` / ``difficulty_tier`` and/or
sub-tests still showing False title_exists/authors_match/doi_resolves inherited
from when they were HALLUCINATED. Only entries that were relabeled (carry a
``relabeled_*`` marker) or that violate the schema's VALID convention are
touched — legitimate VALID entries with a benign False sub-test (e.g.
``cross_db_agreement=false``, ``doi_resolves=false`` for a no-DOI entry,
``fields_complete=false``) are left alone.

Idempotency & determinism
--------------------------
The decision for an entry is a pure function of (entry fields, cached source
records). With a warm cache the run touches the network zero times and produces
byte-identical output. Re-running ``--apply`` on already-correct data is a no-op.

Usage
-----
    python scripts/relabel_ground_truth.py                 # DRY RUN (default)
    python scripts/relabel_ground_truth.py --apply         # Stage 2: write data files
    python scripts/relabel_ground_truth.py --no-network    # cache-only, offline
    python scripts/relabel_ground_truth.py --rate-limit 1.5

Outputs (dry run writes ONLY these, never the data files):
    results/reviewer_experiments/relabel_proposal.json     (per-entry decisions)
    results/reviewer_experiments/relabel_resolution_cache.json  (source records)
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import ssl
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the canonical schema so sub-test recomputation matches the loader.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hallmark.dataset.schema import (  # noqa: E402
    EXPECTED_SUBTESTS,
    HALLUCINATION_TIER_MAP,
    HallucinationType,
)

# ---------------------------------------------------------------------------
# SSL context — use certifi bundle on macOS if available
# ---------------------------------------------------------------------------
try:
    import certifi

    _SSL_CTX: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - environment dependent
    _SSL_CTX = ssl.create_default_context()

# Prefer defusedxml (hardens against XXE / billion-laughs) for the arXiv Atom
# feed; fall back to the stdlib parser if defusedxml is not installed. The arXiv
# API is a trusted source, but defending in depth costs nothing here. The Element
# type and ParseError come from the stdlib regardless of which parser is used.
import xml.etree.ElementTree as _stdlib_ET  # noqa: E402

_XMLElement = _stdlib_ET.Element
_ParseError = _stdlib_ET.ParseError
try:
    import defusedxml.ElementTree as _defused_ET  # type: ignore[import-untyped]

    _xml_fromstring = _defused_ET.fromstring
except ImportError:  # pragma: no cover - environment dependent
    _xml_fromstring = _stdlib_ET.fromstring

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _REPO_ROOT / "data"
SPLITS: dict[str, Path] = {
    "dev_public": DATA_DIR / "v1.0" / "dev_public.jsonl",
    "test_public": DATA_DIR / "v1.0" / "test_public.jsonl",
    "stress_test": DATA_DIR / "v1.0" / "stress_test.jsonl",
    "test_crossdomain": DATA_DIR / "v1.0" / "test_crossdomain.jsonl",
    "test_hidden": DATA_DIR / "hidden" / "test_hidden.jsonl",
}

RESULTS_DIR = _REPO_ROOT / "results" / "reviewer_experiments"
PROPOSAL_PATH = RESULTS_DIR / "relabel_proposal.json"
CACHE_PATH = RESULTS_DIR / "relabel_resolution_cache.json"

MAILTO = "rpatrik1996@gmail.com"
USER_AGENT = f"HALLMARK-Benchmark/2.1 (mailto:{MAILTO})"
RELABELED_BY = "systematic-relabel-2026-05-30"

# Deterministic decision thresholds
# ---------------------------------------------------------------------------
# A flip to VALID requires a FULLY-CORRECT citation: title must match a real
# paper EXACTLY (normalized), authors must match as a SET, and (where present)
# the defective field must check out. The old single TITLE_MATCH=0.90 gate
# flipped 1-token near_miss edits and overlap>=0.5 flipped author-defect cohorts.
TITLE_EXACT = 1.0  # normalized-title equality required to clear a title defect
TITLE_RETYPE_MIN = 0.78  # "this is the same real paper, but corrupted" floor for retype
AUTHOR_SET_TOL = 1  # max symmetric-difference last names to still call author-SET equal
YEAR_TOL = 1  # arXiv preprint vs proceedings year drift
VENUE_SIM = 0.45  # token-Jaccard floor for "venue is consistent with the real paper"

# Author-faithfulness gate (BUG-1 fix). A citation of a many-author paper is
# FAITHFUL when its author list is the real list, exactly OR truncated with an
# "and others"/"et al." marker (a correct-order leading prefix), tolerant of
# diacritic/LaTeX-encoding slips. We separate this from the deliberate defects:
#   * partial_author_list = a STRICT subset WITHOUT a marker that keeps the LAST
#     author (so it is NOT a contiguous leading prefix) -> stays HALLUCINATED.
#   * swapped_authors = a genuinely different author SET -> stays HALLUCINATED.
# Thresholds were calibrated on the cohort (OPT/PaLM/Flamingo/LLaMA/LLaMA-2/
# PaLM-E/Least-to-Most: subset 0.79-1.00) vs. genuine llm_generated swaps
# (subset <= 0.56) — a clean gap at ~0.70.
AUTHOR_SUBSET_FAITHFUL = 0.75  # >= this share of entry last-names are real authors
AUTHOR_PREFIX_FAITHFUL = 0.45  # ordered leading-prefix agreement floor (short lists)
AUTHOR_LONG_LIST = 10  # real lists this long: positional prefix is fragile; use subset
AUTHOR_EXTRA_TOL = 2  # entry may exceed real length by this much (initials/dup parsing)
# Given-name-aware verified-prefix floor for the subset/long-list branch. A list
# is faithful only when its CONTIGUOUS leading prefix — matching the real paper by
# BOTH last name AND given name, before the first divergence — reaches this depth
# (or covers the whole entry if shorter). Length alone is NOT faithfulness: it
# admitted cherry-picked lists with a wrong lead or a fabricated insertion behind a
# last-name collision (Make-A-Video, Open Problems RLHF, Least-to-Most, Discovering
# LM Behaviors, ConvMAE). The cohort's verified prefix is >= 6; the false flips
# bottom out at <= 3 — a clean gap at 5. (Over-flip fix.)
AUTHOR_VERIFY_RUN = 5

ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(?P<id>.+)$", re.IGNORECASE)
CURRENT_YEAR = date.today().year

# Deliberate-defect provenance. perturbation/adversarial entries are
# HALLUCINATED BY CONSTRUCTION (a known field was corrupted): they never flip
# to VALID. The accidental-real-paper cohort that may legitimately flip is the
# LLM-generated / real-world cohort.
DELIBERATE_DEFECT_METHODS = {"perturbation", "adversarial"}

# Per-source throttle floors (seconds between calls on that host). arXiv's
# export API and the OpenAlex public pool both 429 aggressively under load; the
# verifier diagnosis traced 1460 arXiv + 1020 OpenAlex 429s to a 1s global
# throttle. We space arXiv >= 3s and OpenAlex >= 1s (polite pool via mailto).
SOURCE_MIN_INTERVAL = {
    "arxiv": 3.0,
    "openalex": 1.0,
    "datacite": 1.0,
    "crossref": 1.0,
}
MAX_RETRIES = 4  # exponential backoff attempts on a transient (429/5xx/timeout)
# Live-coverage floor on the RE-RESOLVED set: if arXiv+OpenAlex together cover
# too little, keeps are CrossRef-only (the bug being fixed) -> FAIL LOUDLY.
MIN_LIVE_COVERAGE = 0.50
# Unresolved-fraction ceiling over ALL hallucinated entries (EXECUTION-HOLE fix).
# The live-coverage guard is gated on auth_total>0, so a zero-live warm run that
# punts everything to UNRESOLVED would pass silently. This guard ALSO fails (or
# blocks --apply) when too many entries could not be verified, regardless of
# whether any live call was made. With the SCOPING strategy (deliberate defects
# are KEPT, not unresolved; only plausible_fabrication/llm_generated need live
# resolution), the warm-cache unresolved fraction should be well under this.
MAX_UNRESOLVED_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Text / author utilities (shared conventions with check_mislabeled_entries.py)
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    # Strip ASCII punctuation plus en/em dashes (titles use them interchangeably).
    for ch in ".,;:!?\"'()[]{}/-–—":  # noqa: RUF001 (en/em dash stripping is intentional)
        t = t.replace(ch, " ")
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio on normalized titles."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def extract_last_names(author_str: str) -> set[str]:
    """Lowercased last names from a BibTeX author field.

    Handles both "Last, First" and "First Last" formats, separated by " and ".
    """
    last_names: set[str] = set()
    for author in author_str.split(" and "):
        author = author.strip()
        if not author:
            continue
        if "," in author:
            last_names.add(author.split(",")[0].strip().lower())
        else:
            tokens = author.split()
            if tokens:
                last_names.add(tokens[-1].lower())
    return {n for n in last_names if n}


def last_names_from_real(names: list[str]) -> set[str]:
    """Lowercased last names from a list of real-paper display names."""
    out: set[str] = set()
    for n in names:
        n = n.strip()
        if not n:
            continue
        if "," in n:
            out.add(n.split(",")[0].strip().lower())
        else:
            toks = n.split()
            if toks:
                out.add(toks[-1].lower())
    return {n for n in out if n}


def author_overlap(entry_authors: str, real_names: list[str]) -> float:
    """Overlap ratio = |intersection| / min(|entry|, |real|); 0 if either empty.

    Kept for the audit log (records how lax the old gate was); the flip decision
    now uses ``author_set_equal`` instead.
    """
    a = extract_last_names(entry_authors)
    b = last_names_from_real(real_names)
    if not a or not b:
        return 0.0
    inter = a & b
    return len(inter) / min(len(a), len(b))


def author_set_equal(entry_authors: str, real_names: list[str]) -> bool:
    """Order-insensitive FULL-list author-SET equality, tolerant of ``AUTHOR_SET_TOL``.

    The flip gate for any author-bearing defect: the entry's last-name set must
    equal the real paper's last-name set up to a small symmetric difference
    (one OCR/diacritic/initials slip). A dropped or swapped author IS the
    deliberate defect, so a strict-ish set comparison keeps it HALLUCINATED.
    ``overlap>=0.5`` was far too lax — it flipped PaLM (partial author lists).
    """
    a = extract_last_names(entry_authors)
    b = last_names_from_real(real_names)
    if not a or not b:
        return False
    sym = a.symmetric_difference(b)
    # Tolerate at most AUTHOR_SET_TOL differing last names, and only when both
    # lists are of comparable length (guards "kept first+last of a 7-author paper").
    if abs(len(a) - len(b)) > AUTHOR_SET_TOL:
        return False
    return len(sym) <= AUTHOR_SET_TOL


# ---------------------------------------------------------------------------
# Author-faithfulness classification (BUG-1 fix)
# ---------------------------------------------------------------------------
_LATEX_ACCENT_BRACED = re.compile(r"\{\\[a-zA-Z]+\s*([a-zA-Z])\}")  # {\~a}, {\"a}, {\c c}
_LATEX_ACCENT_BARE = re.compile(r"\\[`'\"^~=.cv]\{?([a-zA-Z])\}?")  # \"a, \'e, \`e, \c{c}
_LATEX_RESIDUE = re.compile(r"\\[a-zA-Z]+|[{}\\~^\"'`=]")

# Trailing truncation markers that signal a FAITHFUL leading-prefix citation.
_TRUNCATION_MARKERS = {"others", "et al", "et al.", "et~al", "et~al.", "etal"}


def _norm_lastname(token: str) -> str:
    """Last name normalized for comparison: strip LaTeX accents + Unicode diacritics.

    ``Sch{\"a}rli`` and ``Schärli`` collapse to ``scharli``; ``Rozi{\\`e}re`` and
    ``Rozière`` to ``roziere``. This removes the encoding-only divergences that
    made a FAITHFUL truncated citation look like an author defect (BUG 1).
    """
    t = _LATEX_ACCENT_BRACED.sub(r"\1", token)
    t = _LATEX_ACCENT_BARE.sub(r"\1", t)
    t = _LATEX_RESIDUE.sub("", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower().strip()


def _ordered_last_names(author: str) -> str:
    """Extract a single normalized last name from one BibTeX author token."""
    author = author.strip()
    if "," in author:
        return _norm_lastname(author.split(",")[0])
    toks = author.split()
    return _norm_lastname(toks[-1]) if toks else ""


def entry_last_names_ordered(author_str: str) -> tuple[list[str], bool]:
    """Ordered, diacritic-normalized entry last names + a truncation flag.

    Returns ``(last_names, truncated)`` where ``truncated`` is True iff the
    author field ends with an "and others"/"et al." marker (the marker itself is
    dropped from the list).
    """
    out: list[str] = []
    truncated = False
    for author in author_str.split(" and "):
        a = author.strip()
        if not a:
            continue
        if a.lower().rstrip(".") in {m.rstrip(".") for m in _TRUNCATION_MARKERS}:
            truncated = True
            continue
        ln = _ordered_last_names(a)
        if ln:
            out.append(ln)
    return out, truncated


def real_last_names_ordered(names: list[str]) -> list[str]:
    """Ordered, diacritic-normalized real-paper last names."""
    out: list[str] = []
    for n in names:
        ln = _ordered_last_names(n)
        if ln:
            out.append(ln)
    return out


def _norm_given(author: str) -> str:
    """Normalized GIVEN-name string of one author token (everything but the last
    name), diacritic/LaTeX-stripped. ``Zhou, Denny`` -> ``denny``; ``Peng Gao`` ->
    ``peng``. Empty when no given name is present (e.g. a bare last name)."""
    author = author.strip()
    if "," in author:
        parts = author.split(",", 1)
        given = parts[1] if len(parts) > 1 else ""
    else:
        toks = author.split()
        given = " ".join(toks[:-1]) if len(toks) > 1 else ""
    g = _LATEX_ACCENT_BRACED.sub(r"\1", given)
    g = _LATEX_ACCENT_BARE.sub(r"\1", g)
    g = _LATEX_RESIDUE.sub("", g)
    g = unicodedata.normalize("NFKD", g)
    g = "".join(c for c in g if not unicodedata.combining(c))
    return g.lower().strip()


def _given_name_conflict(entry_given: str, real_given: str) -> bool:
    """True iff two given-name strings clearly denote DIFFERENT people.

    Tolerant by design — only a CLEAR first-token mismatch is a conflict:
      * either side empty -> no conflict (cannot corroborate, never a false flag);
      * equal first tokens -> no conflict;
      * one first token PREFIXES the other -> no conflict (``J.``/``Jeff`` initials,
        ``Jeff``/``Jeffrey`` nicknames are the same person).
    A genuine substitution behind a last-name collision (``Shunyu`` vs ``Denny``
    Zhou; ``Dahua`` vs ``Ziyi`` Lin) trips the first-token mismatch and is caught.
    """
    if not entry_given or not real_given:
        return False
    et = re.findall(r"[a-z]+", entry_given)
    rt = re.findall(r"[a-z]+", real_given)
    if not et or not rt:
        return False
    a, b = et[0], rt[0]
    if a == b:
        return False
    # one first token PREFIXES the other -> same person (J./Jeff, Jeff/Jeffrey).
    return not (a.startswith(b) or b.startswith(a))


def entry_authors_tokens(author_str: str) -> list[str]:
    """Ordered raw entry author tokens (truncation markers dropped). Parallel to
    ``entry_last_names_ordered`` so positional given-name corroboration aligns."""
    out: list[str] = []
    for author in author_str.split(" and "):
        a = author.strip()
        if not a:
            continue
        if a.lower().rstrip(".") in {m.rstrip(".") for m in _TRUNCATION_MARKERS}:
            continue
        out.append(a)
    return out


def given_aware_contiguous_run(
    entry_authors: str, real_names: list[str], el: list[str], rl: list[str]
) -> int:
    """Length of the contiguous leading prefix matching the real paper by BOTH
    last name AND given name, before the first divergence.

    A last-name-only contiguous run over-counts when a reordered/truncated real
    list happens to collide last names positionally (PaLM/LLaMA-2 deep reorderings)
    — but those collisions sit DEEP, past a clean verified prefix. A masked
    SUBSTITUTION (``Shunyu``/``Denny`` Zhou at lead; ``Dahua``/``Ziyi`` Lin at
    pos 3) breaks the given-name check SHALLOW, exactly where it must.
    """
    e_tok = entry_authors_tokens(entry_authors)
    k = min(len(el), len(rl))
    run = 0
    for i in range(k):
        if el[i] != rl[i]:
            break
        eg = _norm_given(e_tok[i]) if i < len(e_tok) else ""
        rg = _norm_given(real_names[i]) if i < len(real_names) else ""
        if _given_name_conflict(eg, rg):
            break
        run += 1
    return run


def classify_authors(entry_authors: str, real_names: list[str]) -> dict[str, Any]:
    """Classify the entry's author list against the real paper's authors.

    Returns a dict with::

        {"verdict": "faithful" | "partial_no_marker" | "different",
         "truncated": bool, "subset_ratio": float, "prefix_ratio": float,
         "set_equal": bool, "entry_len": int, "real_len": int}

    Verdicts (BUG-1 logic):
      * ``faithful``           — the citation's authors are CORRECT. Either an
        exact set match (tol ``AUTHOR_SET_TOL``), OR a correct-order leading
        PREFIX (truncated / shorter than real) whose names are dominated by real
        authors (``subset_ratio >= AUTHOR_SUBSET_FAITHFUL``). For long real lists
        (>= ``AUTHOR_LONG_LIST``) positional prefix is fragile, so subset
        dominance + length-consistency suffices.
      * ``partial_no_marker``  — a strict subset WITHOUT a truncation marker (the
        schema's ``partial_author_list`` defect: "<50% authors, no 'et al.'").
        Stays HALLUCINATED.
      * ``different``          — the author SET genuinely differs (swapped /
        fabricated names). Stays HALLUCINATED.
    """
    el, truncated = entry_last_names_ordered(entry_authors)
    rl = real_last_names_ordered(real_names)
    result: dict[str, Any] = {
        "verdict": "different",
        "truncated": truncated,
        "subset_ratio": 0.0,
        "prefix_ratio": 0.0,
        "set_equal": False,
        "entry_len": len(el),
        "real_len": len(rl),
    }
    if not el or not rl:
        return result

    eset, rset = set(el), set(rl)
    sym = eset.symmetric_difference(rset)
    # set_equal tolerates ``AUTHOR_SET_TOL`` differing names ONLY as a same-length
    # SUBSTITUTION (a diacritic/initials slip). A LENGTH difference is a dropped or
    # added author — NOT a clean set match — so it must flow to the partial/prefix
    # logic (else a partial_author_list that drops 1 of 3 authors looks set-equal).
    set_equal = len(eset) == len(rset) and len(sym) <= AUTHOR_SET_TOL
    result["set_equal"] = set_equal

    subset_ratio = sum(1 for x in el if x in rset) / len(el)
    k = min(len(el), len(rl))
    prefix_ratio = sum(1 for i in range(k) if el[i] == rl[i]) / len(el)
    # Contiguous leading run: how many of the entry's first names match the real
    # paper's first names IN ORDER before the first divergence. A faithful
    # truncated citation lists a long leading run; a first+last partial list runs
    # for only ~1 (first author) then jumps to the real LAST author.
    contig_run = 0
    for i in range(k):
        if el[i] == rl[i]:
            contig_run += 1
        else:
            break
    # Coverage: fraction of the REAL authors the entry names (capped at 1.0 for
    # over-long parses). A near-complete list (LLaMA-2/Flamingo/L2M) is faithful;
    # a first+last partial keeps only a small fraction.
    coverage = min(1.0, sum(1 for x in rset if x in eset) / len(rl))
    # Given-name-aware verified prefix: the contiguous leading run matching the real
    # paper by BOTH last name AND given name (catches collision-masked substitutions
    # a last-name-only ``contig_run`` would miss). ``lead_ok`` requires the FIRST
    # author to match by last name; ``lead_given_conflict`` flags a wrong lead behind
    # a last-name collision (Shunyu vs Denny Zhou). ``first_missing_pos`` locates the
    # earliest entry author absent from the real set — a candidate fabricated
    # insertion. (Over-flip fix: list length alone is NOT faithfulness.)
    gcontig = given_aware_contiguous_run(entry_authors, real_names, el, rl)
    e_tok = entry_authors_tokens(entry_authors)
    lead_ok = el[0] == rl[0]
    lead_given_conflict = bool(e_tok and real_names) and _given_name_conflict(
        _norm_given(e_tok[0]), _norm_given(real_names[0])
    )
    first_missing_pos = next((i for i, x in enumerate(el) if x not in rset), None)
    result["subset_ratio"] = round(subset_ratio, 4)
    result["prefix_ratio"] = round(prefix_ratio, 4)
    result["contig_run"] = contig_run
    result["coverage"] = round(coverage, 4)
    result["gcontig"] = gcontig
    result["lead_ok"] = lead_ok
    result["lead_given_conflict"] = lead_given_conflict

    # Required preconditions for ANY faithful verdict (over-flip guards): the lead
    # author must match by last name AND not conflict on given name. This kills a
    # cherry-picked list with a WRONG LEAD author (Make-A-Video: Gafni vs Singer)
    # or a collision-masked lead substitution (Least-to-Most: Shunyu vs Denny Zhou).
    lead_corroborated = lead_ok and not lead_given_conflict
    # The given-name-aware verified prefix must reach AUTHOR_VERIFY_RUN, or cover the
    # whole entry / real list when shorter. This is the faithfulness signal that
    # replaces the bare long-list shortcut.
    need = min(len(el), len(rl), AUTHOR_VERIFY_RUN)
    verified_prefix = gcontig >= need

    # 1. Exact set match -> faithful (one diacritic/initials slip tolerated). Only
    #    a SAME-LENGTH substitution counts (a length diff is a drop/add, below). A
    #    collision-masked substitution (ConvMAE: Dahua vs Ziyi Lin, len-equal) is
    #    NOT faithful — it trips the given-aware verified-prefix / lead guard.
    if set_equal:
        if lead_corroborated and verified_prefix:
            result["verdict"] = "faithful"
            return result
        # set-equal by last name but a detectable given-name substitution in the
        # leading prefix -> the deliberate author defect stands.
        result["verdict"] = "different"
        return result

    length_ok = len(el) <= len(rl) + AUTHOR_EXTRA_TOL
    subset_ok = subset_ratio >= AUTHOR_SUBSET_FAITHFUL
    is_subset = eset.issubset(rset)
    # A CLEAN contiguous leading prefix: the entry's names are exactly the real
    # paper's first ``len(el)`` names, in order. This is the signature of a
    # faithful truncated citation. A first+last partial list is NOT clean (it
    # keeps the real LAST author, skipping the middle), so it fails this test.
    clean_leading_prefix = len(el) <= len(rl) and el == rl[: len(el)]

    # 2. partial_author_list defect (check BEFORE faithful so a first+last subset
    #    is not swept into "faithful"): no truncation marker, a strict subset of
    #    the real authors that DROPS interior authors — i.e. NOT a clean leading
    #    prefix. This is exactly the generator's "first + maybe-one-middle + last".
    if not truncated and is_subset and len(el) < len(rl) and not clean_leading_prefix:
        result["verdict"] = "partial_no_marker"
        return result

    # 3. Faithful (possibly truncated) leading citation. Every faithful path now
    #    requires a corroborated lead AND a given-name-aware verified prefix — the
    #    bare ``long_list`` shortcut (list length alone) is dropped.
    if not lead_corroborated:
        # wrong / collision-substituted lead -> author defect (handled below).
        pass
    elif truncated and (clean_leading_prefix or (subset_ok and length_ok)) and verified_prefix:
        # explicit truncation marker + dominant subset + verified prefix: the later
        # names are legitimately dropped (OPT/PaLM-a/LLaMA/PaLM-E).
        result["verdict"] = "faithful"
        return result
    elif clean_leading_prefix and len(el) < len(rl) and verified_prefix:
        result["verdict"] = "faithful"
        return result
    elif subset_ok and length_ok and verified_prefix:
        no_fabricated_insertion = (
            # every entry author is a real author (the absent names, if any, come
            # only from a truncated/reordered real list, all AFTER the verified
            # prefix), ...
            first_missing_pos is None
            # ... or any absent name sits AT/AFTER the verified prefix end and the
            # prefix is itself deep (>= AUTHOR_VERIFY_RUN). A SHALLOW absent name
            # inside the verifiable region is a fabricated insertion (Open Problems
            # RLHF: Halpern @2, contig 1; Discovering LM: Steinhardt @7, gcontig 2).
            or (gcontig >= AUTHOR_VERIFY_RUN and first_missing_pos >= gcontig)
        )
        if no_fabricated_insertion:
            result["verdict"] = "faithful"
            return result

    # 4. Strict subset WITHOUT a marker (any other shape) -> partial list defect.
    if not truncated and is_subset and len(el) < len(rl):
        result["verdict"] = "partial_no_marker"
        return result

    # 5. Otherwise the author set genuinely differs.
    result["verdict"] = "different"
    return result


def venue_of(fields: dict[str, Any]) -> str:
    return (fields.get("booktitle") or fields.get("journal") or "").strip()


def venue_consistent(entry_venue: str, real_venue: str) -> bool:
    """True if the entry venue is plausibly the real paper's venue.

    Token-Jaccard with substring shortcut (handles abbreviations like
    'NeurIPS' vs 'Advances in Neural Information Processing Systems' poorly, so
    a low score is treated as 'cannot confirm' -> conservative KEEP, never a
    false flip). arXiv-as-venue counts as consistent with any real venue only
    when the real record is itself arXiv.
    """
    if not entry_venue or not real_venue:
        # No real venue to compare against: cannot confirm consistency.
        return False
    e, r = entry_venue.lower(), real_venue.lower()
    if e in r or r in e:
        return True
    et = {w for w in re.split(r"\W+", e) if len(w) > 2}
    rt = {w for w in re.split(r"\W+", r) if len(w) > 2}
    if not et or not rt:
        return False
    j = len(et & rt) / len(et | rt)
    return j >= VENUE_SIM


def venue_is_authoritative(best: dict[str, Any] | None) -> bool:
    """True iff the matched real record carries a TRUE venue we can compare against.

    The arXiv DataCite DOI (``10.48550/arXiv.*``) always reports publisher
    "arXiv" — that is the preprint server, NOT a venue authority. A landmark
    paper also published at ACL/ICML/NeurIPS still resolves to publisher="arXiv"
    via its arXiv DOI. So when the matched record is an arXiv/DataCite preprint
    (or its venue normalizes to "arxiv"), we CANNOT confirm or refute the entry's
    venue — venue must be treated as UNKNOWN (never a flip-blocking "wrong
    venue"). OpenAlex / CrossRef venues (a real proceedings/journal string) ARE
    authoritative. (BUG-1/BUG-2 venue false-negative fix.)
    """
    if best is None:
        return False
    src = (best.get("source") or "").lower()
    rv = (best.get("venue") or "").strip().lower()
    if src in ("datacite", "arxiv"):
        return False
    return rv not in ("", "arxiv", "arxiv.org", "arxiv preprint")


def parse_year(value: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(m.group(0)) if m else None


def arxiv_id_year(arxiv_id: str) -> int | None:
    """Derive the publication year from an arXiv ID's YYMM prefix.

    Modern arXiv IDs are ``YYMM.NNNNN`` (since 2007-04); ``2003.00982`` -> 2020.
    Returns None for old-style IDs (e.g. ``cs/0309048``) we cannot decode.
    """
    m = re.match(r"^(\d{2})(\d{2})\.", arxiv_id.strip())
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    if not (1 <= mm <= 12):
        return None
    # arXiv's YYMM scheme started 2007; 00-06 would be 2100s, impossible here.
    return 2000 + yy


def authoritative_year(
    entry_doi: str, candidates: list[dict[str, Any]], best: dict[str, Any]
) -> tuple[int | None, str]:
    """Pick the most reliable 'real' year for the year-delta comparison.

    Source-quality order (most → least authoritative):
      1. the entry's OWN arXiv DOI, decoded from its YYMM prefix (the citation's
         own claimed identifier resolves to a fixed month/year);
      2. an arXiv candidate's year, or its ID decoded from YYMM;
      3. a DataCite candidate's year;
      4. the best candidate's year (may be a noisy OpenAlex/CrossRef year).

    Returns (year, source_tag). This avoids the OpenAlex artifact where a
    well-known 2017 paper is reported with a spurious recent year.
    """
    m = ARXIV_DOI_RE.match(entry_doi or "")
    if m:
        y = arxiv_id_year(m.group("id"))
        if y is not None:
            return y, "entry-arxiv-doi"
    for src in ("arxiv", "datacite"):
        for c in candidates:
            if c.get("source") == src:
                if c.get("year") is not None:
                    return int(c["year"]), src
                if src == "arxiv" and c.get("id"):
                    y = arxiv_id_year(str(c["id"]))
                    if y is not None:
                        return y, "arxiv-id"
    by = best.get("year")
    return (int(by) if by is not None else None), best.get("source", "best")


# ---------------------------------------------------------------------------
# HTTP / source clients (with on-disk cache)
# ---------------------------------------------------------------------------


class Resolver:
    """Multi-source bibliographic resolver with an on-disk cache.

    Every public method returns a normalized record dict::

        {"source": <str>, "id": <str>, "title": <str>,
         "authors": [<display name>, ...], "year": <int|None>,
         "venue": <str>, "doi": <str|None>}

    or ``None`` (definitively not found) or raises nothing (errors -> None,
    but a *network error* is recorded as ``__error__`` in the cache so the
    caller can distinguish "looked, found nothing" from "could not look").
    """

    def __init__(
        self,
        cache: dict[str, Any],
        *,
        rate_limit: float = 1.0,
        timeout: int = 30,
        allow_network: bool = True,
    ) -> None:
        self.cache = cache
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.allow_network = allow_network
        # Per-thread "cache-only" override (SCOPING fix): when set, this worker's
        # fetches read the cache but never touch the network — used so deliberate
        # perturbation/adversarial entries (HALLUCINATED by construction) are
        # re-typed from cache only and never spend a live API call. Live calls are
        # reserved for the flip-candidate cohort (plausible_fabrication /
        # llm_generated / real_world / possible NO-OP perturbations).
        self._tls = threading.local()
        # Per-source throttle clocks (host -> last monotonic request time).
        self._last_by_source: dict[str, float] = {}
        self.network_calls = 0
        # Per-source live success / transient-error tallies for the coverage log.
        self.source_live: dict[str, dict[str, int]] = {}
        # Thread-safety: parallel workers share the cache dict and the throttle
        # clocks. Decisions stay deterministic because each entry's verdict is a
        # pure function of its own URL fetches (keyed by URL in the cache); only
        # the *order* of network calls — never any per-entry result — depends on
        # thread scheduling.
        self._throttle_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()

    # -- low-level fetch -----------------------------------------------------
    @staticmethod
    def _source_of(url: str) -> str:
        if "arxiv.org" in url:
            return "arxiv"
        if "openalex.org" in url:
            return "openalex"
        if "datacite.org" in url:
            return "datacite"
        if "crossref.org" in url:
            return "crossref"
        return "other"

    def _throttle(self, source: str) -> None:
        """Space out network calls PER SOURCE; safe under concurrent workers.

        arXiv needs >= 3s; OpenAlex/CrossRef/DataCite >= 1s. A global 1s clock
        (the old behavior) hammered arXiv into sustained 429s.
        """
        min_iv = max(self.rate_limit, SOURCE_MIN_INTERVAL.get(source, self.rate_limit))
        with self._throttle_lock:
            now = time.monotonic()
            last = self._last_by_source.get(source, 0.0)
            wait = min_iv - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last_by_source[source] = time.monotonic()
            self.network_calls += 1

    @staticmethod
    def _is_transient(err: str | None) -> bool:
        """True for errors that must NEVER be cached as a definitive 'no match'."""
        if not err:
            return False
        if err.startswith("HTTP "):
            try:
                code = int(err.split()[1])
            except (IndexError, ValueError):
                return True
            return code == 429 or 500 <= code < 600
        # timeout / DNS / SSL / reset / connection refused, etc.
        return True

    def _record_live(self, source: str, *, ok: bool, transient: bool) -> None:
        with self._stats_lock:
            s = self.source_live.setdefault(source, {"ok": 0, "transient": 0, "notfound": 0})
            if ok:
                s["ok"] += 1
            elif transient:
                s["transient"] += 1
            else:
                s["notfound"] += 1

    def _fetch(self, url: str) -> tuple[str | None, str | None]:
        """Return (body, error). Cached. body is None on error.

        Transient errors (HTTP 429/5xx/timeout/DNS) are retried with exponential
        backoff up to MAX_RETRIES and are NEVER written to the cache as a
        verdict — only a successful body or a DEFINITIVE not-found (HTTP 404) is
        cached. This is the core cache-poisoning fix: a warm re-run with
        --no-network reproduces from the good cache without replaying 429s.
        """
        ck = f"GET::{url}"
        with self._cache_lock:
            if ck in self.cache:
                cached = self.cache[ck]
                return cached.get("body"), cached.get("error")
        if not self.allow_network or getattr(self._tls, "cache_only", False):
            return None, "no-network (cache miss)"

        source = self._source_of(url)
        body: str | None = None
        err: str | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle(source)
            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", USER_AGENT)
                with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                err = None
                break
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if not self._is_transient(err):
                    break  # 404 etc. — definitive, stop retrying
            except Exception as exc:  # timeout, DNS, SSL, reset, ...
                err = f"{type(exc).__name__}: {exc}"
            # transient -> exponential backoff (1.5, 3, 6, 12 ... seconds * source floor)
            if attempt < MAX_RETRIES - 1:
                backoff = SOURCE_MIN_INTERVAL.get(source, 1.0) * (2 ** (attempt + 1))
                time.sleep(min(backoff, 60.0))

        transient = self._is_transient(err)
        self._record_live(source, ok=(err is None), transient=transient)
        # Cache ONLY definitive outcomes: a body, or a non-transient not-found.
        if err is None or not transient:
            with self._cache_lock:
                self.cache[ck] = {"body": body, "error": err}
        return body, err

    @contextmanager
    def cache_only(self, enabled: bool) -> Iterator[None]:
        """Within this block, this thread's fetches never hit the network.

        Used by the per-entry SCOPING gate so deliberate-defect entries are
        re-typed from cache alone (no live API spend). Thread-local so concurrent
        workers can independently scope their own entries.
        """
        prev = getattr(self._tls, "cache_only", False)
        self._tls.cache_only = enabled
        try:
            yield
        finally:
            self._tls.cache_only = prev

    # -- arXiv ---------------------------------------------------------------
    # NOTE: each public search now returns (record|None, error|None) so the
    # per-entry resolver can distinguish "looked, found nothing" (error None)
    # from "could not look" (transient error) — the UNRESOLVED gate (fix d).
    def arxiv_by_id(self, arxiv_id: str) -> tuple[dict[str, Any] | None, str | None]:
        # strip a trailing version (v1, v2, ...) for the lookup. HTTPS endpoint.
        clean = re.sub(r"v\d+$", "", arxiv_id.strip())
        url = (
            f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(clean)}&max_results=1"
        )
        body, err = self._fetch(url)
        if body is None:
            return None, err
        return self._parse_arxiv_atom(body, prefer_id=clean), err

    def arxiv_search(self, title: str) -> tuple[dict[str, Any] | None, str | None]:
        q = urllib.parse.quote(f'ti:"{title}"')
        url = f"https://export.arxiv.org/api/query?search_query={q}&max_results=3"
        body, err = self._fetch(url)
        if body is None:
            return None, err
        return self._best_arxiv_candidate(body, title), err

    @staticmethod
    def _ns() -> dict[str, str]:
        return {"a": "http://www.w3.org/2005/Atom"}

    def _parse_arxiv_atom(self, body: str, prefer_id: str | None = None) -> dict[str, Any] | None:
        try:
            root = _xml_fromstring(body)
        except _ParseError:
            return None
        ns = self._ns()
        entries = root.findall("a:entry", ns)
        if not entries:
            return None
        return self._arxiv_entry_to_record(entries[0], ns)

    def _best_arxiv_candidate(self, body: str, title: str) -> dict[str, Any] | None:
        try:
            root = _xml_fromstring(body)
        except _ParseError:
            return None
        ns = self._ns()
        best: dict[str, Any] | None = None
        best_sim = -1.0
        for entry in root.findall("a:entry", ns):
            rec = self._arxiv_entry_to_record(entry, ns)
            if rec is None:
                continue
            sim = title_similarity(title, rec["title"])
            if sim > best_sim:
                best_sim, best = sim, rec
        return best

    def _arxiv_entry_to_record(
        self, entry: _XMLElement, ns: dict[str, str]
    ) -> dict[str, Any] | None:
        title_el = entry.find("a:title", ns)
        if title_el is None or not title_el.text:
            return None
        title = " ".join(title_el.text.split())
        authors: list[str] = []
        for a in entry.findall("a:author", ns):
            name_el = a.find("a:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text)
        published = entry.find("a:published", ns)
        year = parse_year(published.text) if published is not None and published.text else None
        id_el = entry.find("a:id", ns)
        aid = ""
        if id_el is not None and id_el.text:
            aid = id_el.text.rsplit("/abs/", 1)[-1]
        return {
            "source": "arxiv",
            "id": aid,
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "arXiv",
            "doi": f"10.48550/arXiv.{re.sub(r'v[0-9]+$', '', aid)}" if aid else None,
        }

    # -- DataCite (authoritative for arXiv DataCite DOIs 10.48550/arXiv.*) ----
    def datacite_by_doi(self, doi: str) -> tuple[dict[str, Any] | None, str | None]:
        url = f"https://api.datacite.org/dois/{urllib.parse.quote(doi, safe='')}"
        body, err = self._fetch(url)
        if body is None:
            return None, err
        try:
            attr = json.loads(body).get("data", {}).get("attributes", {})
        except json.JSONDecodeError:
            return None, err
        titles = attr.get("titles") or []
        title = titles[0].get("title", "") if titles else ""
        if not title:
            return None, err
        creators = attr.get("creators") or []
        authors = []
        for c in creators:
            name = c.get("name") or ""
            if not name and (c.get("givenName") or c.get("familyName")):
                name = f"{c.get('givenName', '')} {c.get('familyName', '')}".strip()
            if name:
                authors.append(name)
        return {
            "source": "datacite",
            "id": attr.get("doi", doi),
            "title": " ".join(title.split()),
            "authors": authors,
            "year": attr.get("publicationYear"),
            "venue": attr.get("publisher", ""),
            "doi": attr.get("doi", doi),
        }, err

    # -- OpenAlex (polite pool via mailto; preferred title-search source) -----
    def openalex_search(self, title: str) -> tuple[dict[str, Any] | None, str | None]:
        q = urllib.parse.quote(title)
        url = f"https://api.openalex.org/works?search={q}&per-page=3&mailto={MAILTO}"
        body, err = self._fetch(url)
        if body is None:
            return None, err
        try:
            results = json.loads(body).get("results", [])
        except json.JSONDecodeError:
            return None, err
        best: dict[str, Any] | None = None
        best_sim = -1.0
        for r in results:
            rec = self._openalex_to_record(r)
            if rec is None:
                continue
            sim = title_similarity(title, rec["title"])
            if sim > best_sim:
                best_sim, best = sim, rec
        return best, err

    @staticmethod
    def _openalex_to_record(r: dict[str, Any]) -> dict[str, Any] | None:
        title = r.get("title") or r.get("display_name") or ""
        if not title:
            return None
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in r.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        loc = (r.get("primary_location") or {}).get("source") or {}
        venue = loc.get("display_name", "") if isinstance(loc, dict) else ""
        doi = r.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        return {
            "source": "openalex",
            "id": r.get("id", ""),
            "title": " ".join(title.split()),
            "authors": authors,
            "year": r.get("publication_year"),
            "venue": venue,
            "doi": doi,
        }

    # -- CrossRef (corroboration only; never sufficient alone for a keep) -----
    def crossref_search(self, title: str) -> tuple[dict[str, Any] | None, str | None]:
        params = {"query.bibliographic": title, "rows": "3", "mailto": MAILTO}
        url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
        body, err = self._fetch(url)
        if body is None:
            return None, err
        try:
            items = json.loads(body).get("message", {}).get("items", [])
        except json.JSONDecodeError:
            return None, err
        best: dict[str, Any] | None = None
        best_sim = -1.0
        for it in items:
            rec = self._crossref_to_record(it)
            if rec is None:
                continue
            sim = title_similarity(title, rec["title"])
            if sim > best_sim:
                best_sim, best = sim, rec
        return best, err

    def crossref_by_doi(self, doi: str) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve a publisher DOI directly (the entry's OWN DOI).

        Used to gate fabricated_doi / hybrid_fabrication flips: a DOI defect can
        only be cleared by resolving the DOI itself. HTTP 404 -> definitive
        non-resolution (record None, err 'HTTP 404'); transient -> err set, None.
        """
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}?mailto={MAILTO}"
        body, err = self._fetch(url)
        if body is None:
            return None, err
        try:
            msg = json.loads(body).get("message", {})
        except json.JSONDecodeError:
            return None, err
        if not msg:
            return None, err
        rec = self._crossref_to_record(msg)
        return rec, err

    @staticmethod
    def _crossref_to_record(it: dict[str, Any]) -> dict[str, Any] | None:
        titles = it.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            return None
        authors = []
        for a in it.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            nm = f"{given} {family}".strip()
            if nm:
                authors.append(nm)
        ct = it.get("container-title") or []
        venue = ct[0] if ct else ""
        year = None
        for k in ("published-print", "published-online", "issued", "created"):
            dp = (it.get(k) or {}).get("date-parts")
            if dp and dp[0] and dp[0][0]:
                year = dp[0][0]
                break
        return {
            "source": "crossref",
            "id": it.get("DOI", ""),
            "title": " ".join(title.split()),
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": it.get("DOI"),
        }


# ---------------------------------------------------------------------------
# Per-entry resolution: gather the best real-paper candidate from all sources.
# ---------------------------------------------------------------------------

# Flip-eligible provenance: only these may flip to VALID, so only these warrant a
# LIVE API call. Everything else (perturbation/adversarial) is HALLUCINATED by
# construction and is re-typed from CACHE ONLY (SCOPING fix — keeps live calls
# small enough to finish in one run; arXiv title-search is 429-hammered).
FLIP_ELIGIBLE_METHODS = {"llm_generated", "real_world"}
FLIP_ELIGIBLE_TYPES = {HallucinationType.PLAUSIBLE_FABRICATION.value}


def needs_live_resolution(entry: dict[str, Any]) -> bool:
    """True iff this entry may flip and therefore warrants a LIVE API call.

    The flip decision is reserved for the accidental-real-paper cohort:
    ``plausible_fabrication`` (any provenance) and any ``llm_generated`` /
    ``real_world`` entry. Deliberate ``perturbation`` / ``adversarial`` entries
    carry a known by-construction defect — they never flip, so we re-type them
    from cache alone and spend no live call on them.
    """
    gm = entry.get("generation_method")
    ht = entry.get("hallucination_type")
    if gm in FLIP_ELIGIBLE_METHODS:
        return True
    return ht in FLIP_ELIGIBLE_TYPES


def resolve_entry(entry: dict[str, Any], resolver: Resolver) -> dict[str, Any]:
    """Return a resolution dict for one entry.

    {
      "candidates": [<record>, ...],      # all retrieved (for the audit log)
      "best": <record|None>,              # highest title-sim candidate
      "best_sim": float,                  # title sim of best
      "arxiv_doi_id": <str|None>,         # if entry DOI is an arXiv DataCite DOI
      "errored": bool,                    # at least one source raised network error
      "any_response": bool,               # any source returned a usable record
    }
    """
    fields = entry.get("fields", {})
    title = fields.get("title", "")
    doi = (fields.get("doi") or "").strip()

    candidates: list[dict[str, Any]] = []
    # Per-source bookkeeping: did we get a usable candidate, and did the call
    # hit a transient error (so "no candidate" might be "could not look")?
    cand_by_source: dict[str, bool] = {}
    err_by_source: dict[str, bool] = {}

    def _note(source: str, rec: dict[str, Any] | None, err: str | None) -> None:
        if rec is not None:
            cand_by_source[source] = True
            candidates.append(rec)
        if err is not None and Resolver._is_transient(err):
            err_by_source[source] = True

    arxiv_id = None
    m = ARXIV_DOI_RE.match(doi)
    if m:
        arxiv_id = m.group("id")

    # 1. arXiv DataCite DOI -> authoritative real paper for this DOI.
    # The DOI *is* the paper's identity. Prefer DataCite REST (authoritative for
    # 10.48550/arXiv.*) plus the arXiv Atom record. If either already matches the
    # entry title strongly, title-search across the other sources can only
    # retrieve the SAME paper (corroborating) or a lower-sim different paper
    # (never chosen) — so we skip those calls (decision-preserving, fewer 429s).
    DOI_AUTHORITATIVE_SIM = 0.92
    doi_authoritative = False
    if arxiv_id:
        dc, dc_err = resolver.datacite_by_doi(doi)
        _note("datacite", dc, dc_err)
        dc_matches = (
            dc is not None and title_similarity(title, dc["title"]) >= DOI_AUTHORITATIVE_SIM
        )
        if dc_matches:
            doi_authoritative = True
        # arXiv Atom record: DataCite is already AUTHORITATIVE for 10.48550/arXiv.*,
        # so the Atom record is only corroboration. When DataCite already confirms
        # the paper, fetch the Atom record CACHE-ONLY (avoid the ≥3s arXiv live
        # call + 429 backoff — the SCOPING "use cached arXiv records" rule). When
        # DataCite did NOT confirm, allow a live Atom lookup to resolve identity.
        with resolver.cache_only(dc_matches):
            rec, ax_err = resolver.arxiv_by_id(arxiv_id)
        if rec is not None:
            _note("arxiv", rec, None)
            if title_similarity(title, rec["title"]) >= DOI_AUTHORITATIVE_SIM:
                doi_authoritative = True
        elif not dc_matches:
            # Only record a transient arXiv error when we actually attempted live.
            _note("arxiv", None, ax_err)

    # 2. Title search across sources. OpenAlex first (most generous title search,
    # polite pool), then CrossRef (corroboration). arXiv title-search is
    # DELIBERATELY AVOIDED as a LIVE call — it 429-hammers (3s floor + sustained
    # throttling); we only use a CACHED arXiv title-search result if one already
    # exists (SCOPING strategy). OpenAlex is the authoritative title-search source
    # for the no-DOI flip-candidate cohort.
    if title and not doi_authoritative:
        oa, oa_err = resolver.openalex_search(title)
        _note("openalex", oa, oa_err)
        # arXiv title-search: cache-only (never a fresh live 429-prone call).
        with resolver.cache_only(True):
            ax, _ax_err = resolver.arxiv_search(title)
        # A cache-miss here returns err "no-network (cache miss)" (transient) but
        # OpenAlex/CrossRef are the authoritative title sources, so a missing
        # arXiv title-search does NOT by itself flag the entry unresolvable.
        if ax is not None:
            _note("arxiv", ax, None)
        cr, cr_err = resolver.crossref_search(title)
        _note("crossref", cr, cr_err)

    # Resolve the entry's OWN non-arXiv DOI (fabricated_doi / hybrid_fabrication
    # gate): does it resolve at all, and to a paper matching the entry title?
    own_doi_record: dict[str, Any] | None = None
    own_doi_resolves: bool | None = None  # None = no DOI / not checked
    if doi and not arxiv_id:
        dd, dd_err = resolver.crossref_by_doi(doi)
        if dd is not None:
            own_doi_resolves = True
            own_doi_record = dd
            candidates.append(dd)
            cand_by_source["crossref-doi"] = True
        elif dd_err is not None and Resolver._is_transient(dd_err):
            own_doi_resolves = None  # could not check -> unknown
        else:
            own_doi_resolves = False  # definitive: DOI does not resolve

    # Determine the best candidate by title similarity to the ENTRY title.
    best: dict[str, Any] | None = None
    best_sim = 0.0
    for rec in candidates:
        sim = title_similarity(title, rec["title"])
        rec["_title_sim"] = round(sim, 4)
        if sim > best_sim:
            best_sim, best = sim, rec

    # UNRESOLVED gate (fix d): trust a keep/flip only when at least one of the
    # AUTHORITATIVE sources (arXiv or OpenAlex or DataCite) actually responded.
    # If those all failed to return a candidate AND at least one was a transient
    # error, we genuinely could not look — flag unresolved (don't trust CrossRef
    # alone, the very bug being fixed). A definitive CrossRef-only hit is still
    # recorded as best, but `authoritative_responded` lets decide() downgrade it.
    authoritative_responded = any(cand_by_source.get(s) for s in ("arxiv", "openalex", "datacite"))
    authoritative_errored = any(err_by_source.get(s) for s in ("arxiv", "openalex", "datacite"))
    any_response = bool(candidates)
    # "errored" = we could not look at all (no candidate from anywhere) and a
    # transient error occurred on some source.
    errored = (not any_response) and bool(err_by_source)

    return {
        "candidates": candidates,
        "best": best,
        "best_sim": round(best_sim, 4),
        "arxiv_doi_id": arxiv_id,
        "errored": errored,
        "any_response": any_response,
        "authoritative_responded": authoritative_responded,
        "authoritative_errored": authoritative_errored,
        "own_doi_resolves": own_doi_resolves,
        "own_doi_record": own_doi_record,
        "cand_by_source": cand_by_source,
        "err_by_source": err_by_source,
    }


# ---------------------------------------------------------------------------
# Sub-test recomputation
# ---------------------------------------------------------------------------


def subtests_for_valid(
    entry_fields: dict[str, Any], best: dict[str, Any]
) -> dict[str, bool | None]:
    """Recompute the 6 sub-tests for a now-VALID entry from the real paper."""
    doi = (entry_fields.get("doi") or "").strip()
    has_arxiv_doi = bool(ARXIV_DOI_RE.match(doi))
    real_doi = best.get("doi")
    # doi_resolves: True if entry has a DOI that we resolved (arXiv DOI resolves
    # via arXiv/DataCite; a matching real DOI resolves); None if entry has no DOI.
    if not doi:
        doi_resolves: bool | None = None
    elif has_arxiv_doi or (real_doi and doi.lower() == str(real_doi).lower()):
        doi_resolves = True
    else:
        # entry has a DOI but it isn't the matched paper's DOI / not arXiv.
        # Conservative: leave as resolved-unknown True only when arxiv; else None.
        doi_resolves = None
    return {
        "doi_resolves": doi_resolves,
        "title_exists": True,
        "authors_match": True,
        "venue_correct": True,
        "fields_complete": bool(
            entry_fields.get("title") and entry_fields.get("author") and entry_fields.get("year")
        ),
        "cross_db_agreement": True,
    }


def subtests_for_type(htype: str, entry_fields: dict[str, Any]) -> dict[str, bool | None]:
    """Sub-tests for a retyped HALLUCINATED entry, from the schema's ground truth.

    Use ``EXPECTED_SUBTESTS`` (the same table the generators copy) so a retype
    to ANY type produces the canonical sub-tests — not just the three the old
    code special-cased. Dynamic fields (``doi_resolves``, ``fields_complete``)
    are filled from the entry's own fields.
    """
    title = entry_fields.get("title", "")
    author = entry_fields.get("author", "")
    year = entry_fields.get("year", "")
    doi = (entry_fields.get("doi") or "").strip()
    base: dict[str, bool | None] = dict(EXPECTED_SUBTESTS[HallucinationType(htype)])
    # doi_resolves: None when the schema marks it dynamic — resolve from entry.
    if base.get("doi_resolves") is None:
        base["doi_resolves"] = True if doi else None
    # fields_complete: future_date keeps the schema's hardcoded False; all other
    # types derive it from the entry's own (title, author, year) presence.
    if htype != HallucinationType.FUTURE_DATE.value:
        base["fields_complete"] = bool(title and author and year)
    return base


def subtests_for_fabrication(entry_fields: dict[str, Any]) -> dict[str, bool | None]:
    """Sub-tests for a confirmed plausible_fabrication (no real paper)."""
    title = entry_fields.get("title", "")
    author = entry_fields.get("author", "")
    year = entry_fields.get("year", "")
    return {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_correct": True,
        "fields_complete": bool(title and author and year),
        "cross_db_agreement": False,
    }


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


# Hallucination types whose defect lives in the TITLE field. A flip requires
# EXACT normalized-title equality; a near-miss 1-token edit IS the defect.
_TITLE_DEFECT_TYPES = {
    HallucinationType.NEAR_MISS_TITLE.value,
    HallucinationType.CHIMERIC_TITLE.value,
}
# Defect lives in the AUTHOR field -> flip requires author-SET equality.
_AUTHOR_DEFECT_TYPES = {
    HallucinationType.AUTHOR_MISMATCH.value,  # "swapped_authors"
    HallucinationType.PARTIAL_AUTHOR_LIST.value,
    HallucinationType.MERGED_CITATION.value,  # authors from a different paper
    HallucinationType.PLACEHOLDER_AUTHORS.value,
}
# Defect lives in the VENUE field -> flip requires venue consistency.
_VENUE_DEFECT_TYPES = {
    HallucinationType.WRONG_VENUE.value,
    HallucinationType.NONEXISTENT_VENUE.value,
    HallucinationType.PREPRINT_AS_PUBLISHED.value,
}
# Defect lives in the DOI field -> flip requires the entry's OWN DOI to resolve
# to the matched paper.
_DOI_DEFECT_TYPES = {
    HallucinationType.FABRICATED_DOI.value,
    HallucinationType.HYBRID_FABRICATION.value,
}
# Year/version is the defect -> never flips (stays HALLUCINATED, retype only).
_YEAR_DEFECT_TYPES = {
    HallucinationType.FUTURE_DATE.value,
    HallucinationType.ARXIV_VERSION_MISMATCH.value,
}


def _title_exact(entry_title: str, real_title: str) -> bool:
    return bool(entry_title) and normalize_title(entry_title) == normalize_title(real_title)


def decide(entry: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    """Apply the deterministic, defect-dimension-gated decision rules.

    Core invariant: an entry flips to VALID ONLY if it is a FULLY-CORRECT
    citation — it matches a real paper on EVERY field, INCLUDING the one its
    ``hallucination_type`` marks defective. Provenance narrows scope: entries
    generated by deliberate perturbation/adversarial corruption are HALLUCINATED
    by construction and never flip (the only exception is a verified NO-OP where
    the corruption left the defect field unchanged from the real paper).
    """
    fields = entry.get("fields", {})
    title = fields.get("title", "")
    author = fields.get("author", "")
    entry_venue = venue_of(fields)
    entry_year = parse_year(fields.get("year", ""))
    entry_doi = (fields.get("doi") or "").strip()
    old_type = entry.get("hallucination_type")
    gen_method = entry.get("generation_method")
    deliberate = gen_method in DELIBERATE_DEFECT_METHODS
    # SCOPING (fix): non-flip-eligible entries are resolved CACHE-ONLY and must
    # never become UNRESOLVED for lack of a live call — they are HALLUCINATED by
    # construction. A cache miss for such an entry => KEEP its built-in type.
    scoped_cache_only = resolution.get("live_scoped") is False

    best = resolution["best"]
    best_sim = resolution["best_sim"]

    decision: dict[str, Any] = {
        "bibtex_key": entry.get("bibtex_key"),
        "old_label": entry.get("label"),
        "old_type": old_type,
        "generation_method": gen_method,
        "decision": None,
        "new_label": None,
        "new_type": None,
        "match_scores": {
            "title_sim": best_sim,
            "title_exact": None,
            "author_set_equal": None,
            "author_overlap": None,
            "venue_consistent": None,
            "year_delta": None,
            "own_doi_resolves": resolution.get("own_doi_resolves"),
        },
        "sources": sorted({c["source"] for c in resolution["candidates"]}),
        "matched_source": best["source"] if best else None,
        "matched_id": best["id"] if best else None,
        "matched_title": best["title"] if best else None,
        "authoritative_responded": resolution.get("authoritative_responded", False),
        "evidence": "",
        "low_confidence": False,
    }

    def keep(reason: str, new_type: str | None = None) -> dict[str, Any]:
        decision["decision"] = "keep"
        decision["new_label"] = "HALLUCINATED"
        # Keep the entry's existing (correct-by-construction) type unless we have
        # positive evidence to retype. Never collapse to plausible_fabrication
        # for a type whose defect we could not disprove.
        decision["new_type"] = new_type if new_type is not None else old_type
        decision["evidence"] = reason
        return decision

    def retype(new_type: str, reason: str, low_conf: bool = False) -> dict[str, Any]:
        decision["decision"] = "retype"
        decision["new_label"] = "HALLUCINATED"
        decision["new_type"] = new_type
        decision["evidence"] = reason
        decision["low_confidence"] = low_conf
        return decision

    def unresolved(reason: str) -> dict[str, Any]:
        decision["decision"] = "unresolved"
        decision["evidence"] = reason
        decision["low_confidence"] = True
        return decision

    # -- No usable candidate from any source --------------------------------
    if best is None:
        # SCOPING: a non-flip-eligible (deliberate-defect) entry resolved
        # cache-only with no cached candidate is HALLUCINATED by construction —
        # KEEP its built-in type; do NOT mark it UNRESOLVED (it never needed a
        # live call and never flips).
        if scoped_cache_only:
            return keep(
                f"Scoped-out deliberate {old_type} (cache-only, no cached "
                f"record); HALLUCINATED by construction — KEEP. (scoping)"
            )
        # Could not look (transient on the authoritative sources) -> UNRESOLVED.
        if resolution.get("errored") or resolution.get("authoritative_errored"):
            return unresolved(
                "No usable record; a transient error hit arXiv/OpenAlex/DataCite "
                "(could not verify) — flag for manual review. (fix a/d)"
            )
        # Sources responded definitively, nothing matched the title.
        if not resolution.get("authoritative_responded"):
            # CrossRef-only negative is not trustworthy for a 'fabrication' verdict.
            return unresolved(
                "No candidate, and arXiv+OpenAlex+DataCite did not respond "
                "(CrossRef-only) — cannot assert fabrication. (fix d)"
            )
        # Deliberate fabrications stay as-is; only an llm/real_world entry with
        # no real paper anywhere is newly confirmed plausible_fabrication.
        if old_type == HallucinationType.PLAUSIBLE_FABRICATION.value or deliberate:
            return keep(
                f"No real paper matches title on authoritative sources "
                f"(checked: {decision['sources'] or 'none'}); fabrication confirmed."
            )
        return keep(
            f"No real paper matches title (checked: {decision['sources'] or 'none'}); "
            f"confirmed fabrication.",
            new_type=HallucinationType.PLAUSIBLE_FABRICATION.value,
        )

    # -- We have a best candidate -------------------------------------------
    real_authors = best.get("authors", [])
    real_venue = best.get("venue", "")
    real_year, year_src = authoritative_year(entry_doi, resolution["candidates"], best)
    a_overlap = author_overlap(author, real_authors)
    a_set_eq = author_set_equal(author, real_authors)
    # BUG-1 fix: faithful-truncation aware author classification (diacritic/LaTeX
    # normalized, ordered-prefix aware). ``a_faithful`` — not the strict set-equal
    # — is the gate for clearing an author defect.
    author_class = classify_authors(author, real_authors)
    a_faithful = author_class["verdict"] == "faithful"
    a_partial_no_marker = author_class["verdict"] == "partial_no_marker"
    t_exact = _title_exact(title, best["title"])
    v_consistent = venue_consistent(entry_venue, real_venue)
    # BUG-2 / venue false-negative fix: only a TRUE venue (OpenAlex/CrossRef
    # proceedings string) can refute the entry's venue. An arXiv-DOI preprint's
    # publisher="arXiv" is not a venue authority -> venue is UNKNOWN, never wrong.
    venue_authoritative = venue_is_authoritative(best)
    venue_refuted = venue_authoritative and bool(entry_venue) and not v_consistent
    year_delta = (
        (entry_year - int(real_year))
        if (entry_year is not None and real_year is not None)
        else None
    )
    year_ok = year_delta is None or abs(year_delta) <= YEAR_TOL

    decision["match_scores"].update(
        {
            "title_exact": t_exact,
            "author_set_equal": a_set_eq,
            "author_faithful": a_faithful,
            "author_verdict": author_class["verdict"],
            "author_subset_ratio": author_class["subset_ratio"],
            "author_prefix_ratio": author_class["prefix_ratio"],
            "author_contig_run": author_class.get("contig_run"),
            "author_gcontig": author_class.get("gcontig"),
            "author_lead_ok": author_class.get("lead_ok"),
            "author_lead_given_conflict": author_class.get("lead_given_conflict"),
            "author_truncated": author_class["truncated"],
            "author_overlap": round(a_overlap, 4),
            "venue_consistent": v_consistent,
            "venue_authoritative": venue_authoritative,
            "venue_refuted": venue_refuted,
            "year_source": year_src,
            "year_delta": year_delta,
        }
    )

    # Same real paper at all? Title must clear EXACT (the strict bar); a fuzzy
    # match in [TITLE_RETYPE_MIN, 1) means "same paper, but title corrupted".
    title_same_paper = t_exact or best_sim >= TITLE_RETYPE_MIN

    if not title_same_paper:
        # Best candidate isn't even the same paper -> genuine fabrication only
        # for the non-deliberate cohort; deliberate stays as its built type.
        if scoped_cache_only:
            # Deliberate, cache-only: KEEP its built type (never unresolved).
            return keep(
                f"Scoped-out deliberate {old_type} (cache-only); best title_sim="
                f"{best_sim:.3f} — HALLUCINATED by construction, KEEP. (scoping)"
            )
        if not resolution.get("authoritative_responded"):
            return unresolved(
                f"Best title_sim={best_sim:.3f} via CrossRef-only "
                f"(arXiv/OpenAlex silent) — cannot trust. (fix d)"
            )
        if deliberate:
            return keep(
                f"Best title_sim={best_sim:.3f} < {TITLE_RETYPE_MIN}; deliberate "
                f"{old_type} — no real paper reproduced, stays HALLUCINATED."
            )
        return keep(
            f"Best title_sim={best_sim:.3f} < {TITLE_RETYPE_MIN} "
            f"(closest '{best['title'][:60]}' via {best['source']}); fabrication.",
            new_type=HallucinationType.PLAUSIBLE_FABRICATION.value,
        )

    # From here: the entry references a real paper (same-paper title match).
    # Compute whether EVERY field — including the defect dimension — checks out.
    # ---------------------------------------------------------------------- #
    # DELIBERATE-DEFECT PROVENANCE (fix c): perturbation/adversarial entries are
    # HALLUCINATED by construction. They flip ONLY if the corruption was a
    # verified NO-OP — i.e. the defect field still equals the real paper's. That
    # is rare; the common case keeps the original (correct) type.
    # ---------------------------------------------------------------------- #
    if deliberate:
        # Check whether the defect field actually differs from the real paper.
        if old_type in _TITLE_DEFECT_TYPES:
            if t_exact:  # corruption produced the identical title -> NO-OP
                pass  # fall through to full-correctness check below
            else:
                return keep(
                    f"Deliberate {old_type}: entry title differs from real "
                    f"'{best['title'][:50]}' (sim={best_sim:.3f}, exact={t_exact}). "
                    f"The title edit IS the defect. (fix b)"
                )
        elif old_type in _AUTHOR_DEFECT_TYPES:
            # A deliberate author-defect perturbation (swapped/partial/placeholder)
            # altered the author field BY CONSTRUCTION — it is HALLUCINATED and
            # NEVER flips. We do not trust the faithfulness heuristic to declare a
            # provable corruption a NO-OP (e.g. a 2-author paper dropped to its
            # first author can look like a faithful 1-author prefix). Always KEEP.
            return keep(
                f"Deliberate {old_type}: author field corrupted by construction "
                f"(verdict={author_class['verdict']}, subset={author_class['subset_ratio']}). "
                f"The author change IS the defect — stays HALLUCINATED. (fix b/c)"
            )
        elif old_type in _VENUE_DEFECT_TYPES:
            # A deliberate venue defect flips ONLY if we can POSITIVELY confirm
            # the entry's venue matches a TRUE (authoritative) venue. If venue is
            # refuted OR the only venue we have is the non-authoritative arXiv-DOI
            # publisher (cannot confirm), KEEP — never flip on an unverifiable
            # venue. (Guards a wrong_venue perturbation with an arXiv-DOI record.)
            if venue_refuted or not (venue_authoritative and v_consistent):
                return keep(
                    f"Deliberate {old_type}: entry venue '{entry_venue[:40]}' "
                    f"not positively confirmed against a true venue "
                    f"(authoritative={venue_authoritative}, consistent={v_consistent}). "
                    f"Venue defect not disproven — KEEP. (fix b)"
                )
        elif old_type in _DOI_DEFECT_TYPES:
            if resolution.get("own_doi_resolves") is not True:
                return keep(
                    f"Deliberate {old_type}: entry DOI '{entry_doi}' did not "
                    f"resolve to the matched paper (own_doi_resolves="
                    f"{resolution.get('own_doi_resolves')}). DOI IS the defect. (fix b)"
                )
        elif old_type in _YEAR_DEFECT_TYPES:
            # Year/version defect never flips.
            return keep(
                f"Deliberate {old_type}: year/version is the defect "
                f"(entry={entry_year}, real={real_year}, delta={year_delta}); "
                f"stays HALLUCINATED. (fix b)"
            )
        # else: any other deliberate type with title-match but unknown defect
        # field — fall through to the full-correctness gate (must clear all).

    # ---------------------------------------------------------------------- #
    # FULL-CORRECTNESS FLIP GATE (applies to llm_generated/real_world AND to a
    # deliberate NO-OP that fell through): require EXACT title + FAITHFUL authors
    # (incl. truncation rule, BUG-1) + venue not-refuted (BUG-2: an arXiv-DOI
    # publisher is not a venue authority) + year-ok + (DOI resolves if present).
    # ---------------------------------------------------------------------- #
    fail: list[str] = []
    if not t_exact:
        fail.append(f"title not exact (sim={best_sim:.3f})")
    if not a_faithful:
        fail.append(f"authors not faithful (verdict={author_class['verdict']})")
    # Venue: only a TRUE (authoritative) venue can fail the gate. An arXiv-DOI
    # preprint's publisher="arXiv" cannot refute the entry's venue (BUG-2 fix).
    if venue_refuted:
        fail.append(f"venue '{entry_venue[:30]}' != real '{real_venue[:30]}'")
    if not year_ok:
        fail.append(f"year_delta={year_delta} (>{YEAR_TOL})")
    # DOI: if the entry carries a non-arXiv DOI, it must resolve to this paper.
    if entry_doi and not resolution.get("arxiv_doi_id"):
        if resolution.get("own_doi_resolves") is False:
            fail.append(f"DOI {entry_doi} does not resolve")
        elif resolution.get("own_doi_resolves") is None:
            fail.append(f"DOI {entry_doi} resolution unknown")

    if not fail:
        # FULLY CORRECT -> flip to VALID. Require authoritative corroboration.
        if not resolution.get("authoritative_responded"):
            if scoped_cache_only:
                # Deliberate NO-OP candidate but only non-authoritative cache —
                # do not flip a by-construction hallucination; KEEP its type.
                return keep(
                    f"Scoped-out deliberate {old_type}: fields appear correct but "
                    f"only non-authoritative cache responded — KEEP. (scoping)"
                )
            return unresolved(
                "All fields match but only CrossRef responded (arXiv/OpenAlex "
                "silent) — withhold flip pending authoritative confirmation. (fix d)"
            )
        decision["decision"] = "flip_valid"
        decision["new_label"] = "VALID"
        decision["new_type"] = None
        decision["evidence"] = (
            f"FULLY-CORRECT citation: title exact, authors faithful "
            f"(verdict={author_class['verdict']}, subset={author_class['subset_ratio']}, "
            f"truncated={author_class['truncated']}), venue not refuted, "
            f"year_delta={year_delta} via {best['source']} ({best['id']}). (fix b/c)"
        )
        return decision

    # Same real paper, but some field is wrong -> RE-TYPE to the TRUE defect.
    # BUG-2 branch order: assign the type to the field that ACTUALLY fails. Title
    # first (the strongest signal), then the author dimension (distinguishing a
    # marker-less partial list from a genuine swap), then venue, then year, then
    # DOI. Crucially: when authors ARE faithful, NEVER fall through to
    # swapped_authors — the defect lives in venue/year/DOI.
    if not t_exact:
        return retype(
            HallucinationType.NEAR_MISS_TITLE.value
            if best_sim >= 0.85
            else HallucinationType.CHIMERIC_TITLE.value,
            f"Same paper but title differs (sim={best_sim:.3f}): "
            f"'{title[:40]}' vs '{best['title'][:40]}'. {', '.join(fail)}.",
        )
    if not a_faithful:
        new_author_type = (
            HallucinationType.PARTIAL_AUTHOR_LIST.value
            if a_partial_no_marker
            else HallucinationType.AUTHOR_MISMATCH.value
        )
        return retype(
            new_author_type,
            f"Same paper (title exact) but authors "
            f"{'are a marker-less partial list' if a_partial_no_marker else 'differ'} "
            f"(verdict={author_class['verdict']}, subset={author_class['subset_ratio']}, "
            f"prefix={author_class['prefix_ratio']}): entry='{author[:50]}' vs "
            f"real={real_authors[:4]} via {best['source']}.",
        )
    if venue_refuted:
        return retype(
            HallucinationType.WRONG_VENUE.value,
            f"Same paper (title+authors match) but venue '{entry_venue[:40]}' "
            f"inconsistent with real '{real_venue[:40]}' via {best['source']}.",
        )
    if not year_ok:
        if entry_year is not None and entry_year > CURRENT_YEAR:
            return retype(
                HallucinationType.FUTURE_DATE.value,
                f"Same paper, authors match, entry year {entry_year} is in the "
                f"future (real={real_year}).",
            )
        return retype(
            HallucinationType.ARXIV_VERSION_MISMATCH.value,
            f"Same paper, authors match, year shifted (entry={entry_year}, "
            f"real={real_year} [{year_src}], delta={year_delta}).",
            low_conf=True,
        )
    # Remaining failure is a DOI defect.
    return retype(
        HallucinationType.FABRICATED_DOI.value
        if resolution.get("own_doi_resolves") is False
        else (old_type or HallucinationType.HYBRID_FABRICATION.value),
        f"Same paper, title+authors+venue match, but DOI {entry_doi} "
        f"{'does not resolve' if resolution.get('own_doi_resolves') is False else 'unverified'}.",
    )


# ---------------------------------------------------------------------------
# VALID-entry schema cleanup (the leftover from the prior 32 flips)
# ---------------------------------------------------------------------------


def valid_needs_cleanup(entry: dict[str, Any]) -> bool:
    """True if a VALID entry carries the stale-hallucination inconsistency.

    Targets ONLY entries that either (a) were relabeled by a prior pass, or
    (b) violate the schema's VALID convention by retaining a hallucination_type
    or difficulty_tier. Benign VALID entries with a single legitimately-False
    sub-test (cross_db_agreement / fields_complete / doi_resolves-without-DOI)
    are NOT touched.
    """
    if entry.get("label") != "VALID":
        return False
    if entry.get("hallucination_type") is not None:
        return True
    if entry.get("difficulty_tier") is not None:
        return True
    if entry.get("relabeled_by") or entry.get("relabeled_from"):
        st = entry.get("subtests", {})
        # Stale only if a core "real-paper" sub-test is still False.
        if any(st.get(k) is False for k in ("title_exists", "authors_match")):
            return True
    return False


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def purge_transient_cache(cache: dict[str, Any]) -> int:
    """Drop any cache entry whose stored ``error`` is TRANSIENT (429/5xx/timeout/
    DNS/SSL/reset). Returns the number purged.

    A transient outcome is "could not look", NOT a verdict — caching it would
    poison a warm re-run (a real paper looks like a fabrication because its
    lookup 429'd once). The fetch path already refuses to cache transients, but
    historical caches can still hold ~hundreds of residual 429/5xx bodies, so we
    purge defensively on load (and via ``--purge-transient``). A successful body
    or a DEFINITIVE not-found (HTTP 404) is kept.
    """
    to_drop = [
        k
        for k, v in cache.items()
        if isinstance(v, dict) and v.get("body") is None and Resolver._is_transient(v.get("error"))
    ]
    for k in to_drop:
        del cache[k]
    return len(to_drop)


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return loaded
        except json.JSONDecodeError:
            print(f"warning: cache at {path} is corrupt; starting fresh", file=sys.stderr)
    return {}


def save_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def apply_decision_to_record(
    rec: dict[str, Any], decision: dict[str, Any], resolution: dict[str, Any]
) -> None:
    """Mutate a raw JSON record in place per the decision (for --apply)."""
    fields = rec.get("fields", {})
    best = resolution.get("best")
    prov = {
        "relabeled_from": rec.get("label"),
        "relabel_reason": decision["evidence"],
        "relabeled_by": RELABELED_BY,
        "matched_source": decision["matched_source"],
        "matched_id": decision["matched_id"],
        "match_scores": decision["match_scores"],
    }
    if decision["decision"] == "flip_valid":
        rec["label"] = "VALID"
        rec.pop("hallucination_type", None)
        rec.pop("difficulty_tier", None)
        rec["subtests"] = subtests_for_valid(fields, best or {})
        rec.update(prov)
    elif decision["decision"] == "retype":
        rec["label"] = "HALLUCINATED"
        rec["hallucination_type"] = decision["new_type"]
        rec["difficulty_tier"] = HALLUCINATION_TIER_MAP[
            HallucinationType(decision["new_type"])
        ].value
        rec["subtests"] = subtests_for_type(decision["new_type"], fields)
        rec.update(prov)
    elif decision["decision"] == "keep":
        rec["label"] = "HALLUCINATED"
        new_type = decision.get("new_type") or rec.get("hallucination_type")
        old_rec_type = rec.get("hallucination_type")
        rec["hallucination_type"] = new_type
        if new_type is not None:
            rec["difficulty_tier"] = HALLUCINATION_TIER_MAP[HallucinationType(new_type)].value
        # Only rewrite subtests when the type actually changed (e.g. a newly
        # confirmed plausible_fabrication). When we KEEP the original
        # by-construction type, its subtests are already correct — leave them.
        if new_type != old_rec_type and isinstance(new_type, str):
            if new_type == HallucinationType.PLAUSIBLE_FABRICATION.value:
                rec["subtests"] = subtests_for_fabrication(fields)
            else:
                rec["subtests"] = subtests_for_type(new_type, fields)
            rec.update(prov)
    # unresolved -> no mutation


def cleanup_valid_record(rec: dict[str, Any]) -> None:
    """Mutate a VALID record in place: drop stale htype/tier, fix sub-tests."""
    rec.pop("hallucination_type", None)
    rec.pop("difficulty_tier", None)
    st = dict(rec.get("subtests", {}))
    # A relabeled-to-VALID entry's real-paper sub-tests must be True; preserve
    # genuinely-None doi_resolves only when the entry truly has no DOI.
    doi = (rec.get("fields", {}).get("doi") or "").strip()
    st["title_exists"] = True
    st["authors_match"] = True
    st["venue_correct"] = True
    st["cross_db_agreement"] = True
    if not doi:
        st["doi_resolves"] = None
    elif ARXIV_DOI_RE.match(doi):
        st["doi_resolves"] = True
    else:
        st.setdefault("doi_resolves", None)
    fields = rec.get("fields", {})
    st["fields_complete"] = bool(
        fields.get("title") and fields.get("author") and fields.get("year")
    )
    rec["subtests"] = st


class _SplitContext:
    """Per-split shared state for parallel resolution workers."""

    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        split_name: str,
        n_hall: int,
        resolver: Resolver,
        cache: dict[str, Any],
    ) -> None:
        self.records = records
        self.split_name = split_name
        self.n_hall = n_hall
        self.resolver = resolver
        self.cache = cache
        self.done = 0
        self.progress_lock = threading.Lock()


def _resolve_one(idx: int, ctx: _SplitContext) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Resolve+decide one HALLUCINATED record. Thread-safe; verdict is a pure
    function of the record's own URL fetches (worker-count independent)."""
    rec = ctx.records[idx]
    # SCOPING: deliberate-defect entries (perturbation/adversarial, not
    # plausible_fabrication) are resolved CACHE-ONLY — they never flip, so we
    # spend no live API call on them. Flip-eligible entries get full (cache +
    # live) resolution.
    live = needs_live_resolution(rec)
    with ctx.resolver.cache_only(not live):
        resolution = resolve_entry(rec, ctx.resolver)
    resolution["live_scoped"] = live
    decision = decide(rec, resolution)
    decision["live_scoped"] = live
    decision["split"] = ctx.split_name
    decision["title"] = rec.get("fields", {}).get("title", "")
    with ctx.progress_lock:
        ctx.done += 1
        done = ctx.done
    if done % 25 == 0:
        print(
            f"  [{ctx.split_name}] {done}/{ctx.n_hall} hallucinated "
            f"(net calls so far: {ctx.resolver.network_calls})",
            file=sys.stderr,
        )
        save_cache(ctx.cache, CACHE_PATH)  # periodic checkpoint
    return idx, decision, resolution


def run(
    *,
    apply: bool,
    rate_limit: float,
    allow_network: bool,
    timeout: int,
    workers: int = 1,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(CACHE_PATH)
    # PURGE residual transient (429/5xx/timeout) cache entries BEFORE running, so
    # no "could not look" outcome is ever read back as a verdict (cache-poisoning
    # guard). The fetch path no longer caches transients, but historical caches
    # can still hold them.
    n_purged = purge_transient_cache(cache)
    if n_purged:
        print(f"Purged {n_purged} residual transient cache entries before run.", file=sys.stderr)
        save_cache(cache, CACHE_PATH)
    resolver = Resolver(cache, rate_limit=rate_limit, timeout=timeout, allow_network=allow_network)

    proposal: dict[str, Any] = {
        "generated": date.today().isoformat(),
        "relabeled_by": RELABELED_BY,
        "cache_transient_purged": n_purged,
        "thresholds": {
            "title_exact": TITLE_EXACT,
            "title_retype_min": TITLE_RETYPE_MIN,
            "author_set_tol": AUTHOR_SET_TOL,
            "author_subset_faithful": AUTHOR_SUBSET_FAITHFUL,
            "author_prefix_faithful": AUTHOR_PREFIX_FAITHFUL,
            "venue_sim": VENUE_SIM,
            "year_tol": YEAR_TOL,
            "deliberate_defect_methods": sorted(DELIBERATE_DEFECT_METHODS),
            "flip_eligible_methods": sorted(FLIP_ELIGIBLE_METHODS),
            "source_min_interval": SOURCE_MIN_INTERVAL,
            "max_retries": MAX_RETRIES,
            "min_live_coverage": MIN_LIVE_COVERAGE,
            "max_unresolved_fraction": MAX_UNRESOLVED_FRACTION,
        },
        "mode": "apply" if apply else "dry-run",
        "splits": {},
        "entries": [],
        "valid_cleanups": [],
        "source_coverage": {},
    }

    for split_name, path in SPLITS.items():
        if not path.exists():
            print(f"warning: split {split_name} not found at {path}; skipping", file=sys.stderr)
            continue
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        counts = {"flip_valid": 0, "retype": 0, "keep": 0, "unresolved": 0, "valid_cleanup": 0}
        n_hall = sum(1 for r in records if r.get("label") == "HALLUCINATED")
        print(
            f"[{split_name}] {len(records)} entries ({n_hall} HALLUCINATED) ...",
            file=sys.stderr,
        )

        # ---- VALID-entry cleanup (cheap, no network) -------------------- #
        for rec in records:
            if rec.get("label") == "VALID" and valid_needs_cleanup(rec):
                counts["valid_cleanup"] += 1
                proposal["valid_cleanups"].append(
                    {
                        "split": split_name,
                        "bibtex_key": rec.get("bibtex_key"),
                        "title": rec.get("fields", {}).get("title", ""),
                        "had_hallucination_type": rec.get("hallucination_type"),
                        "had_difficulty_tier": rec.get("difficulty_tier"),
                        "old_subtests": dict(rec.get("subtests", {})),
                    }
                )
                if apply:
                    cleanup_valid_record(rec)

        # ---- HALLUCINATED re-verification (network-bound) --------------- #
        # Resolve+decide in parallel for throughput; the result for each entry
        # is a pure function of its own URL fetches, so the verdict is identical
        # regardless of worker count. We collect into a dict keyed by the record
        # index and then iterate IN RECORD ORDER so the proposal is byte-stable.
        hall_indices = [i for i, r in enumerate(records) if r.get("label") == "HALLUCINATED"]
        ctx = _SplitContext(
            records=records,
            split_name=split_name,
            n_hall=n_hall,
            resolver=resolver,
            cache=cache,
        )
        results_by_idx: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        if workers > 1 and hall_indices:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for idx, decision, resolution in ex.map(
                    lambda i, _ctx=ctx: _resolve_one(i, _ctx), hall_indices
                ):
                    results_by_idx[idx] = (decision, resolution)
        else:
            for idx in hall_indices:
                _i, decision, resolution = _resolve_one(idx, ctx)
                results_by_idx[idx] = (decision, resolution)

        # Aggregate deterministically in record order.
        for idx in hall_indices:
            decision, resolution = results_by_idx[idx]
            counts[decision["decision"]] += 1
            proposal["entries"].append(decision)
            if apply:
                apply_decision_to_record(records[idx], decision, resolution)

        proposal["splits"][split_name] = counts

        if apply:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
            print(f"[{split_name}] WROTE {path}", file=sys.stderr)

    # ---- Per-source LIVE coverage log + fail-loud guard (fix a) ----------- #
    # Only LIVE calls this run count (cache hits are not in source_live). On a
    # fully warm re-run there are zero live calls -> coverage check is skipped.
    coverage: dict[str, Any] = {"live_calls": resolver.network_calls}
    for src, s in sorted(resolver.source_live.items()):
        total = s["ok"] + s["transient"] + s["notfound"]
        coverage[src] = {
            **s,
            "total": total,
            "success_rate": round(s["ok"] / total, 4) if total else None,
            "transient_rate": round(s["transient"] / total, 4) if total else None,
        }
    proposal["source_coverage"] = coverage
    print("\n=== PER-SOURCE LIVE COVERAGE (this run) ===", file=sys.stderr)
    for src, c in coverage.items():
        if src == "live_calls":
            continue
        print(
            f"  {src:10s} ok={c['ok']:4d} transient={c['transient']:4d} "
            f"notfound={c['notfound']:4d} success_rate={c['success_rate']}",
            file=sys.stderr,
        )

    # Fail loudly if the AUTHORITATIVE sources were largely unreachable on the
    # re-resolved (live) set — keeps would degenerate to CrossRef-only (the bug).
    auth_ok = sum(resolver.source_live.get(s, {}).get("ok", 0) for s in ("arxiv", "openalex"))
    auth_total = sum(sum(resolver.source_live.get(s, {}).values()) for s in ("arxiv", "openalex"))
    if auth_total > 0:
        auth_cov = auth_ok / auth_total
        coverage["arxiv_openalex_success_rate"] = round(auth_cov, 4)
        if auth_cov < MIN_LIVE_COVERAGE:
            save_cache(cache, CACHE_PATH)
            PROPOSAL_PATH.write_text(
                json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise SystemExit(
                f"FATAL (fix a): arXiv+OpenAlex live success rate {auth_cov:.1%} "
                f"< {MIN_LIVE_COVERAGE:.0%} on {auth_total} live calls. Keeps would "
                f"be CrossRef-only — the very bug being fixed. Re-run later with a "
                f"higher --rate-limit; cache + partial proposal saved."
            )

    # ---- Unresolved-fraction guard (EXECUTION-HOLE fix) ------------------- #
    # The live-coverage guard above is gated on auth_total>0, so a zero-live warm
    # run that punts everything to UNRESOLVED would pass it silently. This guard
    # ALSO fails — regardless of whether any live call ran — when too many of the
    # decided entries could not be verified. With the SCOPING strategy, deliberate
    # defects are KEPT (not unresolved), so a healthy run sits well below the
    # ceiling; a high fraction means the cache was cold/poisoned and the proposal
    # is untrustworthy. On --apply this BLOCKS the write; on a dry run it FAILS
    # loudly after persisting the (diagnostic) partial proposal.
    total_unresolved = sum(c.get("unresolved", 0) for c in proposal["splits"].values())
    total_decided = sum(
        c.get("flip_valid", 0) + c.get("retype", 0) + c.get("keep", 0) + c.get("unresolved", 0)
        for c in proposal["splits"].values()
    )
    unresolved_fraction = (total_unresolved / total_decided) if total_decided else 0.0
    coverage["total_decided"] = total_decided
    coverage["total_unresolved"] = total_unresolved
    coverage["unresolved_fraction"] = round(unresolved_fraction, 4)
    print(
        f"\n=== UNRESOLVED FRACTION: {total_unresolved}/{total_decided} = "
        f"{unresolved_fraction:.1%} (ceiling {MAX_UNRESOLVED_FRACTION:.0%}) ===",
        file=sys.stderr,
    )
    if total_decided > 0 and unresolved_fraction > MAX_UNRESOLVED_FRACTION:
        save_cache(cache, CACHE_PATH)
        PROPOSAL_PATH.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"FATAL (execution-hole fix): unresolved fraction {unresolved_fraction:.1%} "
            f"> {MAX_UNRESOLVED_FRACTION:.0%} ({total_unresolved}/{total_decided} entries). "
            f"Too many entries could not be verified — the cache is likely cold or "
            f"poisoned and this proposal must not be applied. Warm the cache (re-run "
            f"with network for the flip-eligible cohort) and retry; partial proposal saved."
        )

    save_cache(cache, CACHE_PATH)
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nProposal written to {PROPOSAL_PATH}", file=sys.stderr)
    print(
        f"Cache written to {CACHE_PATH} (network calls this run: {resolver.network_calls})",
        file=sys.stderr,
    )
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Stage 2: write the data files. Default is a dry run (proposal only).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry run (default behavior; writes only the proposal + cache).",
    )
    ap.add_argument(
        "--no-network",
        action="store_true",
        help="Use the resolution cache only; never touch the network.",
    )
    ap.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Minimum seconds between network requests (default: 1.0).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds (default: 30).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel resolution workers (default: 1). Decisions are "
            "deterministic and identical regardless of worker count; only the "
            "order of network calls changes. The global rate-limit is still "
            "honored across all workers."
        ),
    )
    args = ap.parse_args()

    apply = args.apply and not args.dry_run
    proposal = run(
        apply=apply,
        rate_limit=args.rate_limit,
        allow_network=not args.no_network,
        timeout=args.timeout,
        workers=max(1, args.workers),
    )

    # Console summary
    print("\n=== SUMMARY (per split) ===")
    for split, counts in proposal["splits"].items():
        print(
            f"  {split:18s} flip_valid={counts['flip_valid']:3d} "
            f"retype={counts['retype']:3d} keep={counts['keep']:4d} "
            f"unresolved={counts['unresolved']:3d} valid_cleanup={counts['valid_cleanup']:3d}"
        )


if __name__ == "__main__":
    main()
