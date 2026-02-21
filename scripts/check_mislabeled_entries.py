#!/usr/bin/env python3
"""Check potentially mislabeled plausible_fabrication entries.  [analysis]

Background
----------
During the v1.0 data quality audit, a heuristic re-classification pass ran over
LLM-generated entries, comparing each entry's DOI and metadata against CrossRef.
The heuristic used similarity thresholds that were too aggressive: it flagged
entries as HALLUCINATED / plausible_fabrication even when the title similarity
score was in the 70-75% range (well below the 0.85 threshold used elsewhere).
A common cause is author name format mismatch (e.g., "Stiennon, Nisan" stored
as "Stiennon, N." in CrossRef), which drove the Jaccard score to 0.00 and
caused otherwise valid papers to be mislabeled.

Known examples of likely-real papers that were mislabeled:
  - "Learning to Summarize with Human Feedback" (arXiv:2009.01325)
  - "A Simple Framework for Contrastive Learning" (SimCLR, arXiv:2002.05709)
  - "Communication-Efficient Learning of Deep Networks" (FedAvg, arXiv:1602.05629)
  - "Benchmarking Graph Neural Networks" (arXiv:2003.00982)

Suspect criteria (either condition triggers)
--------------------------------------------
1. hallucination_type == "plausible_fabrication"  AND  explanation contains "[Re-classified]"
2. hallucination_type == "plausible_fabrication"  AND  generation_method == "llm_generated"
   AND DOI starts with "10.48550" (arXiv DOI prefix)

This script is READ-ONLY — it never modifies the data files.
All re-classification decisions must be made manually by the benchmark author.

Usage
-----
    python scripts/check_mislabeled_entries.py               # full check with API calls
    python scripts/check_mislabeled_entries.py --dry-run     # list suspects only
    python scripts/check_mislabeled_entries.py --rate-limit 2.0  # slower API cadence
"""

from __future__ import annotations

import argparse
import difflib
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SSL context — use certifi bundle on macOS if available
# ---------------------------------------------------------------------------
try:
    import certifi

    _SSL_CTX: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/v1.0")
SPLITS = {
    "dev": DATA_DIR / "dev_public.jsonl",
    "test": DATA_DIR / "test_public.jsonl",
}
OUTPUT_PATH = DATA_DIR / "mislabeled_review.json"
CROSSREF_BASE = "https://api.crossref.org/works"
USER_AGENT = "HALLMARK-Benchmark/1.0 (mailto:research@example.com)"
TITLE_SIM_THRESHOLD = 0.85
DEFAULT_RATE_LIMIT = 1.0  # seconds between API requests


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    t = title.lower()
    for ch in ".,;:!?\"'()[]{}/-":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio on normalized titles."""
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def extract_last_names(author_str: str) -> set[str]:
    """Return lowercased last names from a BibTeX author field.

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
    return last_names


# ---------------------------------------------------------------------------
# CrossRef API
# ---------------------------------------------------------------------------


def _get_json(url: str) -> dict[str, Any] | None:
    """Fetch URL and parse JSON.  Returns None on any error."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", USER_AGENT)
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"  [HTTP {e.code}] {url}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [ERROR] {exc}", file=sys.stderr)
        return None


def crossref_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch CrossRef metadata for a DOI.  Returns message dict or None."""
    url = f"{CROSSREF_BASE}/{urllib.parse.quote(doi, safe='')}"
    data = _get_json(url)
    if data is None:
        return None
    msg = data.get("message")
    return msg if isinstance(msg, dict) else None


def crossref_title_search(title: str) -> dict[str, Any] | None:
    """Search CrossRef by title and return the top result or None."""
    params = {"query.bibliographic": title, "rows": "1"}
    url = f"{CROSSREF_BASE}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    if data is None:
        return None
    items = data.get("message", {}).get("items", [])
    return items[0] if items else None


def crossref_title_from_message(msg: dict[str, Any]) -> str:
    """Extract title string from a CrossRef message dict."""
    titles = msg.get("title", [])
    return titles[0] if titles else ""


def crossref_last_names_from_message(msg: dict[str, Any]) -> set[str]:
    """Extract author last names from a CrossRef message dict."""
    return {a.get("family", "").lower() for a in msg.get("author", []) if "family" in a}


# ---------------------------------------------------------------------------
# Suspect detection
# ---------------------------------------------------------------------------


def is_suspect(entry: dict[str, Any]) -> bool:
    """Return True if this entry matches either suspect criterion."""
    htype = entry.get("hallucination_type", "")
    if htype != "plausible_fabrication":
        return False

    explanation = entry.get("explanation", "")
    if "[Re-classified]" in explanation:
        return True

    gen_method = str(entry.get("generation_method", ""))
    doi = str(entry.get("fields", {}).get("doi", ""))
    return gen_method == "llm_generated" and doi.startswith("10.48550")


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_entry(entry: dict[str, Any], split: str, rate_limit: float) -> dict[str, Any]:
    """Run CrossRef verification for one suspect entry.

    Returns a result dict with recommendation and evidence.
    """
    fields = entry.get("fields", {})
    title = fields.get("title", "")
    author = fields.get("author", "")
    doi = fields.get("doi", "")
    bibtex_key = entry.get("bibtex_key", "")

    result: dict[str, Any] = {
        "bibtex_key": bibtex_key,
        "split": split,
        "title": title,
        "doi": doi,
        "explanation_snippet": entry.get("explanation", "")[:120],
        "generation_method": entry.get("generation_method", ""),
        "crossref_match": "NOT_CHECKED",
        "crossref_title": "",
        "title_similarity": None,
        "authors_match": None,
        "recommendation": "NEEDS_MANUAL_REVIEW",
        "evidence": "",
    }

    time.sleep(rate_limit)

    crossref_msg: dict[str, Any] | None = None
    lookup_method = ""

    if doi:
        crossref_msg = crossref_by_doi(doi)
        lookup_method = "doi"

    # Fall back to title search if DOI lookup failed or no DOI
    if crossref_msg is None:
        crossref_msg = crossref_title_search(title)
        lookup_method = "title_search"

    if crossref_msg is None:
        result["crossref_match"] = "ERROR"
        result["recommendation"] = "NEEDS_MANUAL_REVIEW"
        result["evidence"] = "CrossRef returned no data (network error or not indexed)"
        return result

    cr_title = crossref_title_from_message(crossref_msg)
    cr_last_names = crossref_last_names_from_message(crossref_msg)

    sim = title_similarity(title, cr_title) if cr_title else 0.0
    our_last_names = extract_last_names(author)
    authors_overlap = bool(our_last_names & cr_last_names)

    result["crossref_title"] = cr_title
    result["title_similarity"] = round(sim, 4)
    result["authors_match"] = authors_overlap

    title_ok = sim >= TITLE_SIM_THRESHOLD
    match_overall = title_ok and authors_overlap

    result["crossref_match"] = "YES" if match_overall else "NO"

    if match_overall:
        result["recommendation"] = "RECLASSIFY_AS_VALID"
        result["evidence"] = (
            f"CrossRef match via {lookup_method}: title_sim={sim:.3f}>={TITLE_SIM_THRESHOLD}"
            f", authors overlap ({our_last_names & cr_last_names})"
        )
    elif title_ok and not authors_overlap:
        # Title matches but authors don't — could be format mismatch (the known bug)
        result["crossref_match"] = "PARTIAL"
        result["recommendation"] = "NEEDS_MANUAL_REVIEW"
        result["evidence"] = (
            f"Title match (sim={sim:.3f}) but no author overlap. "
            f"Our last names: {our_last_names}. "
            f"CrossRef last names: {cr_last_names}. "
            f"Likely author format mismatch — check manually."
        )
    else:
        result["recommendation"] = "KEEP_AS_HALLUCINATED"
        reasons = []
        if not title_ok:
            reasons.append(f"title_sim={sim:.3f}<{TITLE_SIM_THRESHOLD}")
        if not authors_overlap:
            reasons.append("no author overlap")
        result["evidence"] = (
            f"CrossRef lookup via {lookup_method} failed: {'; '.join(reasons)}. "
            f"CrossRef title: '{cr_title[:80]}'"
        )

    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_RECS = {
    "RECLASSIFY_AS_VALID": "RECLASSIFY",
    "KEEP_AS_HALLUCINATED": "KEEP",
    "NEEDS_MANUAL_REVIEW": "MANUAL",
    "NOT_CHECKED": "SKIP",
}


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a compact table of results to stdout."""
    # Column widths
    KEY_W = 14
    TITLE_W = 40
    DOI_W = 28
    SIM_W = 6
    MATCH_W = 9
    REC_W = 10

    header = (
        f"{'bibtex_key':<{KEY_W}}  "
        f"{'split':<5}  "
        f"{'title':<{TITLE_W}}  "
        f"{'doi':<{DOI_W}}  "
        f"{'sim':>{SIM_W}}  "
        f"{'match':<{MATCH_W}}  "
        f"{'rec':<{REC_W}}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in results:
        sim_str = f"{r['title_similarity']:.3f}" if r["title_similarity"] is not None else "  n/a"
        match_str = r.get("crossref_match", "?")
        rec_short = _RECS.get(r["recommendation"], r["recommendation"])
        print(
            f"{r['bibtex_key'][:KEY_W]:<{KEY_W}}  "
            f"{r['split']:<5}  "
            f"{r['title'][:TITLE_W]:<{TITLE_W}}  "
            f"{r['doi'][:DOI_W]:<{DOI_W}}  "
            f"{sim_str:>{SIM_W}}  "
            f"{match_str:<{MATCH_W}}  "
            f"{rec_short:<{REC_W}}"
        )

    print(sep)

    # Summary counts
    counts: dict[str, int] = {}
    for r in results:
        counts[r["recommendation"]] = counts.get(r["recommendation"], 0) + 1

    print()
    print("Summary:")
    for rec, label in [
        ("RECLASSIFY_AS_VALID", "Reclassify as VALID"),
        ("NEEDS_MANUAL_REVIEW", "Needs manual review"),
        ("KEEP_AS_HALLUCINATED", "Keep as HALLUCINATED"),
        ("NOT_CHECKED", "Not checked (dry-run)"),
    ]:
        n = counts.get(rec, 0)
        if n:
            print(f"  {label}: {n}")
    print(f"  Total suspects: {len(results)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate potentially mislabeled plausible_fabrication entries "
            "in the HALLMARK benchmark against the CrossRef API."
        )
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=SPLITS["dev"],
        help=f"Dev split path (default: {SPLITS['dev']})",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=SPLITS["test"],
        help=f"Test split path (default: {SPLITS['test']})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output JSON report path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        metavar="SECS",
        help=f"Seconds between CrossRef API requests (default: {DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List suspect entries without making any API calls",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load suspects
    # -----------------------------------------------------------------------
    suspects: list[tuple[dict[str, Any], str]] = []  # (entry, split_name)

    for split_name, path in [("dev", args.dev), ("test", args.test)]:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                if is_suspect(entry):
                    suspects.append((entry, split_name))

    print(f"Found {len(suspects)} suspect entries", file=sys.stderr)
    if not suspects:
        print("Nothing to review.", file=sys.stderr)
        return

    # -----------------------------------------------------------------------
    # Dry-run: list only
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("\n[Dry-run] Suspect entries (no API calls made):\n", file=sys.stderr)
        dry_results = []
        for entry, split in suspects:
            fields = entry.get("fields", {})
            dry_results.append(
                {
                    "bibtex_key": entry.get("bibtex_key", ""),
                    "split": split,
                    "title": fields.get("title", ""),
                    "doi": fields.get("doi", ""),
                    "explanation_snippet": entry.get("explanation", "")[:120],
                    "generation_method": entry.get("generation_method", ""),
                    "crossref_match": "NOT_CHECKED",
                    "crossref_title": "",
                    "title_similarity": None,
                    "authors_match": None,
                    "recommendation": "NOT_CHECKED",
                    "evidence": "dry-run",
                }
            )
        print_report(dry_results)

        output_data = {
            "dry_run": True,
            "total_suspects": len(dry_results),
            "results": dry_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2, ensure_ascii=False)
        print(f"\nDry-run report saved to {args.output}", file=sys.stderr)
        return

    # -----------------------------------------------------------------------
    # Full verification
    # -----------------------------------------------------------------------
    print(
        f"Verifying {len(suspects)} entries against CrossRef (rate_limit={args.rate_limit}s) ...\n",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    for i, (entry, split) in enumerate(suspects, 1):
        key = entry.get("bibtex_key", f"entry_{i}")
        title = entry.get("fields", {}).get("title", "")[:60]
        print(f"[{i:2d}/{len(suspects)}] {key}  '{title}'", file=sys.stderr)

        result = verify_entry(entry, split, args.rate_limit)
        results.append(result)

        # Inline feedback
        rec = result["recommendation"]
        match = result["crossref_match"]
        sim = result["title_similarity"]
        sim_str = f"sim={sim:.3f}" if sim is not None else "sim=n/a"
        print(f"         [{match}] {sim_str}  -> {rec}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    print_report(results)

    reclassify = [r for r in results if r["recommendation"] == "RECLASSIFY_AS_VALID"]
    manual = [r for r in results if r["recommendation"] == "NEEDS_MANUAL_REVIEW"]
    keep = [r for r in results if r["recommendation"] == "KEEP_AS_HALLUCINATED"]

    output_data = {
        "dry_run": False,
        "total_suspects": len(results),
        "summary": {
            "reclassify_as_valid": len(reclassify),
            "needs_manual_review": len(manual),
            "keep_as_hallucinated": len(keep),
        },
        "threshold": TITLE_SIM_THRESHOLD,
        "results": results,
        "reclassify_keys": [r["bibtex_key"] for r in reclassify],
        "manual_review_keys": [r["bibtex_key"] for r in manual],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    print(f"\nFull report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
