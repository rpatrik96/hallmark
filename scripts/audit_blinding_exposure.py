#!/usr/bin/env python3
"""Audit the released prediction records for blinded-field exposure.

HALLMARK withholds ``url`` from verifiers at dispatch time
(:data:`hallmark.dataset.blinding.BLIND_EXCLUDED_FIELDS`). The released
corpus, however, still *stores* a ``url`` on some entries, and an ad-hoc
resume runner that read the corpus JSONL directly bypassed the dispatch-time
blinding for a handful of checkpoints. This script measures what actually
reached the models, from the released artifacts alone, so the finding is
reproducible by a reviewer.

It makes no network calls and no API calls: it joins the per-entry prediction
dumps against the corpus on ``bibtex_key`` and looks for URL material in the
model's stored ``reason`` string.

Strict criterion (the load-bearing evidence)
--------------------------------------------
An entry counts as exposed when all of the following hold.

1. The corpus entry carries a non-empty ``url``.
2. The URL host is not DOI- or arXiv-derived (``doi.org``, ``arxiv.org``, and
   the usual aliases). A verifier handed a DOI can paraphrase it into a
   resolver URL without ever having seen the ``url`` field, so those cases
   prove nothing and are reported separately.
3. A path (or query) token of at least ``--min-token`` characters from that
   URL appears in the model's ``reason``.
4. That token appears in **no other field** of the entry, so it cannot have
   been reconstructed from the metadata the verifier was legitimately given.

Weak signal (reported separately)
---------------------------------
The same match on a DOI- or arXiv-derived URL. A reviewer will reasonably
object that these are paraphrasable from the DOI or eprint, so they are
counted but never mixed into the strict totals.

Retrieval-capable runners (reported separately)
-----------------------------------------------
Agentic, tool-augmented, and cascade runners query external APIs, so a URL in
their reason may have come back from CrossRef rather than from the prompt. A
strict hit there is not evidence of input leakage, so those dumps are split
out under their own heading.

Usage::

    python scripts/audit_blinding_exposure.py
    python scripts/audit_blinding_exposure.py --json audit.json
    python scripts/audit_blinding_exposure.py --show-hits
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hallmark.dataset.blinding import BLIND_EXCLUDED_FIELDS  # noqa: E402

#: Hosts whose URLs a model can reconstruct from a ``doi`` or ``eprint``
#: field. A match on one of these is not evidence that the model saw ``url``.
DOI_DERIVED_HOSTS: frozenset[str] = frozenset(
    {
        "doi.org",
        "dx.doi.org",
        "www.doi.org",
        "doi.acm.org",
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
    }
)

#: Substrings marking a dump as produced by a runner that can retrieve
#: metadata externally, so URL material in its reason may be API output.
RETRIEVAL_MARKERS: tuple[str, ...] = (
    "agentic",
    "tool_augmented",
    "cascade",
    "btu",
    "bibtexupdater",
    "harc",
    "checkifexist",
)

#: Corpus files searched for entries, in join priority order.
DEFAULT_CORPUS: tuple[str, ...] = (
    "data/v1.0/dev_public.jsonl",
    "data/v1.0/test_public.jsonl",
    "data/v1.0/test_crossdomain.jsonl",
    "data/v1.0/stress_test.jsonl",
    "data/v1.0/supplement_chatgpt_citations.jsonl",
    "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl",
    "data/hidden/test_hidden.jsonl",
)

#: Directories scanned for per-entry prediction dumps.
DEFAULT_DUMP_ROOTS: tuple[str, ...] = ("results", "data/v1.0/baseline_results")

#: Path components never scanned (git worktrees of concurrent agents, caches).
SKIP_PARTS: frozenset[str] = frozenset({".git", ".venv", ".claude", "__pycache__"})

MIN_TOKEN_DEFAULT = 8


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def load_corpus(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Index corpus entries by ``bibtex_key``.

    Later files never overwrite an earlier key, so the join is deterministic
    when a key appears in more than one split.
    """
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for rec in _read_jsonl(path):
            key = rec.get("bibtex_key")
            if isinstance(key, str) and key not in index:
                rec["_corpus_file"] = str(path)
                index[key] = rec
    return index


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def _is_identifier_like(tok: str) -> bool:
    """True unless ``tok`` is an ordinary lowercase English word.

    Path segments like ``journals`` or ``proceedings`` clear the length bar but
    appear in model prose for reasons that have nothing to do with the URL
    ("some journals publish in press"), so requiring a digit or a capital
    keeps the test on genuine identifiers.
    """
    return not (tok.isalpha() and tok.islower())


def url_tokens(url: str, min_len: int, path_only: bool = False) -> set[str]:
    """Identifier-like tokens of ``url`` at least ``min_len`` characters long.

    The host is excluded: a model that knows the venue can name the domain
    without having seen the field. Path segments keep their punctuation, so a
    JMLR article id survives whole (``v24/22-0582.html`` -> ``22-0582.html``).

    Query and fragment tokens count too unless ``path_only`` is set. They
    matter: an OpenReview citation carries its whole identity in
    ``?id=HylxE1HKwS``, and a path-only tokenizer scores every such entry as
    clean while the model's reason quotes the id verbatim.
    """
    parsed = urlsplit(url)
    raw: list[str] = list(parsed.path.split("/"))
    if not path_only:
        for blob in (parsed.query, parsed.fragment):
            raw += blob.replace("&", " ").replace("=", " ").replace(";", " ").split()
    return {tok for tok in raw if len(tok) >= min_len and _is_identifier_like(tok)}


def other_field_blob(entry: dict[str, Any]) -> str:
    """Everything the verifier was legitimately given, lowercased.

    ``url`` itself is excluded (it is what we are testing), and so is
    ``raw_bibtex``: it embeds the URL verbatim, so including it would mask
    every real leak rather than reveal one.
    """
    fields = entry.get("fields") or {}
    parts = [str(v) for k, v in fields.items() if k not in BLIND_EXCLUDED_FIELDS]
    parts.append(str(entry.get("bibtex_key", "")))
    parts.append(str(entry.get("bibtex_type", "")))
    return " ".join(parts).lower()


def is_doi_derived(url: str) -> bool:
    return urlsplit(url).netloc.lower() in DOI_DERIVED_HOSTS


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_dump(
    path: Path,
    corpus: dict[str, dict[str, Any]],
    min_token: int,
    path_only: bool = False,
) -> dict[str, Any]:
    """Run the strict and weak tests over one per-entry prediction dump."""
    strict: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    n_records = 0
    n_with_url = 0
    n_joined = 0

    for rec in _read_jsonl(path):
        key = rec.get("bibtex_key")
        reason = rec.get("reason")
        if not isinstance(key, str) or not isinstance(reason, str):
            continue
        n_records += 1
        entry = corpus.get(key)
        if entry is None:
            continue
        n_joined += 1
        url = (entry.get("fields") or {}).get("url")
        if not url:
            continue
        n_with_url += 1

        tokens = url_tokens(str(url), min_token, path_only=path_only)
        if not tokens:
            continue
        blob = other_field_blob(entry)
        low_reason = reason.lower()
        matched = sorted(
            tok for tok in tokens if tok.lower() in low_reason and tok.lower() not in blob
        )
        if not matched:
            continue

        hit = {
            "bibtex_key": key,
            "label": entry.get("label"),
            "url": url,
            "matched_tokens": matched,
            "reason": reason,
            "corpus_file": entry.get("_corpus_file"),
        }
        (weak if is_doi_derived(str(url)) else strict).append(hit)

    return {
        "dump": str(path),
        "runner_class": "retrieval" if is_retrieval_dump(path) else "zero_shot",
        "n_records": n_records,
        "n_joined": n_joined,
        "n_entries_with_url": n_with_url,
        "n_strict_hits": len(strict),
        "n_strict_valid": sum(1 for h in strict if h["label"] == "VALID"),
        "n_strict_hallucinated": sum(1 for h in strict if h["label"] == "HALLUCINATED"),
        "n_weak_doi_derived_hits": len(weak),
        "strict_hits": strict,
        "weak_hits": weak,
    }


def is_retrieval_dump(path: Path) -> bool:
    low = str(path).lower()
    return any(marker in low for marker in RETRIEVAL_MARKERS)


def find_dumps(roots: Iterable[Path]) -> list[Path]:
    """Every per-entry prediction JSONL under ``roots``.

    A dump is any JSONL whose first record carries both ``bibtex_key`` and
    ``reason`` — the shape ``Prediction.to_json()`` writes.
    """
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if SKIP_PARTS & set(path.parts):
                continue
            first = next(_read_jsonl(path), None)
            if first is not None and "bibtex_key" in first and "reason" in first:
                out.append(path)
    return out


def corpus_url_census(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Per-file count of stored ``url`` fields — what the paper must report."""
    census: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        n = n_url = n_url_valid = 0
        for rec in _read_jsonl(path):
            n += 1
            if (rec.get("fields") or {}).get("url"):
                n_url += 1
                if rec.get("label") == "VALID":
                    n_url_valid += 1
        census.append(
            {
                "file": str(path),
                "n_entries": n,
                "n_with_url": n_url,
                "n_with_url_valid": n_url_valid,
                "n_with_url_hallucinated": n_url - n_url_valid,
            }
        )
    return census


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_census(census: list[dict[str, Any]]) -> None:
    print("=" * 78)
    print("CORPUS: entries that still store a `url`")
    print("=" * 78)
    print(f"{'file':<52}{'entries':>9}{'url':>6}{'V':>5}{'H':>5}")
    for row in census:
        print(
            f"{row['file']:<52}{row['n_entries']:>9}{row['n_with_url']:>6}"
            f"{row['n_with_url_valid']:>5}{row['n_with_url_hallucinated']:>5}"
        )


def _print_section(title: str, rows: list[dict[str, Any]], show_hits: bool) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("  (none)")
        return
    print(f"{'dump':<62}{'strict':>7}{'V':>4}{'H':>4}")
    for row in rows:
        print(
            f"{row['dump']:<62}{row['n_strict_hits']:>7}"
            f"{row['n_strict_valid']:>4}{row['n_strict_hallucinated']:>4}"
        )
    if show_hits:
        for row in rows:
            for hit in row["strict_hits"]:
                print()
                print(f"  {row['dump']}")
                print(f"    entry   {hit['bibtex_key']}  ({hit['label']})")
                print(f"    url     {hit['url']}")
                print(f"    tokens  {hit['matched_tokens']}")
                print(f"    reason  {hit['reason'][:400]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpus",
        nargs="*",
        default=list(DEFAULT_CORPUS),
        help="Corpus JSONL files to join against.",
    )
    ap.add_argument(
        "--dump-root",
        nargs="*",
        default=list(DEFAULT_DUMP_ROOTS),
        help="Directories scanned for per-entry prediction dumps.",
    )
    ap.add_argument(
        "--min-token",
        type=int,
        default=MIN_TOKEN_DEFAULT,
        help="Minimum URL-path token length to count as a match.",
    )
    ap.add_argument(
        "--path-only",
        action="store_true",
        help=(
            "Tokenize the URL path only, ignoring the query string. Reproduces "
            "the first pass of this audit, which scored every OpenReview "
            "`?id=` entry as clean."
        ),
    )
    ap.add_argument("--json", type=Path, default=None, help="Write the full report here.")
    ap.add_argument("--show-hits", action="store_true", help="Print every strict hit in full.")
    args = ap.parse_args()

    corpus_paths = [ROOT / p for p in args.corpus]
    dump_roots = [ROOT / p for p in args.dump_root]

    census = corpus_url_census(corpus_paths)
    corpus = load_corpus(corpus_paths)
    dumps = find_dumps(dump_roots)

    reports = [audit_dump(p, corpus, args.min_token, path_only=args.path_only) for p in dumps]
    for rep in reports:
        rep["dump"] = str(Path(rep["dump"]).relative_to(ROOT))

    zero_shot = [r for r in reports if r["runner_class"] == "zero_shot" and r["n_strict_hits"]]
    retrieval = [r for r in reports if r["runner_class"] == "retrieval" and r["n_strict_hits"]]
    zero_shot.sort(key=lambda r: (-r["n_strict_hits"], r["dump"]))
    retrieval.sort(key=lambda r: (-r["n_strict_hits"], r["dump"]))

    _print_census(census)
    _print_section(
        "STRICT EXPOSURE — zero-shot runners (evidence of input leakage)",
        zero_shot,
        args.show_hits,
    )
    _print_section(
        "STRICT MATCHES — retrieval-capable runners (NOT evidence: the URL "
        "can come back from an API)",
        retrieval,
        args.show_hits,
    )

    n_weak = sum(r["n_weak_doi_derived_hits"] for r in reports)
    n_strict_zs = sum(r["n_strict_hits"] for r in reports if r["runner_class"] == "zero_shot")
    print()
    print("=" * 78)
    print("TOTALS")
    print("=" * 78)
    print(f"  dumps scanned                                 {len(reports)}")
    print(f"  corpus entries indexed                        {len(corpus)}")
    print(f"  strict hits, zero-shot runners                {n_strict_zs}")
    print(
        f"  strict matches, retrieval-capable runners     "
        f"{sum(r['n_strict_hits'] for r in retrieval)}"
    )
    print(f"  weak (DOI/arXiv-derived) matches, all runners {n_weak}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "min_token": args.min_token,
                    "path_only": args.path_only,
                    "doi_derived_hosts": sorted(DOI_DERIVED_HOSTS),
                    "blind_excluded_fields": sorted(BLIND_EXCLUDED_FIELDS),
                    "corpus_census": census,
                    "reports": reports,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
