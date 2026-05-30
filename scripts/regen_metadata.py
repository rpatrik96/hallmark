#!/usr/bin/env python3
"""Regenerate ``data/v1.0/metadata.json`` count fields from the live data files.

Why this exists
---------------
``scripts/stages/finalize.py:_compute_metadata`` is the canonical metadata
computation, but it only runs as the final stage of a full ``build_dataset``
re-run and it only knows the three v1.0 pipeline splits (dev_public /
test_public / test_hidden). The shipped ``metadata.json`` carries more: the
``stress_test`` split, the ``v1_1_extensions`` block (``test_crossdomain`` with
its ``source_distribution`` / ``subcorpora``), and ``data_sources`` /
``upstream_updates`` provenance.

After the ground-truth relabel, the per-split label counts (valid / hallucinated)
and the tier / type distributions shifted, but the *structure* of metadata.json
and its descriptive provenance fields did not. So this script does the minimal,
deterministic thing: it RECOMPUTES every count/distribution field from the actual
``.jsonl`` data — using the SAME logic as ``_compute_metadata`` — and writes them
back in place, leaving the structural/descriptive fields (file names, rationale
strings, subcorpora descriptions, data_sources, upstream_updates) untouched.

Numbers are never hand-edited: every integer below is computed from the data.
Re-running on already-correct data is a byte-identical no-op (idempotent).

Usage
-----
    python scripts/regen_metadata.py            # rewrite data/v1.0/metadata.json
    python scripts/regen_metadata.py --check     # exit 1 if file is stale (CI-friendly)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hallmark.dataset.schema import BenchmarkEntry  # noqa: E402

DATA_DIR = _REPO_ROOT / "data"
METADATA_PATH = DATA_DIR / "v1.0" / "metadata.json"

# Map every split named in metadata.json to the data file that backs it. The
# hidden split lives under data/hidden/; all others under data/v1.0/.
SPLIT_FILES: dict[str, Path] = {
    "dev_public": DATA_DIR / "v1.0" / "dev_public.jsonl",
    "test_public": DATA_DIR / "v1.0" / "test_public.jsonl",
    "stress_test": DATA_DIR / "v1.0" / "stress_test.jsonl",
    "test_hidden": DATA_DIR / "hidden" / "test_hidden.jsonl",
    "test_crossdomain": DATA_DIR / "v1.0" / "test_crossdomain.jsonl",
}


def _raw_entries(path: Path) -> list[BenchmarkEntry]:
    """Parse every entry from a JSONL file WITHOUT filtering canaries.

    ``hallmark.dataset.schema.load_entries`` drops ``__canary__`` watermark
    entries so they don't pollute evaluation metrics, but the canonical
    ``finalize.py:_compute_metadata`` runs over the full in-memory splits at build
    time — canary included. metadata.json therefore counts the canary (e.g.
    stress_test total=122, valid=1), so we parse raw to match.
    """
    entries: list[BenchmarkEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))
    return entries


def _split_counts(path: Path) -> dict[str, Any]:
    """Recompute one split's count block from its data file.

    Mirrors ``finalize.py:_compute_metadata`` exactly: ``valid`` / ``hallucinated``
    from ``label``; ``tier_distribution`` from ``difficulty_tier`` (str keys);
    ``type_distribution`` from ``hallucination_type``; ``generation_methods`` from
    ``generation_method``. All distributions are sorted for byte-stable output.
    """
    entries = _raw_entries(path)
    valid_count = sum(1 for e in entries if e.label == "VALID")
    hall_count = sum(1 for e in entries if e.label == "HALLUCINATED")

    tier_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)

    for e in entries:
        if e.difficulty_tier:
            tier_counts[str(e.difficulty_tier)] += 1
        if e.hallucination_type:
            type_counts[e.hallucination_type] += 1
        method_counts[e.generation_method] += 1

    return {
        "total": len(entries),
        "valid": valid_count,
        "hallucinated": hall_count,
        "tier_distribution": dict(sorted(tier_counts.items())),
        "type_distribution": dict(sorted(type_counts.items())),
        "generation_methods": dict(sorted(method_counts.items())),
    }


def _source_distribution(path: Path) -> dict[str, int]:
    """Per-source counts for cross-domain (its entries carry a ``source`` field)."""
    counts: dict[str, int] = defaultdict(int)
    for e in _raw_entries(path):
        if e.source:
            counts[str(e.source)] += 1
    return dict(sorted(counts.items()))


def _apply_counts(block: dict[str, Any], counts: dict[str, Any]) -> None:
    """Overwrite the count fields of a split block in place.

    Only keys that already exist in the block are updated, so we never inject a
    field the shipped schema did not carry (e.g. test_hidden historically omits
    ``valid``/``hallucinated`` — but if present they are corrected). ``total`` is
    always set since every split block carries it. The ``type_distribution`` for a
    split that counts VALID entries by a literal ``"valid"`` key (test_crossdomain)
    keeps that convention.
    """
    block["total"] = counts["total"]
    if "count" in block:
        block["count"] = counts["total"]
    for key in ("valid", "hallucinated", "tier_distribution", "generation_methods"):
        if key in block:
            block[key] = counts[key]
        elif key in ("valid", "hallucinated"):
            # Surface a missing label count rather than silently dropping it: the
            # relabel makes these meaningful for every split.
            block[key] = counts[key]
    # type_distribution: preserve a pre-existing "valid" tally convention.
    if "type_distribution" in block:
        new_types = dict(counts["type_distribution"])
        if "valid" in block["type_distribution"]:
            new_types = {"valid": counts["valid"], **new_types}
        block["type_distribution"] = new_types


def regenerate(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata with all count fields recomputed from the data files."""
    # v1.0 splits live under metadata["splits"].
    v10_splits: dict[str, Any] = metadata.get("splits", {})
    for split_name, block in v10_splits.items():
        path = SPLIT_FILES.get(split_name)
        if path is None or not path.exists():
            continue
        _apply_counts(block, _split_counts(path))

    # v1.1 cross-domain split lives under v1_1_extensions.splits.
    ext = metadata.get("v1_1_extensions", {})
    ext_splits: dict[str, Any] = ext.get("splits", {})
    for split_name, block in ext_splits.items():
        path = SPLIT_FILES.get(split_name)
        if path is None or not path.exists():
            continue
        _apply_counts(block, _split_counts(path))
        if "source_distribution" in block:
            block["source_distribution"] = _source_distribution(path)

    # total_entries = sum of the v1.0 split totals (the canonical benchmark size;
    # the v1.1 cross-domain split is eval-only and excluded, matching the prior
    # convention where total_entries covered only metadata["splits"]).
    metadata["total_entries"] = sum(block.get("total", 0) for block in v10_splits.values())
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if metadata.json is stale vs the data.",
    )
    args = ap.parse_args()

    original_text = METADATA_PATH.read_text()
    metadata = json.loads(original_text)
    regenerate(metadata)

    new_text = json.dumps(metadata, indent=4, ensure_ascii=False) + "\n"

    if args.check:
        if new_text != original_text:
            sys.stderr.write("metadata.json is STALE — run `python scripts/regen_metadata.py`\n")
            return 1
        print("metadata.json is up to date.")
        return 0

    METADATA_PATH.write_text(new_text)
    print(f"Wrote {METADATA_PATH}")
    print(f"  total_entries = {metadata['total_entries']}")
    for name, block in metadata.get("splits", {}).items():
        print(
            f"  {name}: total={block.get('total')} "
            f"valid={block.get('valid')} hallucinated={block.get('hallucinated')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
