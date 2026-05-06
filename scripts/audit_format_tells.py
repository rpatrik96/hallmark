"""Audit format tells in HALLMARK perturbation generators.

Walks the generator package, extracts module-level string constants,
classifies each as a potential format tell, counts occurrences in
dev_public.jsonl, and writes tables/format_tells_audit.csv.

Usage:
    uv run python scripts/audit_format_tells.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
GENERATORS_DIR = REPO_ROOT / "hallmark" / "dataset" / "generators"
DEV_PUBLIC = REPO_ROOT / "data" / "v1.0" / "dev_public.jsonl"
OUTPUT_CSV = REPO_ROOT / "tables" / "format_tells_audit.csv"

# ---------------------------------------------------------------------------
# Heuristics for "potential format tell"
# ---------------------------------------------------------------------------

# DOI prefixes whose registrant range is unregistered (provably fictional)
_FAKE_DOI_PATTERN = re.compile(r"^10\.(9999[0-9]|8888[0-3]|7777[0-3]|6666[0-1])")

# Common placeholder author tokens
_PLACEHOLDER_AUTHOR_TOKENS = {
    "john doe",
    "jane smith",
    "author one",
    "author two",
    "first author",
    "last author",
    "unknown author",
}

# Templated venue patterns (formulaic, not modelled on real venue names)
_VENUE_TEMPLATE_PATTERN = re.compile(
    r"(intl\.|international) (conf\.|conference|workshop|symposium) on .+",
    re.IGNORECASE,
)


def _is_potential_tell(value: str) -> bool:
    """Return True if value matches any format-tell heuristic."""
    v = value.strip()
    if _FAKE_DOI_PATTERN.match(v):
        return True
    if v.lower() in _PLACEHOLDER_AUTHOR_TOKENS:
        return True
    return bool(_VENUE_TEMPLATE_PATTERN.match(v))


# ---------------------------------------------------------------------------
# Hallucination-type affinity per constant name
# ---------------------------------------------------------------------------
_CONST_TYPE_MAP: dict[str, list[str]] = {
    "FAKE_DOI_PREFIXES": ["fabricated_doi"],
    "FAKE_AUTHORS": ["placeholder_authors"],
    "HYBRID_FAKE_AUTHORS": ["hybrid_fabrication"],
    "FAKE_VENUES": ["nonexistent_venue", "preprint_as_published"],
    "CHIMERIC_TITLE_TEMPLATES": ["chimeric_title"],
    "ML_BUZZWORD_WORDS": ["chimeric_title"],
    "PLAUSIBLE_FIRST_NAMES": ["plausible_fabrication"],
    "PLAUSIBLE_LAST_NAMES": ["plausible_fabrication"],
    "PLAUSIBLE_METHODS": ["plausible_fabrication"],
    "PLAUSIBLE_DOMAINS": ["plausible_fabrication"],
    "PLAUSIBLE_PROPERTIES": ["plausible_fabrication"],
    "PLAUSIBLE_NOUNS": ["plausible_fabrication"],
    "PLAUSIBLE_SETTINGS": ["plausible_fabrication"],
    "VALID_VENUES": ["wrong_venue", "preprint_as_published"],
    "VALID_JOURNALS": ["wrong_venue", "preprint_as_published"],
    "VALID_CONFERENCES": ["wrong_venue", "preprint_as_published"],
    "REAL_VENUES": ["plausible_fabrication", "arxiv_version_mismatch"],
}


# ---------------------------------------------------------------------------
# Extract module-level list/set/tuple constants from a .py file via AST
# ---------------------------------------------------------------------------


def _extract_string_constants(path: Path) -> dict[str, list[str]]:
    """Return {var_name: [str_values]} for module-level Assign nodes."""
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}

    result: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Only simple name assignments
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        var_name: str = node.targets[0].id  # type: ignore[union-attr]
        val = node.value
        # Accept List, Set, Tuple
        if isinstance(val, (ast.List, ast.Set, ast.Tuple)):
            strings = []
            for elt in val.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    strings.append(elt.value)
            if strings:
                result[var_name] = strings
    return result


# ---------------------------------------------------------------------------
# Load dev_public entries
# ---------------------------------------------------------------------------


def _load_dev_public() -> list[dict[str, object]]:
    entries = []
    with DEV_PUBLIC.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _entry_text(entry: dict[str, object]) -> str:
    """Flatten all string-valued fields + authors into a single searchable string."""
    parts: list[str] = []
    fields = entry.get("fields", {})
    if isinstance(fields, dict):
        for v in fields.values():
            if isinstance(v, str):
                parts.append(v)
    raw = entry.get("raw_bibtex", "")
    if isinstance(raw, str):
        parts.append(raw)
    return " ".join(parts).lower()


def _count_occurrences(value: str, entries: list[dict[str, object]]) -> int:
    """Count how many entries contain *value* as a substring (case-insensitive)."""
    needle = value.lower()
    return sum(1 for e in entries if needle in _entry_text(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Collect constants from all generator .py files
    py_files = sorted(GENERATORS_DIR.glob("*.py"))
    all_rows: list[dict[str, object]] = []

    for py_path in py_files:
        if py_path.name.startswith("__"):
            continue
        generator_name = py_path.stem
        constants = _extract_string_constants(py_path)
        for const_name, values in constants.items():
            htypes = _CONST_TYPE_MAP.get(const_name, ["unknown"])
            for val in values:
                all_rows.append(
                    {
                        "generator": generator_name,
                        "constant_name": const_name,
                        "value": val,
                        "hallucination_types_affected": "|".join(htypes),
                        "is_potential_tell": _is_potential_tell(val),
                    }
                )

    # Load dev_public
    print(f"Loading {DEV_PUBLIC} …")
    entries = _load_dev_public()
    n_total = len(entries)
    hal_entries = [e for e in entries if e.get("label") == "HALLUCINATED"]
    n_hal = len(hal_entries)
    print(f"  Total entries: {n_total}, hallucinated: {n_hal}")

    # Count occurrences in dev_public for each value
    print(f"Counting occurrences for {len(all_rows)} constant values …")
    for row in all_rows:
        row["occurrence_count_dev_public"] = _count_occurrences(str(row["value"]), entries)

    # Write CSV
    fieldnames = [
        "generator",
        "constant_name",
        "value",
        "occurrence_count_dev_public",
        "hallucination_types_affected",
        "is_potential_tell",
    ]
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    tell_rows = [r for r in all_rows if r["is_potential_tell"]]
    tell_values = {str(r["value"]) for r in tell_rows}

    # Types affected by tells
    types_with_tells: set[str] = set()
    for r in tell_rows:
        for t in str(r["hallucination_types_affected"]).split("|"):
            types_with_tells.add(t)

    # Fraction of hallucinated entries containing at least one tell
    def entry_has_tell(entry: dict[str, object]) -> bool:
        text = _entry_text(entry)
        return any(v.lower() in text for v in tell_values)

    hal_with_tell = [e for e in hal_entries if entry_has_tell(e)]
    n_hal_tell = len(hal_with_tell)
    frac_hal_tell = n_hal_tell / n_hal if n_hal else 0.0

    # Per-type breakdown
    type_counts: dict[str, dict[str, int]] = {}
    for e in hal_entries:
        htype = str(e.get("hallucination_type", "unknown"))
        if htype not in type_counts:
            type_counts[htype] = {"total": 0, "with_tell": 0}
        type_counts[htype]["total"] += 1
        if entry_has_tell(e):
            type_counts[htype]["with_tell"] += 1

    # Upper-bound calculation:
    # The tell-exploitable types are those with tells; the upper bound on
    # format-tell contribution to DR equals the *joint detection rate* on
    # tell-bearing types weighted by their share of the hallucinated set.
    # We conservatively assume *all* detections on tell-bearing types could
    # come from the tell (worst case). The share of those types is:
    tell_type_entries = sum(v["total"] for k, v in type_counts.items() if k in types_with_tells)
    share_tell_types = tell_type_entries / n_hal if n_hal else 0.0

    print()
    print("=" * 60)
    print("FORMAT TELLS AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total potential tell values:          {len(tell_values)}")
    print(f"Hallucination types with tells:       {sorted(types_with_tells)}")
    print(f"Hallucinated entries in dev_public:   {n_hal}")
    print(f"  … with at least one tell:           {n_hal_tell} ({frac_hal_tell:.1%})")
    print()
    print("Per-type breakdown:")
    for htype, counts in sorted(type_counts.items()):
        frac = counts["with_tell"] / counts["total"] if counts["total"] else 0.0
        marker = " ← tell-bearing" if htype in types_with_tells else ""
        print(
            f"  {htype:<35s}  {counts['with_tell']:>4}/{counts['total']:<4}  ({frac:.0%}){marker}"
        )
    print()
    print(
        f"Upper bound on format-tell contribution to overall DR:\n"
        f"  Tell-bearing types share of hallucinated set: "
        f"{tell_type_entries}/{n_hal} = {share_tell_types:.1%}\n"
        f"  => Conservative upper bound Y ≤ {share_tell_types * 100:.1f} pp"
    )
    print("=" * 60)

    # Exit with error if no tells found (sanity check)
    if not tell_values:
        print("ERROR: no tells found — check heuristics", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
