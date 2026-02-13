"""Stage 8: Sanitize all splits — absorbs logic from sanitize_dataset.py + 8 fix scripts.

Applied transformations (in order):
1. Strip DBLP numeric suffixes from author names
2. Normalize bibtex_type (article/misc -> inproceedings for hallucinated)
3. Strip leaky fields that only appear in hallucinated entries
4. Fix temporal distribution (years -> 2021-2023, except future_date)
5. Populate missing publication_date from year field
6. Recompute fields_complete subtest
7. Fix cross_db_agreement for types with real title+authors
8. Drop retracted_paper entries from all splits
9. Remove fabricated real-world entries

Fixes NOT absorbed (unnecessary with clean generation):
- fix_near_miss_titles.py — generator already produces correct mutations
- fix_plausible_fabrication.py — scale_up generates with unique title pools
- fix_retracted_papers.py — retracted_paper type dropped entirely
- fix_mislabeled_llm_entries.py — filtered during stage 6 integration
"""

from __future__ import annotations

import logging
import random
import re

from hallmark.dataset.schema import BenchmarkEntry

logger = logging.getLogger(__name__)

# Fields that ONLY appear in hallucinated entries — leak labels
HALLUCINATION_ONLY_FIELDS = {
    "address",
    "archiveprefix",
    "edition",
    "editor",
    "eprint",
    "note",
    "number",
    "organization",
    "series",
    "volume",
}

# Additional fields to strip (valid entries never have them)
STRIP_FIELDS = HALLUCINATION_ONLY_FIELDS | {"pages", "publisher"}

# Valid year range for non-future_date entries
VALID_YEAR_MIN = 2021
VALID_YEAR_MAX = 2023

# Keys of fabricated entries in real_world_incidents.jsonl
FAKE_REALWORLD_KEYS = {
    "realworld_future_date_pattern",
    "realworld_nonexistent_venue",
    "realworld_fabricated_doi",
    "realworld_hybrid_fabrication",
}

# Types where title+authors are real -> cross_db_agreement should be True
CROSS_DB_TRUE_TYPES = {"wrong_venue", "preprint_as_published", "arxiv_version_mismatch"}

# Required fields for fields_complete subtest
REQUIRED_FIELDS = {"title", "author", "year"}
VENUE_FIELDS = {"booktitle", "journal"}
IDENTIFIER_FIELDS = {"doi", "url"}

# DBLP numeric suffix pattern
_DBLP_SUFFIX = re.compile(r" \d{4}")


def _strip_dblp_suffixes(entries: list[BenchmarkEntry]) -> int:
    """Strip DBLP numeric suffixes from author names (e.g., 'Author 0001' -> 'Author')."""
    modified = 0
    for entry in entries:
        author = entry.fields.get("author", "")
        if not author:
            continue
        authors = author.split(" and ")
        cleaned = [_DBLP_SUFFIX.sub("", a.strip()) for a in authors]
        cleaned_str = " and ".join(cleaned)
        if cleaned_str != author:
            entry.fields["author"] = cleaned_str
            modified += 1
    return modified


def _fix_bibtex_type(entries: list[BenchmarkEntry]) -> int:
    """Normalize article/misc -> inproceedings for hallucinated entries."""
    fixed = 0
    for entry in entries:
        if entry.label != "HALLUCINATED":
            continue
        if entry.bibtex_type in ("article", "misc"):
            entry.bibtex_type = "inproceedings"
            fields = entry.fields
            if "journal" in fields and "booktitle" not in fields:
                fields["booktitle"] = fields.pop("journal")
            elif "journal" in fields and "booktitle" in fields:
                del fields["journal"]
            fixed += 1
    return fixed


def _strip_leaky_fields(entries: list[BenchmarkEntry]) -> int:
    """Remove fields that only appear in hallucinated entries (label leak)."""
    stripped = 0
    for entry in entries:
        before = set(entry.fields.keys())
        for f in STRIP_FIELDS:
            entry.fields.pop(f, None)
        if set(entry.fields.keys()) != before:
            stripped += 1

    # Also strip any remaining hall-only fields dynamically
    valid_field_set: set[str] = set()
    for e in entries:
        if e.label == "VALID":
            valid_field_set.update(e.fields.keys())

    hall_only: set[str] = set()
    for e in entries:
        if e.label == "HALLUCINATED":
            for f in e.fields:
                if f not in valid_field_set:
                    hall_only.add(f)

    if hall_only:
        logger.info("Stripping additional hall-only fields: %s", hall_only)
        for e in entries:
            for f in hall_only:
                e.fields.pop(f, None)

    return stripped


def _fix_temporal_distribution(entries: list[BenchmarkEntry], seed: int) -> int:
    """Constrain hallucination years to valid range (except future_date)."""
    fixed = 0
    for entry in entries:
        if entry.label != "HALLUCINATED":
            continue
        if entry.hallucination_type == "future_date":
            continue

        year_str = entry.fields.get("year", "")
        if not year_str.isdigit():
            continue

        year = int(year_str)
        if year < VALID_YEAR_MIN or year > VALID_YEAR_MAX:
            rng = random.Random(f"{seed}_{entry.bibtex_key}")
            entry.fields["year"] = str(rng.randint(VALID_YEAR_MIN, VALID_YEAR_MAX))
            fixed += 1
    return fixed


def _fix_publication_dates(entries: list[BenchmarkEntry]) -> int:
    """Populate missing publication_date from year field."""
    fixed = 0
    for entry in entries:
        if not entry.publication_date:
            year = entry.fields.get("year", "")
            if year and year.isdigit():
                entry.publication_date = f"{year}-01-01"
                fixed += 1
    return fixed


def _recompute_fields_complete(entries: list[BenchmarkEntry]) -> int:
    """Recompute fields_complete subtest based on required fields + identifier.

    Also coerces any non-boolean values to bool (guards against generator bugs
    that assign the identifier string instead of bool(identifier)).
    """
    changed = 0
    for entry in entries:
        fields = entry.fields

        has_required = all(fields.get(f, "").strip() for f in REQUIRED_FIELDS)
        has_venue = any(fields.get(f, "").strip() for f in VENUE_FIELDS)
        has_identifier = any(fields.get(f, "").strip() for f in IDENTIFIER_FIELDS)

        new_val = has_required and has_venue and has_identifier
        old_val = entry.subtests.get("fields_complete")
        # Coerce any non-boolean values (e.g., stale string from generator bug)
        if old_val is not None and not isinstance(old_val, bool):
            old_val = bool(old_val)
        if old_val != new_val:
            changed += 1
        entry.subtests["fields_complete"] = new_val
    return changed


def _fix_cross_db_agreement(entries: list[BenchmarkEntry]) -> int:
    """Fix cross_db_agreement for types where title+authors are real."""
    fixed = 0
    for entry in entries:
        if (
            entry.label == "HALLUCINATED"
            and entry.hallucination_type in CROSS_DB_TRUE_TYPES
            and entry.subtests.get("cross_db_agreement") is False
        ):
            entry.subtests["cross_db_agreement"] = True
            fixed += 1
    return fixed


_VENUE_CODE_MAP = {
    "neurips": "nips",
    "icml": "icml",
    "iclr": "iclr",
    "cvpr": "cvpr",
    "iccv": "iccv",
    "eccv": "eccv",
    "aaai": "aaai",
    "acl": "acl",
    "emnlp": "emnlp",
    "naacl": "naacl",
    "sigir": "sigir",
    "kdd": "kdd",
    "www": "www",
    "ijcai": "ijcai",
    "colt": "colt",
    "aistats": "aistats",
}


def _ensure_url_presence(entries: list[BenchmarkEntry]) -> int:
    """Add plausible DBLP-style URLs to entries missing them (prevents URL-presence leak)."""
    import hashlib

    fixed = 0
    for entry in entries:
        if entry.fields.get("url"):
            continue

        fields = entry.fields
        author = fields.get("author", "Unknown")
        year = fields.get("year", "2022")
        booktitle = fields.get("booktitle", "").lower()

        # Determine venue code
        venue_code = "conf"
        for name, code in _VENUE_CODE_MAP.items():
            if name in booktitle:
                venue_code = code
                break

        # Extract first author's last name
        first_author = author.split(" and ")[0].strip()
        parts = first_author.split()
        lastname = re.sub(r"[^a-zA-Z]", "", parts[-1]) if parts else "Author"

        # Hash-based suffix for uniqueness
        key_hash = hashlib.md5(entry.bibtex_key.encode()).hexdigest()[:4].upper()
        yr = year[-2:] if len(year) >= 2 else "22"

        entry.fields["url"] = f"https://dblp.org/rec/conf/{venue_code}/{lastname}{key_hash}{yr}"
        fixed += 1

    return fixed


def _drop_retracted_paper(entries: list[BenchmarkEntry]) -> tuple[list[BenchmarkEntry], int]:
    """Remove retracted_paper entries (not in taxonomy)."""
    filtered = [e for e in entries if e.hallucination_type != "retracted_paper"]
    dropped = len(entries) - len(filtered)
    return filtered, dropped


def _remove_fake_realworld(entries: list[BenchmarkEntry]) -> tuple[list[BenchmarkEntry], int]:
    """Remove fabricated entries from real-world incidents."""
    filtered = [e for e in entries if e.bibtex_key not in FAKE_REALWORLD_KEYS]
    removed = len(entries) - len(filtered)
    return filtered, removed


def stage_sanitize(
    splits: dict[str, list[BenchmarkEntry]],
    seed: int,
) -> dict[str, list[BenchmarkEntry]]:
    """Apply all sanitization fixes to all splits.

    Args:
        splits: Dict mapping split name to list of entries.
        seed: Random seed for deterministic year remapping.

    Returns:
        Sanitized splits dict.
    """
    logger.info("Starting sanitization...")

    total_fixes: dict[str, int] = {}

    for split_name, entries in splits.items():
        logger.info("Sanitizing %s (%d entries)...", split_name, len(entries))

        # 1. Strip DBLP suffixes
        n = _strip_dblp_suffixes(entries)
        total_fixes[f"{split_name}_dblp"] = n

        # 2. Fix bibtex_type
        n = _fix_bibtex_type(entries)
        total_fixes[f"{split_name}_bibtex_type"] = n

        # 3. Strip leaky fields
        n = _strip_leaky_fields(entries)
        total_fixes[f"{split_name}_leaky_fields"] = n

        # 3b. Ensure all entries have a URL (prevents URL-presence leak)
        n = _ensure_url_presence(entries)
        total_fixes[f"{split_name}_url_presence"] = n

        # 4. Fix temporal distribution
        n = _fix_temporal_distribution(entries, seed)
        total_fixes[f"{split_name}_temporal"] = n

        # 5. Fix publication dates
        n = _fix_publication_dates(entries)
        total_fixes[f"{split_name}_pub_dates"] = n

        # 6. Recompute fields_complete
        n = _recompute_fields_complete(entries)
        total_fixes[f"{split_name}_fields_complete"] = n

        # 7. Fix cross_db_agreement
        n = _fix_cross_db_agreement(entries)
        total_fixes[f"{split_name}_cross_db"] = n

        # 8. Drop retracted_paper
        entries, n = _drop_retracted_paper(entries)
        total_fixes[f"{split_name}_retracted"] = n

        # 9. Remove fake real-world
        entries, n = _remove_fake_realworld(entries)
        total_fixes[f"{split_name}_fake_rw"] = n

        splits[split_name] = entries

    # Log summary
    for key, count in sorted(total_fixes.items()):
        if count > 0:
            logger.info("  %s: %d", key, count)

    total = sum(total_fixes.values())
    logger.info("Sanitization complete: %d total fixes applied", total)

    return splits
