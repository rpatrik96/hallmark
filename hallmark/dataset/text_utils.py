"""Text matching and BibTeX parsing utilities shared across HALLMARK scripts."""

from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    normalized = title.lower()
    for char in ".,;:!?\"'()[]{}":
        normalized = normalized.replace(char, " ")
    return " ".join(normalized.split())


def title_similarity(title1: str, title2: str) -> float:
    """SequenceMatcher-based similarity. More precise than Jaccard for near-duplicate detection."""
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()


def title_jaccard(title1: str, title2: str) -> float:
    """Word-level Jaccard similarity. Faster than SequenceMatcher; order-independent."""
    words_a = set(title1.lower().split())
    words_b = set(title2.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def extract_last_names(author_str: str) -> set[str]:
    """Extract lowercased last names from a BibTeX-style author string.

    Handles both "Last, First" and "First Last" formats, joined by " and ".
    """
    last_names: set[str] = set()
    for part in author_str.split(" and "):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            last_name = part.split(",")[0].strip()
        else:
            tokens = part.split()
            last_name = tokens[-1] if tokens else ""
        if last_name:
            last_names.add(last_name.lower())
    return last_names


def authors_match_fuzzy(
    bib_authors: str,
    crossref_authors: list[dict],
    threshold: float = 0.5,
) -> bool:
    """Return True if last-name Jaccard overlap between bib_authors and crossref_authors >= threshold."""
    bib_lastnames = extract_last_names(bib_authors)
    cr_lastnames = {a.get("family", "").lower() for a in crossref_authors if a.get("family")}
    if not bib_lastnames or not cr_lastnames:
        return False
    overlap = len(bib_lastnames & cr_lastnames) / len(bib_lastnames | cr_lastnames)
    return overlap >= threshold


def parse_bibtex_entry(bibtex_str: str) -> dict | None:
    """Parse a single BibTeX entry into {"type": str, "key": str, "fields": dict}.

    Uses a brace-counting parser to correctly handle nested braces in field values
    (e.g. title = {A {Transformer} Model}). Returns None if the entry header can't
    be parsed or an unexpected exception occurs.
    """
    try:
        match = re.match(r"@(\w+)\{([^,]+),", bibtex_str, re.IGNORECASE)
        if not match:
            return None

        entry_type = match.group(1).lower()
        key = match.group(2).strip()

        fields: dict[str, str] = {}
        field_pattern = re.compile(r"(\w+)\s*=\s*")
        for fm in field_pattern.finditer(bibtex_str):
            field_name = fm.group(1).lower()
            start = fm.end()
            if start >= len(bibtex_str):
                continue
            delim = bibtex_str[start]
            if delim == "{":
                depth = 1
                i = start + 1
                while i < len(bibtex_str) and depth > 0:
                    if bibtex_str[i] == "{":
                        depth += 1
                    elif bibtex_str[i] == "}":
                        depth -= 1
                    i += 1
                if depth == 0:
                    fields[field_name] = bibtex_str[start + 1 : i - 1].strip()
            elif delim == '"':
                end = bibtex_str.find('"', start + 1)
                if end != -1:
                    fields[field_name] = bibtex_str[start + 1 : end].strip()

        return {"type": entry_type, "key": key, "fields": fields}
    except Exception as e:
        logger.debug("Failed to parse BibTeX: %s", e)
        return None


def parse_bibtex_fields(bibtex_str: str) -> dict[str, str]:
    """Extract field values from a BibTeX string.

    Simpler than parse_bibtex_entry — skips entry type/key, only returns fields.
    Uses a single-level regex (no nested-brace support); sufficient for well-formed
    machine-generated BibTeX.
    """
    fields: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", bibtex_str):
        fields[m.group(1).lower()] = m.group(2).strip()
    return fields
