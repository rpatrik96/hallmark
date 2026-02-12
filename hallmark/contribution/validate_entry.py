"""Validate community-contributed benchmark entries.

Ensures contributed entries meet schema requirements, have proper annotations,
and don't duplicate existing entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

from hallmark.dataset.schema import (
    SUBTEST_NAMES,
    BenchmarkEntry,
    GenerationMethod,
    HallucinationType,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a contributed entry."""

    valid: bool
    errors: list[str]
    warnings: list[str]


def _validate_required_fields(entry: BenchmarkEntry) -> list[str]:
    """Check required BibTeX fields are present."""
    errors: list[str] = []
    if not entry.bibtex_key:
        errors.append("Missing bibtex_key")
    if not entry.bibtex_type:
        errors.append("Missing bibtex_type")
    if not entry.fields.get("title"):
        errors.append("Missing title field")
    if not entry.fields.get("author"):
        errors.append("Missing author field")
    if not entry.fields.get("year"):
        errors.append("Missing year field")
    return errors


def _validate_hallucination_metadata(entry: BenchmarkEntry) -> list[str]:
    """Validate hallucination-specific fields."""
    errors: list[str] = []
    if entry.label == "HALLUCINATED":
        if not entry.hallucination_type:
            errors.append("Hallucinated entry missing hallucination_type")
        else:
            valid_types = {t.value for t in HallucinationType}
            if entry.hallucination_type not in valid_types:
                errors.append(
                    f"Unknown hallucination_type: {entry.hallucination_type}. Valid: {valid_types}"
                )
        if entry.difficulty_tier is None:
            errors.append("Hallucinated entry missing difficulty_tier")
        elif entry.difficulty_tier not in (1, 2, 3):
            errors.append(f"Invalid difficulty_tier: {entry.difficulty_tier}")
        if not entry.explanation:
            errors.append("Hallucinated entry missing explanation")
    if entry.label == "VALID" and entry.hallucination_type is not None:
        errors.append("Valid entry should not have hallucination_type")
    return errors


def _validate_metadata(entry: BenchmarkEntry) -> tuple[list[str], list[str]]:
    """Validate generation method, subtests, and dates."""
    errors: list[str] = []
    warnings: list[str] = []

    valid_methods = {m.value for m in GenerationMethod}
    if entry.generation_method not in valid_methods:
        warnings.append(
            f"Unknown generation_method: {entry.generation_method}. Valid: {valid_methods}"
        )

    if not entry.subtests:
        warnings.append("No sub-test ground truth provided")
    else:
        for name in entry.subtests:
            if name not in SUBTEST_NAMES:
                warnings.append(f"Unknown sub-test: {name}")

    if not entry.publication_date:
        warnings.append("Missing publication_date")
    if not entry.added_to_benchmark:
        warnings.append("Missing added_to_benchmark")

    return errors, warnings


def validate_entry(
    entry: BenchmarkEntry,
    existing_entries: list[BenchmarkEntry] | None = None,
) -> ValidationResult:
    """Validate a single contributed entry.

    Checks:
    1. Required fields are present and non-empty
    2. Label is valid
    3. Hallucinated entries have type, tier, explanation
    4. Valid entries don't have hallucination metadata
    5. Generation method is recognized
    6. Sub-test ground truth is provided
    7. Entry doesn't duplicate existing entries
    """
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_validate_required_fields(entry))

    if entry.label not in ("VALID", "HALLUCINATED"):
        errors.append(f"Invalid label: {entry.label}")

    errors.extend(_validate_hallucination_metadata(entry))

    meta_errors, meta_warnings = _validate_metadata(entry)
    errors.extend(meta_errors)
    warnings.extend(meta_warnings)

    if existing_entries:
        for existing in existing_entries:
            if _is_duplicate(entry, existing):
                errors.append(f"Duplicate of existing entry: {existing.bibtex_key}")
                break

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_batch(
    entries: list[BenchmarkEntry],
    existing_entries: list[BenchmarkEntry] | None = None,
) -> list[ValidationResult]:
    """Validate a batch of contributed entries."""
    results = []
    # Also check for duplicates within the batch
    seen: list[BenchmarkEntry] = list(existing_entries) if existing_entries else []

    for entry in entries:
        result = validate_entry(entry, seen)
        results.append(result)
        if result.valid:
            seen.append(entry)

    return results


def _is_duplicate(
    entry: BenchmarkEntry,
    existing: BenchmarkEntry,
    title_threshold: float = 0.95,
) -> bool:
    """Check if two entries are duplicates based on title similarity."""
    title1 = entry.fields.get("title", "").lower().strip()
    title2 = existing.fields.get("title", "").lower().strip()

    if not title1 or not title2:
        return False

    similarity = fuzz.token_sort_ratio(title1, title2) / 100.0
    return similarity >= title_threshold
