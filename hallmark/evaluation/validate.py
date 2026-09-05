"""Validation for pre-computed baseline reference results.

Validates that reference result files are intact, match expected checksums,
and deserialize correctly as EvaluationResult objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from hallmark.dataset.schema import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating reference results."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#: Splits carrying an entry that a result may legitimately not score, and how
#: many. ``stress_test`` holds 121 hallucinated entries plus a single VALID
#: contamination canary; with one valid entry the false-positive rate is not a
#: measurement, so the cascade rows are scored on the 121 and say so in the
#: paper. The validator had no way to express that, and failed three released
#: results on it -- which meant the whole ``baselines.yml`` matrix went red the
#: first time its conclusion was allowed to mean anything.
#:
#: Deliberately not a tolerance. An off-by-one allowance would also pass a run
#: that silently dropped an entry; this permits exactly the documented count.
CANARY_ENTRIES: dict[str, int] = {"stress_test": 1}


def _allowed_entry_counts(split_name: str, expected_total: int | None) -> set[int]:
    """Entry counts a result may report for *split_name*.

    The split total always, plus the total minus that split's canary entries
    where a split has any.
    """
    if expected_total is None:
        return set()
    allowed = {expected_total}
    canaries = CANARY_ENTRIES.get(split_name)
    if canaries:
        allowed.add(expected_total - canaries)
    return allowed


def validate_reference_results(
    results_dir: str | Path,
    metadata_path: str | Path | None = None,
    *,
    strict: bool = False,
) -> ValidationResult:
    """Validate pre-computed reference results.

    Checks:
        1. manifest.json exists and is valid JSON
        2. All listed result files exist and checksums match
        3. Each result JSON deserializes as a valid EvaluationResult
        4. num_entries matches dataset metadata (if metadata_path provided)
        5. (strict) Rejects F1=0.0 as likely failed run

    Args:
        results_dir: Directory containing manifest.json and result files.
        metadata_path: Optional path to dataset metadata.json for cross-validation.
        strict: If True, reject results with F1=0.0.

    Returns:
        ValidationResult with pass/fail and error list.
    """
    results_dir = Path(results_dir)
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Check manifest exists
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return ValidationResult(passed=False, errors=["manifest.json not found"])

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return ValidationResult(passed=False, errors=[f"manifest.json is invalid: {e}"])

    files: dict[str, dict] = manifest.get("files", {})

    # Empty manifest is valid (placeholder state)
    if not files:
        return ValidationResult(passed=True, warnings=["manifest has no files"])

    # Load metadata for cross-validation if provided
    metadata_splits: dict | None = None
    if metadata_path is not None:
        meta_path = Path(metadata_path)
        if meta_path.exists():
            try:
                metadata_splits = json.loads(meta_path.read_text()).get("splits", {})
            except (json.JSONDecodeError, OSError):
                warnings.append("Could not load metadata.json for cross-validation")

    # 2-5. Validate each listed file
    for filename, file_info in files.items():
        file_path = results_dir / filename
        expected_sha = file_info.get("sha256", "")

        # File exists?
        if not file_path.exists():
            errors.append(f"{filename}: file not found")
            continue

        # Checksum matches?
        actual_sha = compute_sha256(file_path)
        if expected_sha and actual_sha != expected_sha:
            errors.append(
                f"{filename}: checksum mismatch "
                f"(expected {expected_sha[:12]}..., got {actual_sha[:12]}...)"
            )
            continue

        # Deserializes as valid EvaluationResult?
        try:
            data = json.loads(file_path.read_text())
            result = EvaluationResult.from_dict(data)
        except Exception as e:
            errors.append(f"{filename}: failed to deserialize: {e}")
            continue

        # Cross-validate num_entries with metadata
        if metadata_splits is not None:
            split_name = result.split_name
            if split_name in metadata_splits:
                expected_total = metadata_splits[split_name].get("total")
                allowed = _allowed_entry_counts(split_name, expected_total)
                if expected_total is not None and result.num_entries not in allowed:
                    errors.append(
                        f"{filename}: num_entries={result.num_entries} "
                        f"doesn't match metadata total={expected_total} "
                        f"for split {split_name}"
                    )
                # A count short by exactly the canaries is only fine if the
                # canaries are what is missing. The canary is the split's only
                # VALID entry, so a short result that still reports a VALID
                # entry dropped a hallucinated one instead -- which the count
                # allowance alone cannot see, and which made it a one-entry
                # tolerance in practice.
                canaries = CANARY_ENTRIES.get(split_name, 0)
                if (
                    expected_total is not None
                    and canaries
                    and result.num_entries == expected_total - canaries
                    and result.num_valid != 0
                ):
                    errors.append(
                        f"{filename}: num_entries={result.num_entries} omits {canaries} "
                        f"entry(ies) from split {split_name} but num_valid="
                        f"{result.num_valid}; the omitted entry is not the VALID canary, "
                        "so a scored entry was dropped"
                    )

        # Strict: reject F1=0.0 as likely failed run
        if strict and result.f1_hallucination == 0.0:
            errors.append(f"{filename}: F1=0.0 (likely failed run; use --no-strict to allow)")

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
