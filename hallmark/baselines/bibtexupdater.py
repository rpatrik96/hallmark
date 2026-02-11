"""Baseline wrapper for bibtex-updater's fact-checking CLI (bibtex-check).

Maps bibtex-check JSONL output to HALLMARK Prediction format.
bibtex-updater verifies citations against CrossRef, DBLP, Semantic Scholar.

Install: pipx install bibtex-updater  (or uv tool install bibtex-updater)

NOTE: bibtex-updater requires bibtexparser 1.x which conflicts with
hallmark's bibtexparser>=2.0.  It must be installed in an isolated
environment (pipx / uv tool) and invoked as a CLI subprocess.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)

# Map bibtex-check status to HALLMARK label
STATUS_TO_LABEL: dict[str, str] = {
    "verified": "VALID",
    "not_found": "HALLUCINATED",
    "title_mismatch": "HALLUCINATED",
    "author_mismatch": "HALLUCINATED",
    "year_mismatch": "VALID",  # Minor mismatch, not hallucination
    "venue_mismatch": "VALID",  # Minor mismatch
    "partial_match": "HALLUCINATED",
    "hallucinated": "HALLUCINATED",
    "api_error": "VALID",  # Conservative: don't flag on errors
    "url_verified": "VALID",
    "url_accessible": "VALID",
    "url_not_found": "HALLUCINATED",
    "url_content_mismatch": "HALLUCINATED",
    "book_verified": "VALID",
    "book_not_found": "HALLUCINATED",
    "working_paper_verified": "VALID",
    "working_paper_not_found": "HALLUCINATED",
    "skipped": "VALID",  # Conservative
}

# Map bibtex-check status to confidence
STATUS_TO_CONFIDENCE: dict[str, float] = {
    "verified": 0.95,
    "not_found": 0.80,
    "title_mismatch": 0.85,
    "author_mismatch": 0.75,
    "year_mismatch": 0.60,
    "venue_mismatch": 0.60,
    "partial_match": 0.70,
    "hallucinated": 0.90,
    "api_error": 0.30,
    "url_verified": 0.90,
    "url_accessible": 0.70,
    "url_not_found": 0.75,
    "url_content_mismatch": 0.80,
    "book_verified": 0.90,
    "book_not_found": 0.75,
    "working_paper_verified": 0.85,
    "working_paper_not_found": 0.70,
    "skipped": 0.50,
}


def entries_to_bib(entries: list[BenchmarkEntry]) -> str:
    """Convert benchmark entries to a BibTeX string."""
    return "\n\n".join(e.to_bibtex() for e in entries)


def run_bibtex_check(
    entries: list[BenchmarkEntry],
    extra_args: list[str] | None = None,
    timeout: float = 600.0,
    rate_limit: int = 120,
    academic_only: bool = True,
    **_kw: object,
) -> list[Prediction]:
    """Run bibtex-check on a list of entries and return predictions.

    Writes entries to a temp .bib file, runs bibtex-check with --jsonl output,
    and parses the results into Prediction objects.  On timeout, reads any
    partial JSONL output that was written before the process was killed.

    Args:
        entries: Benchmark entries to verify.
        extra_args: Additional CLI arguments for bibtex-check.
        timeout: Timeout in seconds (default: 600).
        rate_limit: API requests per minute (default: 120, up from CLI default 45).
        academic_only: Skip web/book/working-paper checks (default: True).
    """
    start_time = time.time()

    # Use a directory we control to avoid cleanup race on timeout
    tmpdir = tempfile.mkdtemp()
    bib_path = Path(tmpdir) / "input.bib"
    jsonl_path = Path(tmpdir) / "results.jsonl"

    try:
        # Write BibTeX file
        bib_content = entries_to_bib(entries)
        bib_path.write_text(bib_content)

        # Build command with performance optimizations
        cmd = [
            "bibtex-check",
            str(bib_path),
            "--jsonl",
            str(jsonl_path),
            "--rate-limit",
            str(rate_limit),
        ]
        if academic_only:
            cmd.append("--academic-only")
        if extra_args:
            cmd.extend(extra_args)

        # Run bibtex-check
        logger.info(f"Running: {' '.join(cmd)}")
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode not in (0, 2, 4):
                logger.error(f"bibtex-check failed (exit {result.returncode}): {result.stderr}")
        except FileNotFoundError:
            logger.error("bibtex-check not found. Install with: pipx install bibtex-updater")
            return _fallback_predictions(entries)
        except subprocess.TimeoutExpired:
            timed_out = True

        elapsed = time.time() - start_time

        # Parse JSONL output (works for both complete and partial results)
        predictions: list[Prediction] = []
        if jsonl_path.exists():
            predictions = _parse_jsonl_output(jsonl_path, elapsed, len(entries))

        checked = len(predictions)

        if timed_out:
            logger.warning(
                f"bibtex-check timed out after {timeout}s: "
                f"{checked}/{len(entries)} entries completed"
            )

        if not predictions and not timed_out:
            logger.warning("No JSONL output file produced")

    finally:
        # Clean up temp files
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    # Fill in missing predictions (entries not in output)
    predicted_keys = {p.bibtex_key for p in predictions}
    for entry in entries:
        if entry.bibtex_key not in predicted_keys:
            reason = (
                f"bibtex-check timed out ({checked}/{len(entries)} completed)"
                if timed_out
                else "Entry not in bibtex-check output"
            )
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=reason,
                    wall_clock_seconds=elapsed / len(entries),
                )
            )

    return predictions


def _parse_jsonl_output(
    jsonl_path: Path,
    total_elapsed: float,
    total_entries: int,
) -> list[Prediction]:
    """Parse bibtex-check JSONL output into Predictions."""
    predictions = []
    per_entry_time = total_elapsed / total_entries if total_entries > 0 else 0.0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON line: {line[:100]}")
                continue

            key = record.get("key", "")
            status = record.get("status", "skipped")
            confidence = record.get("confidence", STATUS_TO_CONFIDENCE.get(status, 0.5))
            mismatched = record.get("mismatched_fields", [])
            api_sources = record.get("api_sources", [])
            errors = record.get("errors", [])

            label = STATUS_TO_LABEL.get(status, "VALID")

            reason_parts = [f"Status: {status}"]
            if mismatched:
                reason_parts.append(f"Mismatched: {mismatched}")
            if errors:
                reason_parts.append(f"Errors: {errors}")

            predictions.append(
                Prediction(
                    bibtex_key=key,
                    label=label,  # type: ignore[arg-type]
                    confidence=confidence,
                    reason="; ".join(reason_parts),
                    api_sources_queried=api_sources,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(api_sources),
                )
            )

    return predictions


def _fallback_predictions(entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Generate conservative fallback predictions when bibtex-check fails."""
    return [
        Prediction(
            bibtex_key=e.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason="Fallback: bibtex-check unavailable",
        )
        for e in entries
    ]
