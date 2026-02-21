"""Baseline wrapper for verify-citations CLI tool.

Maps verify-citations color-coded terminal output to HALLMARK Prediction format.
verify-citations checks BibTeX against arXiv, ACL Anthology, Semantic Scholar,
DBLP, Google Scholar, and DuckDuckGo.

Installation:
    pip install verify-citations

Usage:
    verify-citations --bibtex-file path/to/references.bib

Output markers:
    ✓ Green: Successfully verified
    ✗ Red: Critical errors (paper not found, metadata mismatch)
    ⚠ Yellow: Warnings (403 errors, unverified metadata)
    i Cyan: Informational
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.baselines.common import entries_to_bib, fallback_predictions, run_with_prescreening
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

# ANSI color code pattern for stripping
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# Unicode info marker (U+2139) - using escape to avoid ruff RUF001
_INFO_MARKER = "\u2139"

# Map verify-citations status markers to HALLMARK label and confidence
MARKER_TO_LABEL: dict[str, str] = {
    "✓": "VALID",
    "✗": "HALLUCINATED",
    "⚠": "HALLUCINATED",
    _INFO_MARKER: "VALID",
}

MARKER_TO_CONFIDENCE: dict[str, float] = {
    "✓": 0.90,
    "✗": 0.80,
    "⚠": 0.60,
    _INFO_MARKER: 0.50,
}

# API sources queried by verify-citations
API_SOURCES = [
    "arxiv",
    "acl_anthology",
    "semantic_scholar",
    "dblp",
    "google_scholar",
    "duckduckgo",
]


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    return ANSI_ESCAPE.sub("", text)


def run_verify_citations(
    entries: list[BlindEntry],
    timeout: float = 600.0,
    skip_prescreening: bool = False,
) -> list[Prediction]:
    """Run verify-citations on a list of entries and return predictions.

    Writes entries to a temp .bib file, runs verify-citations CLI,
    and parses the color-coded terminal output into Prediction objects.

    Pre-screening (DOI check, year bounds, author heuristics) runs before
    verify-citations to catch obvious hallucinations early, then results are merged.

    Args:
        entries: Benchmark entries to verify.
        timeout: Timeout in seconds (default: 600).
        skip_prescreening: Skip pre-screening checks (default: False).
    """

    def _run_tool(tool_entries: list[BlindEntry]) -> list[Prediction]:
        return _run_verify_citations_subprocess(tool_entries, timeout=timeout)

    return run_with_prescreening(
        entries,
        _run_tool,
        skip_prescreening=skip_prescreening,
        backfill_reason="Entry not in verify-citations output",
    )


def _run_verify_citations_subprocess(
    entries: list[BlindEntry],
    timeout: float = 600.0,
) -> list[Prediction]:
    """Run verify-citations subprocess and return raw predictions (no pre-screening)."""
    predictions = []
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        bib_path = Path(tmpdir) / "input.bib"

        # Write BibTeX file
        bib_content = entries_to_bib(entries)
        bib_path.write_text(bib_content)

        # Build command
        cmd = [
            "verify-citations",
            "--bibtex-file",
            str(bib_path),
        ]

        # Run verify-citations
        logger.info(f"Running: {' '.join(cmd)}")
        stdout = ""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout
            logger.info(
                "verify-citations exit=%d stdout=%d stderr=%d",
                result.returncode,
                len(result.stdout),
                len(result.stderr),
            )
            if result.stderr:
                logger.warning("verify-citations stderr: %s", result.stderr[:500])
            if result.returncode not in (0, 1):
                logger.error(
                    "verify-citations failed (exit %d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
        except FileNotFoundError:
            logger.error("verify-citations not found. Install with: pip install verify-citations")
            return fallback_predictions(
                entries,
                reason="Fallback: verify-citations unavailable",
                api_sources=API_SOURCES,
                api_calls=0,
            )
        except subprocess.TimeoutExpired as exc:
            # Capture any partial output produced before timeout
            raw = exc.stdout or b""
            stdout = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            logger.warning(f"verify-citations timed out after {timeout}s")

        elapsed = time.time() - start_time

        # Parse terminal output (works for both complete and partial results)
        if stdout:
            predictions = _parse_terminal_output(stdout, entries, elapsed, len(entries))

    return predictions


def _parse_terminal_output(
    stdout: str,
    entries: list[BlindEntry],
    total_elapsed: float,
    total_entries: int,
) -> list[Prediction]:
    """Parse verify-citations terminal output into Predictions.

    The actual output format is::

        [1/5] Verifying:
          [bibtex_key] Title (Authors, Year)
          Status: ✓ VERIFIED

        [2/5] Verifying:
          [bibtex_key] Title (Authors, Year)
          Status: ✗ ISSUES FOUND
            ⚠ Title matches but author list mismatch detected
    """
    predictions = []
    per_entry_time = total_elapsed / total_entries if total_entries > 0 else 0.0

    # Strip ANSI color codes
    clean_output = strip_ansi_codes(stdout)

    # Create a mapping of keys for quick lookup
    key_to_entry = {e.bibtex_key: e for e in entries}

    # Parse blocks: find key lines and their status lines
    current_key: str | None = None
    for line in clean_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Match entry key: "  [bibtex_key] Title..."
        key_match = re.match(r"^\[([^\]]+)\]\s+", stripped)
        if key_match and key_match.group(1) in key_to_entry:
            current_key = key_match.group(1)
            continue

        # Match status line: "Status: ✓ VERIFIED" or "Status: ✗ ISSUES FOUND"
        status_match = re.match(r"^Status:\s+([✓✗⚠])\s+(.+)", stripped)
        if status_match and current_key:
            marker = status_match.group(1)
            status_text = status_match.group(2)

            label = MARKER_TO_LABEL.get(marker, "VALID")
            confidence = MARKER_TO_CONFIDENCE.get(marker, 0.5)

            predictions.append(
                Prediction(
                    bibtex_key=current_key,
                    label=label,  # type: ignore[arg-type]
                    confidence=confidence,
                    reason=f"verify-citations: {status_text}",
                    api_sources_queried=API_SOURCES,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(API_SOURCES),
                )
            )
            current_key = None

    return predictions
