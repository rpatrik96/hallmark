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

from hallmark.dataset.schema import BenchmarkEntry, Prediction

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


def entries_to_bib(entries: list[BenchmarkEntry]) -> str:
    """Convert benchmark entries to a BibTeX string."""
    return "\n\n".join(e.to_bibtex() for e in entries)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    return ANSI_ESCAPE.sub("", text)


def run_verify_citations(
    entries: list[BenchmarkEntry],
    timeout: float = 600.0,
) -> list[Prediction]:
    """Run verify-citations on a list of entries and return predictions.

    Writes entries to a temp .bib file, runs verify-citations CLI,
    and parses the color-coded terminal output into Prediction objects.
    """
    predictions = []
    start_time = time.time()
    timed_out = False

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
            if result.returncode not in (0, 1):
                logger.error(f"verify-citations failed (exit {result.returncode}): {result.stderr}")
        except FileNotFoundError:
            logger.error("verify-citations not found. Install with: pip install verify-citations")
            return _fallback_predictions(entries)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            # Capture any partial output produced before timeout
            raw = exc.stdout or b""
            stdout = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

        elapsed = time.time() - start_time

        # Parse terminal output (works for both complete and partial results)
        if stdout:
            predictions = _parse_terminal_output(stdout, entries, elapsed, len(entries))

        checked = len(predictions)

        if timed_out:
            logger.warning(
                f"verify-citations timed out after {timeout}s: "
                f"{checked}/{len(entries)} entries completed"
            )

        if not predictions and not timed_out:
            logger.warning("No stdout from verify-citations")

    # Fill in missing predictions (entries not in output)
    predicted_keys = {p.bibtex_key for p in predictions}
    for entry in entries:
        if entry.bibtex_key not in predicted_keys:
            reason = (
                f"verify-citations timed out ({checked}/{len(entries)} completed)"
                if timed_out
                else "Entry not in verify-citations output"
            )
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=reason,
                    wall_clock_seconds=elapsed / len(entries),
                    api_sources_queried=API_SOURCES,
                    api_calls=0 if timed_out else len(API_SOURCES),
                )
            )

    return predictions


def _parse_terminal_output(
    stdout: str,
    entries: list[BenchmarkEntry],
    total_elapsed: float,
    total_entries: int,
) -> list[Prediction]:
    """Parse verify-citations terminal output into Predictions.

    The output format looks like:
        ✓ entry_key: Successfully verified
        ✗ entry_key: Paper not found
        ⚠ entry_key: Warning - 403 error
    """
    predictions = []
    per_entry_time = total_elapsed / total_entries if total_entries > 0 else 0.0

    # Strip ANSI color codes
    clean_output = strip_ansi_codes(stdout)

    # Create a mapping of keys for quick lookup
    key_to_entry = {e.bibtex_key: e for e in entries}

    # Parse line by line
    for line in clean_output.splitlines():
        line = line.strip()
        if not line:
            continue

        # Look for status markers
        marker = None
        for m in ["✓", "✗", "⚠", _INFO_MARKER]:
            if m in line:
                marker = m
                break

        if not marker:
            continue

        # Extract entry key (usually after marker, before colon)
        # Format: "✓ key: message" or "✗ key: message"
        match = re.match(r"[✓✗⚠\u2139]\s+([^\s:]+)\s*:\s*(.+)", line)
        if not match:
            continue

        key = match.group(1)
        message = match.group(2).strip()

        # Skip if not in our entries
        if key not in key_to_entry:
            continue

        label = MARKER_TO_LABEL.get(marker, "VALID")
        confidence = MARKER_TO_CONFIDENCE.get(marker, 0.5)

        reason_parts = [f"Status: {marker}"]
        if message:
            reason_parts.append(message)

        predictions.append(
            Prediction(
                bibtex_key=key,
                label=label,  # type: ignore[arg-type]
                confidence=confidence,
                reason="; ".join(reason_parts),
                api_sources_queried=API_SOURCES,
                wall_clock_seconds=per_entry_time,
                api_calls=len(API_SOURCES),
            )
        )

    return predictions


def _fallback_predictions(entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Generate conservative fallback predictions when verify-citations fails."""
    return [
        Prediction(
            bibtex_key=e.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason="Fallback: verify-citations unavailable",
            api_sources_queried=API_SOURCES,
            api_calls=0,
        )
        for e in entries
    ]
