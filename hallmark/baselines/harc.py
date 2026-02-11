"""Baseline wrapper for HaRC (Hallucinated Reference Checker).

HaRC (https://pypi.org/project/harcx/) validates BibTeX citations against
Semantic Scholar, DBLP, Google Scholar, and Open Library.

Install: pipx install harcx  (or uv tool install harcx)

NOTE: harcx requires bibtexparser 1.x which conflicts with hallmark's
bibtexparser>=2.0.  It must be installed in an isolated environment
(pipx / uv tool) and invoked as a CLI subprocess.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


def _parse_harcx_output(output: str) -> dict[str, list[str]]:
    """Parse harcx CLI output into a dict of {bibtex_key: [issues]}.

    The CLI output format for flagged entries is::

        [bibtex_key]
          Title: ...
          ...
          Issue: description

    Returns an empty dict when all entries are verified successfully.
    """
    flagged: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in output.splitlines():
        # Match entry key lines like "[fake_entry_2024]"
        key_match = re.match(r"^\[(\S+)\]$", line.strip())
        if key_match:
            current_key = key_match.group(1)
            continue

        # Match issue lines like "  Issue: ..."
        issue_match = re.match(r"^\s+Issue:\s+(.+)$", line)
        if issue_match and current_key:
            flagged.setdefault(current_key, []).append(issue_match.group(1))

    return flagged


def run_harc(
    entries: list[BenchmarkEntry],
    author_threshold: float = 0.6,
    check_urls: bool = False,
    api_key: str | None = None,
    **_kw: object,
) -> list[Prediction]:
    """Run HaRC verification on benchmark entries via the harcx CLI.

    Requires: ``harcx`` on PATH (install with ``pipx install harcx``
    or ``uv tool install harcx``).

    Args:
        entries: Benchmark entries to verify.
        author_threshold: Author match threshold (0.0-1.0, default: 0.6).
        check_urls: Whether to verify URL reachability.
        api_key: Optional Semantic Scholar API key.

    Returns:
        List of Predictions.
    """
    harcx_bin = shutil.which("harcx")
    if harcx_bin is None:
        raise ImportError(
            "harcx CLI not found on PATH. "
            "Install with: pipx install harcx  (or uv tool install harcx)"
        )

    # Write entries to a temporary .bib file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
        for entry in entries:
            f.write(entry.to_bibtex() + "\n\n")
        bib_path = f.name

    cmd = [harcx_bin, "-q", "--threshold", str(author_threshold), bib_path]
    if check_urls:
        cmd.append("--check-urls")
    if api_key:
        cmd.extend(["--api-key", api_key])

    api_sources = ["semantic_scholar", "dblp", "google_scholar", "open_library"]

    start = time.time()
    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.warning("harcx timed out after 1800s on %d entries", len(entries))
        timed_out = True
    finally:
        Path(bib_path).unlink(missing_ok=True)

    total_time = time.time() - start
    per_entry_time = total_time / max(len(entries), 1)

    if timed_out:
        return [
            Prediction(
                bibtex_key=e.bibtex_key,
                label="VALID",
                confidence=0.5,
                reason="HaRC: timed out before completion",
                api_sources_queried=api_sources,
                wall_clock_seconds=per_entry_time,
                api_calls=0,
            )
            for e in entries
        ]

    # harcx exits 0 = all OK, 1 = issues found (or error)
    combined_output = result.stdout + result.stderr
    flagged_keys = _parse_harcx_output(combined_output)

    if result.returncode not in (0, 1):
        logger.warning("harcx exited with code %d: %s", result.returncode, combined_output[:500])

    # Map to predictions
    predictions: list[Prediction] = []

    for entry in entries:
        issues = flagged_keys.get(entry.bibtex_key, [])

        if issues:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=min(0.5 + 0.15 * len(issues), 0.95),
                    reason=f"HaRC flagged: {'; '.join(issues)}",
                    api_sources_queried=api_sources,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(api_sources),
                )
            )
        else:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.85,
                    reason="HaRC: No issues found across databases",
                    api_sources_queried=api_sources,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(api_sources),
                )
            )

    return predictions
