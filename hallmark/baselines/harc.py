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

API_SOURCES = ["semantic_scholar", "dblp", "google_scholar", "open_library"]


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


def _run_harcx_batch(
    harcx_bin: str,
    batch: list[BenchmarkEntry],
    author_threshold: float,
    check_urls: bool,
    api_key: str | None,
    timeout: float,
) -> tuple[dict[str, list[str]], set[str], bool]:
    """Run harcx on a batch of entries.

    Returns:
        (flagged_keys, checked_keys, timed_out)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
        for entry in batch:
            f.write(entry.to_bibtex() + "\n\n")
        bib_path = f.name

    cmd = [harcx_bin, "-q", "--threshold", str(author_threshold), bib_path]
    if check_urls:
        cmd.append("--check-urls")
    if api_key:
        cmd.extend(["--api-key", api_key])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {}, set(), True
    finally:
        Path(bib_path).unlink(missing_ok=True)

    combined_output = result.stdout + result.stderr
    flagged = _parse_harcx_output(combined_output)

    if result.returncode not in (0, 1):
        logger.warning("harcx exited with code %d: %s", result.returncode, combined_output[:500])

    checked = {e.bibtex_key for e in batch}
    return flagged, checked, False


def run_harc(
    entries: list[BenchmarkEntry],
    author_threshold: float = 0.6,
    check_urls: bool = False,
    api_key: str | None = None,
    batch_size: int = 20,
    batch_timeout: float = 600.0,
    **_kw: object,
) -> list[Prediction]:
    """Run HaRC verification on benchmark entries via the harcx CLI.

    Processes entries in batches to collect partial results on timeout.
    Each batch has its own timeout; completed batches are preserved even
    if later batches time out.

    Args:
        entries: Benchmark entries to verify.
        author_threshold: Author match threshold (0.0-1.0, default: 0.6).
        check_urls: Whether to verify URL reachability.
        api_key: Optional Semantic Scholar API key.
        batch_size: Entries per batch (default: 20).
        batch_timeout: Timeout per batch in seconds (default: 600).

    Returns:
        List of Predictions.
    """
    harcx_bin = shutil.which("harcx")
    if harcx_bin is None:
        raise ImportError(
            "harcx CLI not found on PATH. "
            "Install with: pipx install harcx  (or uv tool install harcx)"
        )

    start = time.time()
    all_flagged: dict[str, list[str]] = {}
    all_checked: set[str] = set()
    timed_out_batches = 0
    total_batches = (len(entries) + batch_size - 1) // batch_size

    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"HaRC batch {batch_num}/{total_batches}: {len(batch)} entries")

        flagged, checked, timed_out = _run_harcx_batch(
            harcx_bin, batch, author_threshold, check_urls, api_key, batch_timeout
        )

        if timed_out:
            timed_out_batches += 1
            logger.warning(
                f"HaRC batch {batch_num}/{total_batches} timed out after {batch_timeout}s"
            )
            break  # Stop processing further batches

        all_flagged.update(flagged)
        all_checked.update(checked)

    total_time = time.time() - start
    per_entry_time = total_time / max(len(entries), 1)
    checked_count = len(all_checked)

    if timed_out_batches:
        logger.warning(
            f"HaRC completed {checked_count}/{len(entries)} entries "
            f"before timeout ({timed_out_batches} batch(es) timed out)"
        )

    # Map to predictions
    predictions: list[Prediction] = []

    for entry in entries:
        if entry.bibtex_key in all_checked:
            issues = all_flagged.get(entry.bibtex_key, [])
            if issues:
                predictions.append(
                    Prediction(
                        bibtex_key=entry.bibtex_key,
                        label="HALLUCINATED",
                        confidence=min(0.5 + 0.15 * len(issues), 0.95),
                        reason=f"HaRC flagged: {'; '.join(issues)}",
                        api_sources_queried=API_SOURCES,
                        wall_clock_seconds=per_entry_time,
                        api_calls=len(API_SOURCES),
                    )
                )
            else:
                predictions.append(
                    Prediction(
                        bibtex_key=entry.bibtex_key,
                        label="VALID",
                        confidence=0.85,
                        reason="HaRC: No issues found across databases",
                        api_sources_queried=API_SOURCES,
                        wall_clock_seconds=per_entry_time,
                        api_calls=len(API_SOURCES),
                    )
                )
        else:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=f"HaRC: timed out ({checked_count}/{len(entries)} completed)",
                    api_sources_queried=API_SOURCES,
                    wall_clock_seconds=per_entry_time,
                    api_calls=0,
                )
            )

    return predictions
