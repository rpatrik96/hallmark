"""Baseline wrapper for CiteVerifier (from the GhostCite paper).

CiteVerifier (https://github.com/NKU-AOSP-Lab/CiteVerifier) parses
bibliographic references, queries DBLP (local), Google Scholar, and Google
Search, and classifies citation validity as VALID/INVALID/SUSPICIOUS/UNVERIFIED.

Requires cloning the CiteVerifier repo and installing its dependencies.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)

# Map CiteVerifier status to HALLMARK label
CITEVERIFIER_STATUS_MAP: dict[str, str] = {
    "VALID": "VALID",
    "INVALID": "HALLUCINATED",
    "SUSPICIOUS": "HALLUCINATED",
    "UNVERIFIED": "VALID",  # Conservative: don't flag unverified
}

CITEVERIFIER_CONFIDENCE_MAP: dict[str, float] = {
    "VALID": 0.90,
    "INVALID": 0.85,
    "SUSPICIOUS": 0.65,
    "UNVERIFIED": 0.50,
}


def _entries_to_citeverifier_json(entries: list[BenchmarkEntry]) -> list[dict]:
    """Convert benchmark entries to CiteVerifier JSON input format."""
    refs = []
    for entry in entries:
        ref = {
            "id": entry.bibtex_key,
            "title": entry.fields.get("title", ""),
            "authors": entry.fields.get("author", ""),
            "year": entry.fields.get("year", ""),
            "venue": entry.fields.get("booktitle", "") or entry.fields.get("journal", ""),
            "doi": entry.fields.get("doi", ""),
        }
        refs.append(ref)
    return refs


def run_cite_verifier(
    entries: list[BenchmarkEntry],
    citeverifier_path: str | Path = "CiteVerifier",
    concurrent: int = 10,
    dblp_threshold: float = 0.9,
) -> list[Prediction]:
    """Run CiteVerifier on benchmark entries.

    Requires: CiteVerifier repo cloned and dependencies installed.
    See https://github.com/NKU-AOSP-Lab/CiteVerifier

    Args:
        entries: Benchmark entries to verify.
        citeverifier_path: Path to CiteVerifier repository.
        concurrent: Number of concurrent API requests.
        dblp_threshold: DBLP title match threshold.

    Returns:
        List of Predictions.
    """
    cv_path = Path(citeverifier_path)
    verifier_script = cv_path / "verifier.py"
    if not verifier_script.exists():
        logger.error(
            "CiteVerifier not found at %s. "
            "Clone from: https://github.com/NKU-AOSP-Lab/CiteVerifier",
            cv_path,
        )
        return fallback_predictions(entries, reason="Fallback: CiteVerifier unavailable")

    # Write entries as JSON input
    refs = _entries_to_citeverifier_json(entries)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as infile:
        json.dump(refs, infile)
        input_path = infile.name

    output_path = tempfile.mktemp(suffix=".json")

    start = time.time()
    try:
        result = subprocess.run(
            [
                "python3",
                str(verifier_script),
                "--input",
                input_path,
                "--output",
                output_path,
                "--concurrent",
                str(concurrent),
                "--dblp-threshold",
                str(dblp_threshold),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(cv_path),
        )
        if result.returncode != 0:
            logger.error(f"CiteVerifier failed: {result.stderr}")
            return fallback_predictions(
                entries, reason="CiteVerifier: verification failed, defaulting to VALID"
            )

        total_time = time.time() - start

        # Parse output
        with open(output_path) as f:
            cv_results = json.load(f)

    except subprocess.TimeoutExpired:
        logger.error("CiteVerifier timed out")
        return fallback_predictions(
            entries, reason="CiteVerifier: verification failed, defaulting to VALID"
        )
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"CiteVerifier output error: {e}")
        return fallback_predictions(
            entries, reason="CiteVerifier: verification failed, defaulting to VALID"
        )
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

    per_entry_time = total_time / max(len(entries), 1)

    # Index results by reference ID
    result_map: dict[str, dict] = {}
    if isinstance(cv_results, list):
        for r in cv_results:
            rid = r.get("reference_id") or r.get("id", "")
            result_map[rid] = r
    elif isinstance(cv_results, dict):
        result_map = cv_results

    # Map to predictions
    predictions = []
    api_sources = ["dblp", "google_scholar", "google_search"]

    for entry in entries:
        cv_result = result_map.get(entry.bibtex_key, {})
        status = cv_result.get("final_status", "UNVERIFIED").upper()
        title_sim = cv_result.get("title_similarity", 0.0)
        notes = cv_result.get("verification_notes", "")

        label = CITEVERIFIER_STATUS_MAP.get(status, "VALID")
        confidence = CITEVERIFIER_CONFIDENCE_MAP.get(status, 0.5)

        # Adjust confidence based on title similarity
        if status == "VALID" and title_sim > 0:
            confidence = max(confidence, title_sim)

        reason = f"CiteVerifier: {status}"
        if title_sim > 0:
            reason += f" (title_sim={title_sim:.2f})"
        if notes:
            reason += f" - {notes}"

        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label=label,  # type: ignore[arg-type]
                confidence=min(confidence, 1.0),
                reason=reason,
                api_sources_queried=api_sources,
                wall_clock_seconds=per_entry_time,
                api_calls=len(api_sources),
            )
        )

    return predictions
