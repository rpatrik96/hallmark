"""Baseline wrapper for bibtex-updater's fact-checking CLI (bibtex-check).

Maps bibtex-check JSONL output to HALLMARK Prediction format.
bibtex-updater verifies citations against CrossRef, DBLP, Semantic Scholar.

Install: pipx install bibtex-updater  (or uv tool install bibtex-updater)

NOTE: bibtex-updater requires bibtexparser 1.x which conflicts with
hallmark's bibtexparser>=2.0.  It must be installed in an isolated
environment (pipx / uv tool) and invoked as a CLI subprocess.

Newer bibtex-check releases (post-1.2.0) extend the JSONL output contract;
all extensions are presence-detected so older-format records parse exactly
as before (the precomputed reference results are unaffected):

- New problem statuses: ``nonexistent_venue`` (claimed venue unknown to the
  DBLP/OpenAlex venue registries while >=2 sources return the paper with
  other venues) and ``unpublished_at_claimed_venue`` (OpenReview: real paper,
  not accepted at the cited venue; env-gated upstream, off by default).
  ``author_truncated`` is now reachable in default mode (was --strict-only).
- ``coverage_incomplete`` (bool): the verdict is an abstention/API_ERROR
  reached while sources errored or were throttled.  A ``not_found`` carrying
  this flag is NOT a clean exhaustive miss — the wrapper treats it as an
  abstention (conservative VALID), not as evidence of fabrication.
- ``p_valid`` (float in [0, 1]): explicit P(entry as cited is genuine) — the
  value to threshold on.  When present it replaces the older realness
  inversion heuristic for deriving ``Prediction.confidence``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from hallmark.baselines.common import entries_to_bib, fallback_predictions, run_with_prescreening
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

# Map bibtex-check status to HALLMARK label.
# Statuses come from bibtex-updater's FactCheckStatus enum.
STATUS_TO_LABEL: dict[str, str] = {
    # Core academic verification
    "verified": "VALID",
    "not_found": "HALLUCINATED",
    "title_mismatch": "HALLUCINATED",
    "author_mismatch": "HALLUCINATED",
    "year_mismatch": "HALLUCINATED",  # Year differs from database record
    "venue_mismatch": "HALLUCINATED",  # Venue differs from database record
    "nonexistent_venue": "HALLUCINATED",  # Claimed venue unknown to DBLP/OpenAlex venue registries
    "partial_match": "HALLUCINATED",
    "hallucinated": "HALLUCINATED",
    "api_error": "VALID",  # Conservative: don't flag on errors
    # bibtex-updater >=1.2.0 statuses
    "unconfirmed": "VALID",  # Abstention (could-not-verify): conservative VALID
    "given_name_substitution": "HALLUCINATED",  # Co-author given name is a different person
    "arxiv_id_mismatch": "HALLUCINATED",  # Cited arXiv ID resolves to a different paper
    "doi_mismatch": "HALLUCINATED",  # Cited DOI resolves to a different paper
    "title_near_miss": "HALLUCINATED",  # --strict Levenshtein<=1 title near-miss
    "author_truncated": "HALLUCINATED",  # --strict silent author-list truncation
    "strict_warn_preprint_year": "VALID",  # --strict abstention: year unanchored
    "strict_warn_cnv": "VALID",  # --strict could-not-verify promotion (abstention)
    # Pre-API validation (bibtex-check runs these before querying APIs)
    "future_date": "HALLUCINATED",  # Year > current year
    "invalid_year": "HALLUCINATED",  # Non-numeric or < 1800
    "doi_not_found": "HALLUCINATED",  # DOI returns HTTP 404
    # Preprint detection
    "preprint_only": "HALLUCINATED",  # Paper only exists as preprint, not at claimed venue
    "unpublished_at_claimed_venue": "HALLUCINATED",  # OpenReview: real but not accepted at venue
    "published_version_exists": "VALID",  # Informational: published version found
    # Web reference verification
    "url_verified": "VALID",
    "url_accessible": "VALID",
    "url_not_found": "HALLUCINATED",
    "url_content_mismatch": "HALLUCINATED",
    # Book verification
    "book_verified": "VALID",
    "book_not_found": "HALLUCINATED",
    # Working paper verification
    "working_paper_verified": "VALID",
    "working_paper_not_found": "HALLUCINATED",
    # General
    "skipped": "VALID",  # Conservative
}

# Map bibtex-check status to confidence
STATUS_TO_CONFIDENCE: dict[str, float] = {
    "verified": 0.95,
    "not_found": 0.80,
    "title_mismatch": 0.85,
    "author_mismatch": 0.75,
    "year_mismatch": 0.75,
    "venue_mismatch": 0.80,
    "nonexistent_venue": 0.85,
    "partial_match": 0.70,
    "hallucinated": 0.90,
    "api_error": 0.30,
    "unconfirmed": 0.45,
    "given_name_substitution": 0.75,
    "arxiv_id_mismatch": 0.90,
    "doi_mismatch": 0.90,
    "title_near_miss": 0.80,
    "author_truncated": 0.70,
    "strict_warn_preprint_year": 0.40,
    "strict_warn_cnv": 0.40,
    "future_date": 0.95,
    "invalid_year": 0.70,
    "doi_not_found": 0.85,
    "preprint_only": 0.80,
    "unpublished_at_claimed_venue": 0.75,
    "published_version_exists": 0.60,
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


def run_bibtex_check(
    entries: list[BlindEntry],
    extra_args: list[str] | None = None,
    timeout: float = 7200.0,
    rate_limit: int = 120,
    academic_only: bool = True,
    skip_prescreening: bool = False,
    **_kw: object,
) -> list[Prediction]:
    """Run bibtex-check on a list of entries and return predictions.

    Writes entries to a temp .bib file, runs bibtex-check with --jsonl output,
    and parses the results into Prediction objects.  On timeout, reads any
    partial JSONL output that was written before the process was killed.

    Pre-screening (DOI check, year bounds, author heuristics) runs before
    bibtex-check to catch obvious hallucinations early, then results are merged.

    Args:
        entries: Benchmark entries to verify.
        extra_args: Additional CLI arguments for bibtex-check.
        timeout: Timeout in seconds (default: 600).
        rate_limit: API requests per minute (default: 120, up from CLI default 45).
        academic_only: Skip web/book/working-paper checks (default: True).
        skip_prescreening: Skip pre-screening checks (default: False).
    """
    predictions, _ = run_bibtex_check_with_status(
        entries,
        extra_args=extra_args,
        timeout=timeout,
        rate_limit=rate_limit,
        academic_only=academic_only,
        skip_prescreening=skip_prescreening,
    )
    return predictions


def run_bibtex_check_with_status(
    entries: list[BlindEntry],
    extra_args: list[str] | None = None,
    timeout: float = 7200.0,
    rate_limit: int = 120,
    academic_only: bool = True,
    skip_prescreening: bool = False,
    **_kw: object,
) -> tuple[list[Prediction], dict[str, str]]:
    """Run bibtex-check and return both predictions and raw per-entry status strings.

    Same behaviour as ``run_bibtex_check`` for the predictions list; additionally
    returns a status dict that a downstream cascade orchestrator can use to route
    entries to the next stage.

    Status vocabulary
    -----------------
    Values are the raw ``status`` field from the bibtex-check JSONL output, e.g.:

    - ``"verified"`` — found in at least one academic database and metadata matches
    - ``"not_found"`` — no database returned a matching record.  The same status
      string also covers coverage-incomplete lookups (post-1.2.0
      ``coverage_incomplete`` records, where sources errored / were throttled);
      the prediction for those is an abstention-style VALID, and downstream
      cascades should keep routing ``"not_found"`` as uncertain.
    - ``"title_mismatch"`` / ``"author_mismatch"`` / ``"year_mismatch"`` /
      ``"venue_mismatch"`` — found but a field differs from the claimed value
    - ``"nonexistent_venue"`` — claimed venue unknown to the DBLP/OpenAlex venue
      registries while the paper itself is real (positive problem evidence)
    - ``"partial_match"`` — some fields match, others do not
    - ``"api_error"`` — transient API failure; treated conservatively as VALID
    - ``"future_date"`` / ``"invalid_year"`` — pre-API year validation failed
    - ``"doi_not_found"`` — DOI returned HTTP 404
    - ``"preprint_only"`` — paper exists only as a preprint, not at the claimed venue
    - ``"unpublished_at_claimed_venue"`` — OpenReview: real paper, not accepted at
      the cited venue (env-gated upstream, off by default)
    - ``"published_version_exists"`` — informational; published version was found
    - ``"url_verified"`` / ``"url_accessible"`` / ``"url_not_found"`` /
      ``"url_content_mismatch"`` — web-reference results (academic_only=False only)
    - ``"book_verified"`` / ``"book_not_found"`` — book-reference results
    - ``"working_paper_verified"`` / ``"working_paper_not_found"``
    - ``"skipped"`` — entry was skipped by bibtex-check (e.g. unsupported entry type)

    Sentinel values (not from bibtex-check itself)
    -----------------------------------------------
    - ``"missing"`` — bibtex-check produced no JSONL record for this key (e.g. the
      process timed out before reaching it, or the entry was dropped).  The
      prediction for this key is a conservative VALID backfill.
    - ``"prescreening_override"`` — pre-screening changed the final verdict relative
      to the raw bibtex-check result (e.g. pre-screening flagged HALLUCINATED while
      the tool returned VALID).  The cascade orchestrator should treat the prediction
      as already decided and not re-route these entries to another stage.

    Args:
        entries: Benchmark entries to verify.
        extra_args: Additional CLI arguments for bibtex-check.
        timeout: Timeout in seconds (default: 7200).
        rate_limit: API requests per minute (default: 120).
        academic_only: Skip web/book/working-paper checks (default: True).
        skip_prescreening: Skip pre-screening checks (default: False).

    Returns:
        A 2-tuple ``(predictions, status_dict)`` where ``status_dict`` maps every
        input ``bibtex_key`` to a status string.  The dict is guaranteed to contain
        an entry for every key in ``entries``.
    """
    all_keys: list[str] = [e.bibtex_key for e in entries]

    # Step 1: Run the subprocess on all entries to get raw tool predictions.
    # The reason string encodes the raw status as "Status: <status>[; ...]".
    tool_predictions = _run_bibtex_check_subprocess(
        entries,
        extra_args=extra_args,
        timeout=timeout,
        rate_limit=rate_limit,
        academic_only=academic_only,
    )

    # Step 2: Extract raw status from each tool prediction's reason string.
    tool_key_to_status: dict[str, str] = {}
    for pred in tool_predictions:
        reason = pred.reason or ""
        if reason.startswith("Status: "):
            raw_status = reason.split(";")[0].removeprefix("Status: ").strip()
        else:
            raw_status = "skipped"
        tool_key_to_status[pred.bibtex_key] = raw_status

    # Step 3: Determine which keys produced no JSONL output (timeout / dropped).
    tool_key_set = {p.bibtex_key for p in tool_predictions}
    missing_keys: set[str] = {e.bibtex_key for e in entries} - tool_key_set

    # Step 4: Obtain the final merged predictions via run_with_prescreening,
    # which handles backfill and pre-screening merge in one pass.
    def _run_tool(tool_entries: list[BlindEntry]) -> list[Prediction]:
        return _run_bibtex_check_subprocess(
            tool_entries,
            extra_args=extra_args,
            timeout=timeout,
            rate_limit=rate_limit,
            academic_only=academic_only,
        )

    final_predictions = run_with_prescreening(
        entries,
        _run_tool,
        skip_prescreening=skip_prescreening,
        backfill_reason="Entry not in bibtex-check output",
    )

    # Step 5: Detect pre-screening overrides by comparing final label to tool label.
    # An override occurred when pre-screening changed the verdict (tool said VALID
    # but pre-screening raised it to HALLUCINATED, or no tool record existed and
    # pre-screening provided the prediction).
    tool_key_to_label: dict[str, str] = {p.bibtex_key: p.label for p in tool_predictions}
    final_key_to_label: dict[str, str] = {p.bibtex_key: p.label for p in final_predictions}

    # Step 6: Build status dict — guaranteed to cover every input key.
    status_dict: dict[str, str] = {}
    for key in all_keys:
        if key in missing_keys:
            # No tool output; pre-screening may have provided a prediction, but
            # either way the tool had no verdict — report as "missing".
            # If pre-screening also changed the outcome we still prefer "missing"
            # over "prescreening_override" since the tool never ran for this key.
            status_dict[key] = "missing"
        elif not skip_prescreening and final_key_to_label.get(key) != tool_key_to_label.get(key):
            # Pre-screening changed the label relative to the raw tool result.
            status_dict[key] = "prescreening_override"
        else:
            status_dict[key] = tool_key_to_status.get(key, "missing")

    return final_predictions, status_dict


def _run_bibtex_check_subprocess(
    entries: list[BlindEntry],
    extra_args: list[str] | None = None,
    timeout: float = 7200.0,
    rate_limit: int = 120,
    academic_only: bool = True,
) -> list[Prediction]:
    """Run bibtex-check subprocess and return raw predictions (no pre-screening)."""
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
        # Pass S2 API key if available (bibtex-check supports --s2-api-key)
        import os

        s2_key = os.environ.get("S2_API_KEY")
        if s2_key:
            cmd.extend(["--s2-api-key", s2_key])
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
            return fallback_predictions(entries, reason="Fallback: bibtex-check unavailable")
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

    return predictions


def parse_jsonl_to_raw(jsonl_path: Path) -> dict[str, dict]:
    """Parse bibtex-check JSONL output into raw record dicts (no Prediction conversion).

    Returns:
        Mapping from bibtex_key to the full raw record dict containing status,
        mismatched_fields, api_sources, confidence, errors, etc.
    """
    records: dict[str, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON line in raw JSONL: {line[:100]}")
                continue
            key = record.get("key", "")
            if key:
                records[key] = record
    return records


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
            raw_confidence = record.get("confidence", STATUS_TO_CONFIDENCE.get(status, 0.5))
            mismatched = record.get("mismatched_fields", [])
            api_sources = record.get("api_sources", [])
            errors = record.get("errors", [])

            label = STATUS_TO_LABEL.get(status, "VALID")

            # Post-1.2.0 records carry ``coverage_incomplete``: the abstention
            # was reached while sources errored / were throttled, so a
            # ``not_found`` with this flag is NOT a clean exhaustive miss.
            # Treat it as an abstention (conservative VALID), not as evidence
            # of fabrication. For all other statuses the flag is informational
            # and the label mapping is unchanged.
            incomplete_not_found = (
                status == "not_found" and record.get("coverage_incomplete") is True
            )
            if incomplete_not_found:
                label = "VALID"

            p_valid = record.get("p_valid")
            if incomplete_not_found:
                confidence = 0.45
            elif p_valid is not None:
                # Post-1.2.0 records emit an explicit ``p_valid`` = P(entry as
                # cited is genuine) — the documented value to threshold on.
                # HALLMARK's Prediction.confidence is confidence in the
                # assigned label, so VALID keeps p_valid and HALLUCINATED gets
                # 1 - p_valid. Its presence implies the new format, so this
                # replaces the 1.2.0 realness inversion heuristic below.
                confidence = float(p_valid) if label == "VALID" else 1.0 - float(p_valid)
            else:
                # bibtex-updater >=1.2.0 emits ``confidence`` as P(entry is real/valid)
                # (verified ~0.67, mismatch ~0.0, unconfirmed ~0.22) plus a new
                # ``confidence_score`` field. HALLMARK's Prediction.confidence is
                # confidence-in-the-assigned-label, so convert: VALID keeps P(real);
                # HALLUCINATED gets 1 - P(real). We detect the 1.2.0 realness
                # semantics by the presence of the new fields so 0.10.0 output
                # (which already encoded label-confidence) is left unchanged.
                is_v12_realness = "confidence_score" in record or "abstained" in record
                if is_v12_realness and label == "HALLUCINATED":
                    confidence = 1.0 - raw_confidence
                else:
                    confidence = raw_confidence

            reason_parts = [f"Status: {status}"]
            if incomplete_not_found:
                reason_parts.append(
                    "Lookup incomplete due to source errors/throttling — "
                    "abstention, not evidence of fabrication"
                )
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
