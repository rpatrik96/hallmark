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

Every invocation is scored by a batch-level sanity check (``assess_batch_health``)
before the results are handed back.  ``not_found`` is evidence of fabrication only
when the sources were actually reached, so a batch where most entries come back
``not_found`` — or carry an explicit transport-failure status — is reported as a
broken lookup path rather than a bibliography of invented papers.  Callers that
persist results should gate on ``BatchHealth.suspected_transport_failure`` via
``run_bibtex_check_with_health`` and refuse to checkpoint a batch that trips it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hallmark.baselines._cache import redact_command
from hallmark.baselines.common import entries_to_bib, fallback_predictions, run_with_prescreening
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)


class SourceOutageError(RuntimeError):
    """bibtex-check reported that its sources went dark during the run.

    Raised rather than returned so the run cannot be silently scored. The tool
    prints "Treat this run as incomplete and discard its could-not-verify
    verdicts"; before this existed, the wrapper logged that and scored the run
    anyway.
    """


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
    "network_error": "VALID",  # Lookup never reached a source: abstention, not evidence
    "coverage_incomplete": "VALID",  # Sources throttled/errored: abstention, not evidence
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
    "network_error": 0.30,
    "coverage_incomplete": 0.45,
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


# ---------------------------------------------------------------------------
# Batch-level sanity check
# ---------------------------------------------------------------------------

# Statuses that say the lookup never reached a source, as opposed to reaching
# the sources and being told nothing is there.  ``api_error`` is the tool's
# long-standing catch-all; ``network_error`` is carried here so an upgraded tool
# that reports transport failures under their own status is handled the same way.
TRANSPORT_FAILURE_STATUSES: frozenset[str] = frozenset({"api_error", "network_error"})

# Statuses that say the sources answered but the sweep was not exhaustive — a
# source was throttled or errored out, so the absence of a record proves nothing.
# ``coverage_incomplete`` is currently a boolean flag on a ``not_found`` record;
# it is accepted as a status name here so the check keeps working if the tool
# promotes it to one.
COVERAGE_INCOMPLETE_STATUSES: frozenset[str] = frozenset({"coverage_incomplete"})

# Share of a batch that may come back without database evidence before we stop
# believing the batch.  Healthy HALLMARK runs sit at ~52% ``verified`` and 1-3%
# ``not_found``.  On 2026-09-02 a wifi outage made every source lookup fail DNS
# resolution and bibtex-check returned ``not_found`` for 2,500 consecutive
# references; the poisoned chunks ran 85-98% ``not_found``.  30% is an order of
# magnitude above the healthy ceiling and far below the observed poisoning, so it
# separates the two regimes with room on both sides — a genuinely
# fabrication-heavy bibliography stays under it, and no partial outage that
# matters lands beneath it.
NOT_FOUND_SHARE_THRESHOLD: float = 0.30

# Below this many entries the share is too noisy to act on.  At the healthy 3%
# base rate, 6 of 20 entries returning ``not_found`` has probability ~2e-5, so 20
# is the smallest batch where crossing 30% is decisive rather than unlucky.
MIN_BATCH_FOR_HEALTH_CHECK: int = 20


@dataclass(frozen=True)
class BatchHealth:
    """Batch-level sanity signal for one ``bibtex-check`` invocation.

    ``not_found`` means "every source was asked and none had this paper", which
    is only evidence of fabrication when the sources were actually reached.  A
    batch where most entries carry ``not_found`` or a transport-failure status is
    far more likely to be a broken network path than a bibliography of invented
    papers, so a caller that persists results should refuse to checkpoint it and
    re-run once connectivity is back.

    The ``"missing"`` sentinel is deliberately not counted here: it means
    bibtex-check produced no record at all (usually a timeout), which
    ``_run_bibtex_check_subprocess`` already logs on its own terms.

    Attributes:
        total: Number of entries in the batch.
        not_found: Entries whose status was ``not_found``.
        transport_error: Entries whose status was in
            ``TRANSPORT_FAILURE_STATUSES``.
        coverage_incomplete: Entries whose status was in
            ``COVERAGE_INCOMPLETE_STATUSES`` (sources reached but throttled or
            partially errored).
        threshold: No-evidence share above which the batch is suspect.
        min_batch_size: Smallest batch the share is evaluated on.
    """

    total: int
    not_found: int
    transport_error: int
    coverage_incomplete: int = 0
    threshold: float = NOT_FOUND_SHARE_THRESHOLD
    min_batch_size: int = MIN_BATCH_FOR_HEALTH_CHECK

    @property
    def not_found_share(self) -> float:
        """Fraction of the batch that came back ``not_found``."""
        return self.not_found / self.total if self.total else 0.0

    @property
    def transport_error_share(self) -> float:
        """Fraction of the batch that reported an explicit transport failure."""
        return self.transport_error / self.total if self.total else 0.0

    @property
    def no_evidence(self) -> int:
        """Entries the batch produced no usable database evidence for."""
        return self.not_found + self.transport_error + self.coverage_incomplete

    @property
    def no_evidence_share(self) -> float:
        """Fraction of the batch for which no source returned a usable verdict.

        Combines ``not_found``, transport failures and coverage-incomplete
        abstentions so the check reads the same signal whether or not the
        upstream tool has been upgraded to report failed lookups under their own
        status.
        """
        return self.no_evidence / self.total if self.total else 0.0

    @property
    def suspected_transport_failure(self) -> bool:
        """True when the batch looks like a broken lookup path, not a bad bibliography.

        This is the flag to gate checkpointing on.
        """
        return self.total >= self.min_batch_size and self.no_evidence_share > self.threshold

    def warning_message(self) -> str:
        """Operator-facing description of why the batch is not usable."""
        return (
            f"bibtex-check returned no database evidence for "
            f"{self.no_evidence}/{self.total} entries "
            f"({self.no_evidence_share:.1%}: {self.not_found} not_found, "
            f"{self.transport_error} transport errors, "
            f"{self.coverage_incomplete} coverage-incomplete), above the "
            f"{self.threshold:.0%} batch threshold. Healthy runs sit at 1-3% "
            f"not_found, so a share this high is almost always a broken lookup "
            f"path — a DNS, network or proxy outage, or sustained throttling — "
            f"rather than a bibliography of invented papers. Treat these "
            f"predictions as unusable: do not checkpoint this batch, and re-run "
            f"once the lookup path is healthy."
        )


def assess_batch_health(
    statuses: Iterable[str],
    *,
    threshold: float = NOT_FOUND_SHARE_THRESHOLD,
    min_batch_size: int = MIN_BATCH_FOR_HEALTH_CHECK,
) -> BatchHealth:
    """Score one batch of bibtex-check statuses for transport-failure poisoning.

    Args:
        statuses: Raw per-entry status strings, e.g. the values of the dict
            returned by ``run_bibtex_check_with_status``.
        threshold: No-evidence share above which the batch is suspect.
        min_batch_size: Smallest batch on which the share is evaluated.

    Returns:
        A ``BatchHealth`` record; ``suspected_transport_failure`` is the flag a
        caller gates checkpointing on.
    """
    status_list = list(statuses)
    return BatchHealth(
        total=len(status_list),
        not_found=sum(1 for s in status_list if s == "not_found"),
        transport_error=sum(1 for s in status_list if s in TRANSPORT_FAILURE_STATUSES),
        coverage_incomplete=sum(1 for s in status_list if s in COVERAGE_INCOMPLETE_STATUSES),
        threshold=threshold,
        min_batch_size=min_batch_size,
    )


#: Environment variable pinning which ``bibtex-check`` build to run.
#:
#: The wrapper used to invoke ``bibtex-check`` by name, so PATH order decided
#: which build answered -- and on a developer machine the first entry is an
#: EDITABLE install pointing at the bibtexupdater working tree, which changes
#: whenever anyone commits there. A pre-screening ablation run on 2026-09-04
#: was three hours into measuring "1.10.1" when the resolved binary turned out
#: to import from the checkout and report 1.3.1.dev18, with two commits landing
#: in that tree mid-run. Entries scored before and after were scored by
#: different code, and nothing in the output would have said so.
#:
#: Set this to an absolute path to pin a build, e.g. the pipx copy:
#:   HALLMARK_BIBTEX_CHECK_BIN=~/.local/pipx/venvs/bibtex-updater/bin/bibtex-check
BIBTEX_CHECK_BIN_ENV = "HALLMARK_BIBTEX_CHECK_BIN"

#: Environment variable overriding the per-service request rate.
#:
#: The wrapper drives ``--rate-limit 120`` where bibtex-check's own documented
#: baseline is 45 (scale 1.0), so a HALLMARK run paces roughly 2.7x harder than
#: the tool assumes. Under load that shows up as source lookups that never
#: complete, which bibtex-check correctly refuses to turn into verdicts -- and
#: an evaluation whose sources went dark measures availability, not the tool.
#: Pacing is therefore part of the run condition and belongs on the same
#: footing as the binary: settable, and recorded in the log.
BIBTEX_CHECK_RATE_ENV = "HALLMARK_BIBTEX_CHECK_RATE_LIMIT"

#: bibtex-check's exit code for "sources went dark during this run".
#:
#: Its own words, printed alongside it: "Treat this run as incomplete and
#: discard its could-not-verify verdicts." The wrapper used to log that at ERROR
#: and then parse the output and return predictions anyway, so a run the tool
#: had disowned became a scored result. On 2026-09-04, with dblp.org
#: unreachable, that produced bibtexupdater_dev_public.json at DR 0.8185 /
#: FPR 0.0312 / coverage 1.0000 from a run where 25.5% of entries never got a
#: complete set of source lookups.
#:
#: A source that never answered is not evidence a reference is absent, so those
#: entries' abstentions carry no information -- and nothing downstream could
#: tell them from real ones.
EXIT_SOURCE_OUTAGE = 5

#: Set to "1" to score a run bibtex-check reported as a source outage anyway.
#: Deliberately awkward: the numbers are not comparable to a healthy run.
ALLOW_OUTAGE_ENV = "HALLMARK_ALLOW_SOURCE_OUTAGE"

#: "285 of 1119 entries (25.5%) had at least one source lookup that did not
#: complete: dblp (275), openalex (26)"
_OUTAGE_RE = re.compile(
    r"(\d+) of (\d+) entries \(([\d.]+)%\) had at least one source lookup "
    r"that did not complete:\s*([^\n.]*)"
)


def parse_source_condition(output: str) -> dict[str, object] | None:
    """Extract bibtex-check's own source-availability report, if it made one.

    Availability moves outcomes, so it belongs in the result beside the numbers
    rather than only in a log a reader never sees.
    """
    # The final summary line wins: bibtex-check may report progressively.
    matches = list(_OUTAGE_RE.finditer(output))
    if not matches:
        return None
    match = matches[-1]
    per_source: dict[str, int] = {}
    for chunk in match.group(4).split(","):
        chunk = chunk.strip()
        if "(" in chunk and chunk.endswith(")"):
            name, _, count = chunk.partition("(")
            try:
                per_source[name.strip()] = int(count.rstrip(")"))
            except ValueError:
                continue
    return {
        "entries_with_incomplete_lookups": int(match.group(1)),
        "entries_total": int(match.group(2)),
        "incomplete_fraction": float(match.group(3)) / 100.0,
        "per_source_failures": per_source,
    }


def resolve_bibtex_check_rate_limit(default: int) -> int:
    """Per-service request rate for this run, from the environment or *default*."""
    raw = os.environ.get(BIBTEX_CHECK_RATE_ENV)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.error("%s is not an integer: %r -- using %d", BIBTEX_CHECK_RATE_ENV, raw, default)
        return default
    if value <= 0:
        logger.error(
            "%s must be positive, got %d -- using %d", BIBTEX_CHECK_RATE_ENV, value, default
        )
        return default
    return value


def resolve_bibtex_check_bin() -> str | None:
    """Absolute path of the ``bibtex-check`` build this run will use."""
    pinned = os.environ.get(BIBTEX_CHECK_BIN_ENV)
    if pinned:
        expanded = os.path.expanduser(pinned)
        if not Path(expanded).exists():
            logger.error("%s points at a missing file: %s", BIBTEX_CHECK_BIN_ENV, expanded)
            return None
        return expanded
    return shutil.which("bibtex-check")


_VERSION_PROBE = (
    "import importlib.metadata as md, bibtex_updater as m;"
    "print(getattr(m, '__version__', '?'));"
    "print(md.version('bibtex-updater'));"
    "print(m.__file__)"
)


def bibtex_check_version(binary: str | None = None) -> str | None:
    """Version of the bibtex_updater package backing *binary*.

    There is no ``--version`` flag, so ask the interpreter in the binary's own
    environment to import the package and report itself.

    It reports BOTH ``__version__`` and the packaging metadata, because for an
    editable install they disagree: the dist-info is frozen at whatever was
    installed when the editable link was made, while ``__version__`` tracks the
    source. The editable build on this machine answers 0.9.2 and 1.3.1.dev18
    from the same interpreter at the same moment, so no single number identifies
    it -- and the disagreement is itself the signal that a build is editable and
    therefore moves under you. When they agree, only one is reported.

    Returns None rather than raising: provenance must never break an evaluation.
    """
    binary = binary or resolve_bibtex_check_bin()
    if not binary:
        return None
    python = Path(binary).with_name("python")
    if not python.exists():
        python = Path(binary).with_name("python3")
    if not python.exists():
        return None
    try:
        out = subprocess.run(
            [str(python), "-c", _VERSION_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    source_version = lines[0]
    metadata_version = lines[1] if len(lines) > 1 else None
    if metadata_version and metadata_version != source_version:
        return (
            f"{source_version} (dist-info says {metadata_version}; "
            "they disagree, so this is an editable install)"
        )
    return source_version


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
    - ``"not_found"`` — **an exhaustive miss, and a positive claim about the
      lookup**: every source consulted completed successfully and returned no
      matching record.  A transport failure must NOT be reported under this
      status.  If any source failed for a technical reason — DNS resolution,
      connection refused or reset, TLS failure, timeout, 5xx, or a 429 that ended
      the attempt without an answer — the entry is an abstention, not an
      exhaustive miss, even when the sources that did answer found nothing.
      Conflating the two is what let a 2026-09-02 wifi outage produce
      ``not_found`` for 2,500 consecutive references with zero database evidence
      behind any of them.  Older tool builds do carry ``coverage_incomplete`` on
      such records; the wrapper reads that flag and downgrades the prediction to
      an abstention-style VALID, and downstream cascades keep routing
      ``"not_found"`` to Stage 2 as uncertain either way.  ``assess_batch_health``
      is the batch-level backstop for builds that report neither.
    - ``"network_error"`` — the lookup never reached a source (DNS, connection,
      TLS, or timeout).  An abstention: conservative VALID, routed to Stage 2,
      never HALLUCINATED.  Note that an HTTP error response is not this — a 4xx
      or 5xx proves the network came up — so a throttling 429 belongs under
      ``coverage_incomplete``, not here.
    - ``"coverage_incomplete"`` — the sources answered but the sweep was not
      exhaustive (a source was throttled or errored out).  An abstention on the
      same terms as ``network_error``.  In current tool builds this arrives as a
      boolean flag on a ``not_found`` record rather than as a status of its own;
      both shapes are handled.
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

    The batch is scored by ``assess_batch_health`` before returning and a
    poisoned batch is logged at WARNING level.  Callers that persist results
    should use ``run_bibtex_check_with_health`` instead and gate checkpointing on
    ``BatchHealth.suspected_transport_failure``.
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

    # Step 7: Batch-level sanity check. A batch that is mostly no-evidence is a
    # broken lookup path, not a bibliography of invented papers — say so loudly
    # so an operator watching the log can kill the run before it burns Stage 2
    # budget on entries no database was ever asked about.
    health = assess_batch_health(status_dict.values())
    if health.suspected_transport_failure:
        logger.warning(health.warning_message())

    return final_predictions, status_dict


def run_bibtex_check_with_health(
    entries: list[BlindEntry],
    extra_args: list[str] | None = None,
    timeout: float = 7200.0,
    rate_limit: int = 120,
    academic_only: bool = True,
    skip_prescreening: bool = False,
    **_kw: object,
) -> tuple[list[Prediction], dict[str, str], BatchHealth]:
    """Run bibtex-check and return predictions, statuses, and the batch health signal.

    Same behaviour as ``run_bibtex_check_with_status`` with the batch-level sanity
    check returned rather than only logged.  Use this from anything that persists
    results: when ``BatchHealth.suspected_transport_failure`` is True the batch
    carries no database evidence and must not be checkpointed, since a resumed run
    would then treat the poisoned verdicts as settled.

    Args:
        entries: Benchmark entries to verify.
        extra_args: Additional CLI arguments for bibtex-check.
        timeout: Timeout in seconds (default: 7200).
        rate_limit: API requests per minute (default: 120).
        academic_only: Skip web/book/working-paper checks (default: True).
        skip_prescreening: Skip pre-screening checks (default: False).

    Returns:
        A 3-tuple ``(predictions, status_dict, health)``.
    """
    predictions, status_dict = run_bibtex_check_with_status(
        entries,
        extra_args=extra_args,
        timeout=timeout,
        rate_limit=rate_limit,
        academic_only=academic_only,
        skip_prescreening=skip_prescreening,
    )
    return predictions, status_dict, assess_batch_health(status_dict.values())


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
        binary = resolve_bibtex_check_bin() or "bibtex-check"
        rate_limit = resolve_bibtex_check_rate_limit(rate_limit)
        # Say which build is answering. Without this the run is silent about the
        # single fact that decides whether its numbers are comparable to any
        # other run's, and PATH may be resolving an editable install.
        logger.info(
            "bibtex-check binary: %s (bibtex_updater %s)%s",
            binary,
            bibtex_check_version(binary) or "version unknown",
            ""
            if os.environ.get(BIBTEX_CHECK_BIN_ENV)
            else f" [unpinned; set {BIBTEX_CHECK_BIN_ENV} to pin]",
        )
        logger.info("bibtex-check rate limit: %d req/min per service", rate_limit)
        cmd = [
            binary,
            str(bib_path),
            "--jsonl",
            str(jsonl_path),
            "--rate-limit",
            str(rate_limit),
        ]
        if academic_only:
            cmd.append("--academic-only")
        # Pass S2 API key if available (bibtex-check supports --s2-api-key)
        s2_key = os.environ.get("S2_API_KEY")
        if s2_key:
            cmd.extend(["--s2-api-key", s2_key])
        if extra_args:
            cmd.extend(extra_args)

        # Run bibtex-check (the API key is masked — see redact_command)
        logger.info(f"Running: {redact_command(cmd)}")
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == EXIT_SOURCE_OUTAGE:
                condition = parse_source_condition(result.stdout + result.stderr)
                detail = ""
                if condition:
                    detail = (
                        f" {condition['entries_with_incomplete_lookups']} of "
                        f"{condition['entries_total']} entries "
                        f"({condition['incomplete_fraction']:.1%}) had an incomplete "
                        f"lookup: {condition['per_source_failures']}."
                    )
                if os.environ.get(ALLOW_OUTAGE_ENV) == "1":
                    logger.error(
                        "bibtex-check reported a SOURCE OUTAGE (exit %d).%s "
                        "Scoring it anyway because %s=1 -- these numbers are NOT "
                        "comparable to a healthy run.",
                        EXIT_SOURCE_OUTAGE,
                        detail,
                        ALLOW_OUTAGE_ENV,
                    )
                else:
                    raise SourceOutageError(
                        f"bibtex-check reported a source outage (exit "
                        f"{EXIT_SOURCE_OUTAGE}) and asked for the run to be "
                        f"discarded.{detail} A source that never answered is not "
                        "evidence a reference is absent, so this run cannot be "
                        "scored. Re-run when the sources are reachable, or set "
                        f"{ALLOW_OUTAGE_ENV}=1 to score it regardless."
                    )
            elif result.returncode not in (0, 2, 4):
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
