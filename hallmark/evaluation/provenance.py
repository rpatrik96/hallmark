"""Provenance stamping for persisted evaluation results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from hallmark.dataset.schema import EvaluationResult

if TYPE_CHECKING:
    from hallmark.baselines.bibtexupdater import BibtexCheckRun

logger = logging.getLogger(__name__)


def stamp_provenance(
    result: EvaluationResult,
    split: str | None,
    data_dir: str | Path | None,
    version: str,
    baseline: str | None,
    tool_run: BibtexCheckRun | None = None,
) -> None:
    """Record the data revision, run time, and invoked external tool build.

    Args:
        result: The result to stamp.
        split: Split the result was computed on, or None to skip the data hash.
        data_dir: Root holding the versioned split files.
        version: Dataset version containing the split.
        baseline: Baseline that produced the predictions, named in log lines.
        tool_run: The bibtex-check run this result's predictions came from.
            A caller that dispatched the baseline itself passes it, so the
            stamp describes that run alone. A caller with no run to hand over,
            such as ``hallmark evaluate --predictions FILE``, leaves it None
            and the wrapper's own record for this thread answers instead.
    """
    result.run_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        from hallmark.baselines import bibtexupdater

        ran_bibtex_check = (
            tool_run.ran if tool_run is not None else bibtexupdater.ran_bibtex_check()
        )
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - best effort
        logger.error("Could not determine whether bibtex-check ran for %s: %s", baseline, exc)
        ran_bibtex_check = False

    if ran_bibtex_check:
        try:
            binary = bibtexupdater.resolve_bibtex_check_bin()
            tool_version = bibtexupdater.bibtex_check_version(binary)
            if tool_version:
                result.tool_version = f"bibtex-updater {tool_version}"
        except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            logger.error("Could not probe bibtex-check version for %s: %s", baseline, exc)

        try:
            result.source_condition = (
                tool_run.condition
                if tool_run is not None
                else bibtexupdater.last_source_condition()
            )
        except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            logger.error("Could not read bibtex-check source condition for %s: %s", baseline, exc)

    if not split:
        return

    try:
        from hallmark.dataset.loader import DEFAULT_DATA_DIR, SPLIT_PATHS
        from hallmark.evaluation.validate import compute_sha256

        data_root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        split_file = data_root / version / SPLIT_PATHS[split]
        if split_file.exists():
            result.split_sha256 = compute_sha256(split_file)
    except (KeyError, OSError) as exc:  # pragma: no cover - provenance is best effort
        logger.debug("Could not hash split file for provenance: %s", exc)
