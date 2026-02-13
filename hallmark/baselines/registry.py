"""Baseline registry: single source of truth for all HALLMARK baselines.

Provides discovery, availability checking, and dispatch for all baselines.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


@dataclass
class BaselineInfo:
    """Metadata and runner for a single baseline."""

    name: str
    description: str
    runner: Callable[..., list[Prediction]]
    # pip package(s) required — checked via import (empty = no extra deps)
    pip_packages: list[str] = field(default_factory=list)
    # CLI commands to check on PATH (for subprocess-based baselines)
    cli_commands: list[str] = field(default_factory=list)
    # Whether an API key or paid service is needed
    requires_api_key: bool = False
    # Whether this baseline is free to run (no API costs)
    is_free: bool = True
    # Extra kwargs accepted by the runner
    runner_kwargs: dict[str, Any] = field(default_factory=dict)


# Global registry
_REGISTRY: dict[str, BaselineInfo] = {}


def register(info: BaselineInfo) -> None:
    """Register a baseline."""
    _REGISTRY[info.name] = info


def get_registry() -> dict[str, BaselineInfo]:
    """Return the full registry (read-only copy)."""
    return dict(_REGISTRY)


def list_baselines(*, free_only: bool = False) -> list[str]:
    """List registered baseline names."""
    if free_only:
        return [name for name, info in _REGISTRY.items() if info.is_free]
    return list(_REGISTRY.keys())


def check_available(name: str) -> tuple[bool, str]:
    """Check if a baseline's dependencies are installed.

    For CLI-based baselines, checks that the command exists on PATH.
    For library-based baselines, checks that the Python module is importable.

    Returns:
        (available, message) tuple.
    """
    import shutil

    if name not in _REGISTRY:
        return False, f"Unknown baseline: {name}"

    info = _REGISTRY[name]

    # Check CLI commands on PATH (for subprocess-based baselines)
    missing_cmds = [cmd for cmd in info.cli_commands if shutil.which(cmd) is None]
    if missing_cmds:
        return (
            False,
            f"Missing CLI commands: {', '.join(missing_cmds)}. "
            f"Install the package that provides them.",
        )

    # Check importable Python packages
    missing_pkgs = []
    for pkg in info.pip_packages:
        module_name = _PIP_TO_MODULE.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        return (
            False,
            f"Missing packages: {', '.join(missing_pkgs)}. "
            f"Install with: pip install {' '.join(missing_pkgs)}",
        )
    return True, "OK"


def run_baseline(
    name: str,
    entries: list[BenchmarkEntry],
    **kwargs: Any,
) -> list[Prediction]:
    """Run a baseline by name.

    Args:
        name: Registered baseline name.
        entries: Benchmark entries to evaluate.
        **kwargs: Extra arguments forwarded to the runner.

    Returns:
        List of predictions.

    Raises:
        ValueError: If baseline is unknown.
        ImportError: If required packages are missing.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {', '.join(_REGISTRY.keys())}")

    available, msg = check_available(name)
    if not available:
        raise ImportError(msg)

    info = _REGISTRY[name]
    merged_kwargs = {**info.runner_kwargs, **kwargs}
    return info.runner(entries, **merged_kwargs)


# Mapping from pip package names to importable module names
_PIP_TO_MODULE: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
}


# ---------------------------------------------------------------------------
# Register all built-in baselines
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register all built-in baselines on module import."""

    # --- DOI-presence heuristic (no extra deps) ---
    def _run_doi_presence_heuristic(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_presence import run_doi_presence_heuristic

        return run_doi_presence_heuristic(entries, **kw)

    register(
        BaselineInfo(
            name="doi_presence_heuristic",
            description="Trivial heuristic: predict HALLUCINATED when DOI is absent",
            runner=_run_doi_presence_heuristic,
        )
    )

    # --- DOI-only (no extra deps) ---
    def _run_doi_only(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_only import run_doi_only

        return run_doi_only(entries, **kw)

    register(
        BaselineInfo(
            name="doi_only",
            description="Simple DOI resolution check via doi.org",
            runner=_run_doi_only,
        )
    )

    # --- DOI-only (no pre-screening) ---
    def _run_doi_only_no_prescreening(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_only import run_doi_only

        return run_doi_only(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="doi_only_no_prescreening",
            description="DOI resolution check without pre-screening layer (ablation baseline)",
            runner=_run_doi_only_no_prescreening,
        )
    )

    # --- bibtex-updater ---
    def _run_bibtexupdater(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.bibtexupdater import run_bibtex_check

        return run_bibtex_check(entries, **kw)

    register(
        BaselineInfo(
            name="bibtexupdater",
            description="bibtex-check CLI: CrossRef, DBLP, Semantic Scholar",
            runner=_run_bibtexupdater,
            cli_commands=["bibtex-check"],
        )
    )

    # --- bibtex-updater (no pre-screening) ---
    def _run_bibtexupdater_no_prescreening(
        entries: list[BenchmarkEntry], **kw: Any
    ) -> list[Prediction]:
        from hallmark.baselines.bibtexupdater import run_bibtex_check

        return run_bibtex_check(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="bibtexupdater_no_prescreening",
            description="bibtex-check CLI without pre-screening layer (ablation baseline)",
            runner=_run_bibtexupdater_no_prescreening,
            cli_commands=["bibtex-check"],
        )
    )

    # --- HaRC ---
    def _run_harc(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.harc import run_harc

        return run_harc(entries, **kw)

    register(
        BaselineInfo(
            name="harc",
            description="HaRC: Semantic Scholar, DBLP, Google Scholar, Open Library",
            runner=_run_harc,
            cli_commands=["harcx"],
        )
    )

    # --- HaRC (no pre-screening) ---
    def _run_harc_no_prescreening(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.harc import run_harc

        return run_harc(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="harc_no_prescreening",
            description="HaRC without pre-screening layer (ablation baseline)",
            runner=_run_harc_no_prescreening,
            cli_commands=["harcx"],
        )
    )

    # --- verify-citations ---
    def _run_verify_citations(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        return run_verify_citations(entries, **kw)

    register(
        BaselineInfo(
            name="verify_citations",
            description="verify-citations: arXiv, ACL, S2, DBLP, Google Scholar, DuckDuckGo",
            runner=_run_verify_citations,
            cli_commands=["verify-citations"],
        )
    )

    # --- verify-citations (no pre-screening) ---
    def _run_verify_citations_no_prescreening(
        entries: list[BenchmarkEntry], **kw: Any
    ) -> list[Prediction]:
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        return run_verify_citations(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="verify_citations_no_prescreening",
            description="verify-citations without pre-screening layer (ablation baseline)",
            runner=_run_verify_citations_no_prescreening,
            cli_commands=["verify-citations"],
        )
    )

    # --- LLM: OpenAI ---
    def _run_llm_openai(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.llm_verifier import verify_with_openai

        return verify_with_openai(entries, **kw)

    register(
        BaselineInfo(
            name="llm_openai",
            description="GPT-4o citation verification via OpenAI API",
            runner=_run_llm_openai,
            pip_packages=["openai"],
            requires_api_key=True,
            is_free=False,
        )
    )

    # --- LLM: Anthropic ---
    def _run_llm_anthropic(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.llm_verifier import verify_with_anthropic

        return verify_with_anthropic(entries, **kw)

    register(
        BaselineInfo(
            name="llm_anthropic",
            description="Claude citation verification via Anthropic API",
            runner=_run_llm_anthropic,
            pip_packages=["anthropic"],
            requires_api_key=True,
            is_free=False,
        )
    )

    # --- Ensemble ---
    def _run_ensemble(entries: list[BenchmarkEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.ensemble import ensemble_predict

        # Default: combine doi_only + bibtexupdater
        strategy_preds: dict[str, list[Prediction]] = {}
        for dep_name in ["doi_only", "bibtexupdater"]:
            avail, _ = check_available(dep_name)
            if avail:
                try:
                    strategy_preds[dep_name] = run_baseline(dep_name, entries)
                except Exception as e:
                    logger.warning(f"Ensemble: skipping {dep_name}: {e}")

        if not strategy_preds:
            logger.error("Ensemble: no component baselines available")
            return fallback_predictions(entries, reason="Ensemble: no components")

        return ensemble_predict(entries, strategy_preds, **kw)

    register(
        BaselineInfo(
            name="ensemble",
            description="Weighted vote across available free baselines",
            runner=_run_ensemble,
        )
    )


_register_builtins()
