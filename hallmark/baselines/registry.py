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
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, Prediction

logger = logging.getLogger(__name__)

# Re-export BlindEntry so callers can import it from this module.
__all__ = [
    "BaselineInfo",
    "BlindEntry",
    "check_available",
    "get_registry",
    "list_baselines",
    "register",
    "run_baseline",
]


def _to_blind(entries: list[BenchmarkEntry]) -> list[BlindEntry]:
    """Convert BenchmarkEntry list to BlindEntry list, hiding ground-truth labels."""
    return [e.to_blind() for e in entries]


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
    # Environment variable name holding the API key (checked in check_available)
    env_var: str | None = None
    # Whether this baseline is free to run (no API costs)
    is_free: bool = True
    # Extra kwargs accepted by the runner
    runner_kwargs: dict[str, Any] = field(default_factory=dict)
    # How confidence scores are produced: "probabilistic" (LLM self-reported),
    # "heuristic" (rule-based), "binary" (fixed 0/1 values only)
    confidence_type: str = "heuristic"


# Global registry
_REGISTRY: dict[str, BaselineInfo] = {}


def register(info: BaselineInfo) -> None:
    """Register a baseline."""
    if info.name in _REGISTRY:
        logger.warning("Overwriting existing baseline %r", info.name)
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

    # Check required environment variable for API-key baselines
    if info.env_var:
        import os

        if info.env_var not in os.environ:
            return (
                False,
                f"Missing environment variable: {info.env_var}. "
                f"Set it with: export {info.env_var}=<your-key>",
            )

    return True, "OK"


def run_baseline(
    name: str,
    entries: list[BenchmarkEntry],
    split: str | None = None,
    **kwargs: Any,
) -> list[Prediction]:
    """Run a baseline by name.

    Converts BenchmarkEntry objects to BlindEntry before passing to the runner,
    so baseline implementations never see ground-truth labels.

    Args:
        name: Registered baseline name.
        entries: Benchmark entries to evaluate (ground-truth labels are hidden).
        split: Name of the split being evaluated. Used to detect self-referential
            evaluation (e.g., title_oracle on dev_public).
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
    if split is not None:
        merged_kwargs.setdefault("split", split)
    blind_entries = _to_blind(entries)
    return info.runner(blind_entries, **merged_kwargs)


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
    def _run_doi_presence_heuristic(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_presence import run_doi_presence_heuristic

        return run_doi_presence_heuristic(entries, **kw)

    register(
        BaselineInfo(
            name="doi_presence_heuristic",
            description="Trivial heuristic: predict HALLUCINATED when DOI is absent",
            runner=_run_doi_presence_heuristic,
            confidence_type="binary",
        )
    )

    # --- DOI-only (no extra deps) ---
    def _run_doi_only(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_only import run_doi_only

        return run_doi_only(entries, **kw)

    register(
        BaselineInfo(
            name="doi_only",
            description="Simple DOI resolution check via doi.org",
            runner=_run_doi_only,
            confidence_type="binary",
        )
    )

    # --- DOI-only (no pre-screening) ---
    def _run_doi_only_no_prescreening(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.doi_only import run_doi_only

        return run_doi_only(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="doi_only_no_prescreening",
            description="DOI resolution check without pre-screening layer (ablation baseline)",
            runner=_run_doi_only_no_prescreening,
            confidence_type="binary",
        )
    )

    # --- bibtex-updater ---
    def _run_bibtexupdater(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.bibtexupdater import run_bibtex_check

        return run_bibtex_check(entries, **kw)

    register(
        BaselineInfo(
            name="bibtexupdater",
            description="bibtex-check CLI: CrossRef, DBLP, Semantic Scholar",
            runner=_run_bibtexupdater,
            cli_commands=["bibtex-check"],
            confidence_type="heuristic",
        )
    )

    # --- bibtex-updater (no pre-screening) ---
    def _run_bibtexupdater_no_prescreening(
        entries: list[BlindEntry], **kw: Any
    ) -> list[Prediction]:
        from hallmark.baselines.bibtexupdater import run_bibtex_check

        return run_bibtex_check(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="bibtexupdater_no_prescreening",
            description="bibtex-check CLI without pre-screening layer (ablation baseline)",
            runner=_run_bibtexupdater_no_prescreening,
            cli_commands=["bibtex-check"],
            confidence_type="heuristic",
        )
    )

    # --- HaRC ---
    def _run_harc(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.harc import run_harc

        return run_harc(entries, **kw)

    register(
        BaselineInfo(
            name="harc",
            description="HaRC: Semantic Scholar, DBLP, Google Scholar, Open Library",
            runner=_run_harc,
            cli_commands=["harcx"],
            confidence_type="heuristic",
        )
    )

    # --- HaRC (no pre-screening) ---
    def _run_harc_no_prescreening(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.harc import run_harc

        return run_harc(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="harc_no_prescreening",
            description="HaRC without pre-screening layer (ablation baseline)",
            runner=_run_harc_no_prescreening,
            cli_commands=["harcx"],
            confidence_type="heuristic",
        )
    )

    # --- verify-citations ---
    def _run_verify_citations(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        return run_verify_citations(entries, **kw)

    register(
        BaselineInfo(
            name="verify_citations",
            description="verify-citations: arXiv, ACL, S2, DBLP, Google Scholar, DuckDuckGo",
            runner=_run_verify_citations,
            cli_commands=["verify-citations"],
            confidence_type="heuristic",
        )
    )

    # --- verify-citations (no pre-screening) ---
    def _run_verify_citations_no_prescreening(
        entries: list[BlindEntry], **kw: Any
    ) -> list[Prediction]:
        from hallmark.baselines.verify_citations_baseline import run_verify_citations

        return run_verify_citations(entries, skip_prescreening=True, **kw)

    register(
        BaselineInfo(
            name="verify_citations_no_prescreening",
            description="verify-citations without pre-screening layer (ablation baseline)",
            runner=_run_verify_citations_no_prescreening,
            cli_commands=["verify-citations"],
            confidence_type="heuristic",
        )
    )

    # --- LLM: OpenAI ---
    def _run_llm_openai(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.llm_verifier import verify_with_openai

        return verify_with_openai(entries, **kw)

    register(
        BaselineInfo(
            name="llm_openai",
            description="GPT-5.1 citation verification via OpenAI API",
            runner=_run_llm_openai,
            pip_packages=["openai"],
            requires_api_key=True,
            is_free=False,
            env_var="OPENAI_API_KEY",
            confidence_type="probabilistic",
        )
    )

    # --- LLM: Anthropic ---
    def _run_llm_anthropic(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
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
            env_var="ANTHROPIC_API_KEY",
            confidence_type="probabilistic",
        )
    )

    # --- LLM: OpenRouter (per-model baselines) ---
    from hallmark.baselines.llm_verifier import OPENROUTER_MODELS

    for friendly_name, model_id in OPENROUTER_MODELS.items():
        baseline_name = f"llm_openrouter_{friendly_name.replace('-', '_')}"

        def _make_openrouter_runner(
            mid: str,
        ) -> Callable[..., list[Prediction]]:
            def _run(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
                from hallmark.baselines.llm_verifier import verify_with_openrouter

                return verify_with_openrouter(entries, model=mid, **kw)

            return _run

        register(
            BaselineInfo(
                name=baseline_name,
                description=f"OpenRouter citation verification ({model_id})",
                runner=_make_openrouter_runner(model_id),
                pip_packages=["openai"],
                requires_api_key=True,
                is_free=False,
                env_var="OPENROUTER_API_KEY",
                confidence_type="probabilistic",
            )
        )

    # --- Title-match oracle (diagnostic baseline) ---
    # NOTE: This is NOT a legitimate detector.  It exploits label leakage by
    # looking up dev-split VALID titles.  Register it here so it appears in
    # the registry for transparency and paper reporting purposes.
    def _run_title_oracle(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.title_oracle import run_title_oracle
        from hallmark.dataset.loader import load_split

        split = kw.pop("split", None)
        if split == "dev_public":
            logger.warning(
                "Title oracle is self-referential when evaluated on dev_public. "
                "Results reflect label memorization, not detection capability."
            )

        reference_pool = load_split("dev_public")

        # Warn if input entries overlap significantly with the reference pool
        # (catches self-referential evaluation even without explicit split name)
        if split is None:
            input_keys = {e.bibtex_key for e in entries}
            ref_keys = {e.bibtex_key for e in reference_pool}
            overlap = input_keys & ref_keys
            if len(input_keys) > 0 and len(overlap) / len(input_keys) > 0.5:
                logger.warning(
                    "Title oracle: >50%% of input entries overlap with the dev_public reference "
                    "pool (%d/%d). Results reflect label memorization, not detection capability.",
                    len(overlap),
                    len(input_keys),
                )

        return run_title_oracle(entries, reference_pool, **kw)

    register(
        BaselineInfo(
            name="title_oracle",
            description=(
                "Title-match oracle (diagnostic — exploits perturbation structure). "
                "NOT a legitimate detector: uses dev VALID titles as a label look-up table."
            ),
            runner=_run_title_oracle,
            confidence_type="binary",
            # is_free=True (no API key, no extra packages)
        )
    )

    # --- Degenerate baselines (statistical reference points) ---
    def _run_random(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.degenerate import random_baseline

        return random_baseline(entries, **kw)

    register(
        BaselineInfo(
            name="random",
            description="Random baseline: predict HALLUCINATED with probability=prevalence (default 0.5)",
            runner=_run_random,
            confidence_type="binary",
        )
    )

    def _run_always_hallucinated(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.degenerate import always_hallucinated_baseline

        return always_hallucinated_baseline(entries)

    register(
        BaselineInfo(
            name="always_hallucinated",
            description="Constant baseline: predict HALLUCINATED for every entry",
            runner=_run_always_hallucinated,
            confidence_type="binary",
        )
    )

    def _run_always_valid(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.degenerate import always_valid_baseline

        return always_valid_baseline(entries)

    register(
        BaselineInfo(
            name="always_valid",
            description="Constant baseline: predict VALID for every entry",
            runner=_run_always_valid,
            confidence_type="binary",
        )
    )

    # --- Ensemble ---
    def _run_ensemble(entries: list[BlindEntry], **kw: Any) -> list[Prediction]:
        from hallmark.baselines.ensemble import ensemble_predict

        # Default: combine doi_only + bibtexupdater
        # Entries are already BlindEntry here, so call component runners directly
        # (bypassing run_baseline which would attempt a second to_blind() conversion).
        strategy_preds: dict[str, list[Prediction]] = {}
        for dep_name in ["doi_only", "bibtexupdater"]:
            avail, _ = check_available(dep_name)
            if avail:
                try:
                    dep_info = _REGISTRY[dep_name]
                    dep_kwargs = dict(dep_info.runner_kwargs)
                    strategy_preds[dep_name] = dep_info.runner(entries, **dep_kwargs)
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
            confidence_type="heuristic",
        )
    )


_register_builtins()
