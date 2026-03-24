"""Factor analysis for HALLMARK benchmark score matrices.

Analyzes whether tool performance is driven by a single dominant factor
(e.g., 'is it an LLM?') or multiple orthogonal capabilities, following
Hardt's observation that benchmark-model score matrices are often low-rank.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from hallmark.dataset.schema import (
    BenchmarkEntry,
    Prediction,
)

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """Result of PCA on the tool x type score matrix."""

    tool_names: list[str]
    type_names: list[str]
    explained_variance_ratio: list[float]
    effective_rank: int  # PCs needed for 90% variance
    pc1_loadings: dict[str, float]  # type_name -> PC1 loading
    pc1_tool_scores: dict[str, float]  # tool_name -> PC1 score
    total_variance_explained_by_pc1: float


def compute_score_matrix(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
) -> tuple[list[str], list[str], list[list[float]]]:
    """Build tools x hallucination_types detection rate matrix.

    For each tool and each hallucination type, compute detection rate
    (fraction of hallucinated entries correctly identified).
    Only includes hallucinated entries (VALID entries excluded).

    Returns:
        (tool_names, type_names, matrix) where matrix[i][j] is tool i's
        detection rate on hallucination type j.
    """
    # Collect hallucinated entries only
    hallucinated = [e for e in entries if e.label == "HALLUCINATED"]

    # Get all unique hallucination types present, in sorted order
    type_set: set[str] = set()
    for e in hallucinated:
        if e.hallucination_type is not None:
            type_set.add(e.hallucination_type)
    type_names = sorted(type_set)

    # Group entries by hallucination type
    type_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for e in hallucinated:
        if e.hallucination_type is not None:
            type_entries[e.hallucination_type].append(e)

    tool_names = sorted(tool_predictions.keys())

    matrix: list[list[float]] = []
    for tool_name in tool_names:
        pred_map = {p.bibtex_key: p for p in tool_predictions[tool_name]}
        row: list[float] = []
        for h_type in type_names:
            type_e = type_entries[h_type]
            if not type_e:
                row.append(0.0)
                continue
            correct = 0
            for entry in type_e:
                pred = pred_map.get(entry.bibtex_key)
                if pred is not None and pred.label == "HALLUCINATED":
                    correct += 1
            row.append(correct / len(type_e))
        matrix.append(row)

    return tool_names, type_names, matrix


def pca_analysis(
    tool_names: list[str],
    type_names: list[str],
    score_matrix: list[list[float]],
    variance_threshold: float = 0.90,
) -> PCAResult:
    """PCA on the tool x type score matrix.

    Uses numpy SVD. numpy is an optional dependency (ranking extra).

    Args:
        tool_names: row labels
        type_names: column labels
        score_matrix: tools x types detection rates
        variance_threshold: cumulative variance threshold for effective_rank

    Returns:
        PCAResult with explained variance, effective rank, and PC1 interpretation.

    Raises:
        ImportError: if numpy is not installed
        ValueError: if matrix has fewer than 2 tools or 2 types
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for PCA analysis. Install it with: pip install numpy"
        ) from exc

    n_tools = len(tool_names)
    n_types = len(type_names)

    if n_tools < 2:
        raise ValueError(f"PCA requires at least 2 tools, got {n_tools}")
    if n_types < 2:
        raise ValueError(f"PCA requires at least 2 hallucination types, got {n_types}")

    # Convert to numpy array (shape: n_tools x n_types)
    X = np.array(score_matrix, dtype=float)

    # Center by subtracting column means
    col_means = X.mean(axis=0)
    X_centered = X - col_means

    # SVD: X_centered = U * S * Vt
    # U: (n_tools, k), S: (k,), Vt: (k, n_types)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained variance ratio from singular values
    variance = S**2
    total_variance = variance.sum()
    if total_variance == 0.0:
        explained_ratio = [0.0] * len(S)
    else:
        explained_ratio = (variance / total_variance).tolist()

    # Effective rank: number of PCs to reach variance_threshold cumulative variance
    cumulative = 0.0
    effective_rank = len(explained_ratio)
    for i, ev in enumerate(explained_ratio):
        cumulative += ev
        if cumulative >= variance_threshold:
            effective_rank = i + 1
            break

    # PC1 loadings: right singular vector (columns of V = rows of Vt)
    pc1_loadings = {type_names[j]: float(Vt[0, j]) for j in range(n_types)}

    # PC1 tool scores: left singular vector scaled by singular value
    pc1_tool_scores = {tool_names[i]: float(U[i, 0] * S[0]) for i in range(n_tools)}

    total_pc1 = explained_ratio[0] if explained_ratio else 0.0

    return PCAResult(
        tool_names=list(tool_names),
        type_names=list(type_names),
        explained_variance_ratio=explained_ratio,
        effective_rank=effective_rank,
        pc1_loadings=pc1_loadings,
        pc1_tool_scores=pc1_tool_scores,
        total_variance_explained_by_pc1=total_pc1,
    )


def tier_stratified_pca(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    variance_threshold: float = 0.90,
) -> dict[int, PCAResult]:
    """Run PCA separately for each difficulty tier.

    Checks whether the tier structure adds orthogonal signal.

    Returns:
        dict mapping tier (1, 2, 3) -> PCAResult
    """
    results: dict[int, PCAResult] = {}

    for tier in (1, 2, 3):
        tier_entries = [
            e for e in entries if e.label == "HALLUCINATED" and e.difficulty_tier == tier
        ]
        # Include VALID entries too (they don't appear in score matrix but
        # compute_score_matrix filters them out anyway)
        all_entries_for_tier = tier_entries + [e for e in entries if e.label == "VALID"]

        tool_names, type_names, matrix = compute_score_matrix(
            all_entries_for_tier, tool_predictions
        )

        if len(tool_names) < 2 or len(type_names) < 2:
            logger.warning(
                "Tier %d: not enough tools (%d) or types (%d) for PCA, skipping",
                tier,
                len(tool_names),
                len(type_names),
            )
            continue

        try:
            result = pca_analysis(tool_names, type_names, matrix, variance_threshold)
            results[tier] = result
        except (ValueError, ImportError) as exc:
            logger.warning("Tier %d PCA failed: %s", tier, exc)

    return results
