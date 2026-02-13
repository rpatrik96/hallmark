"""Ranking tools using Plackett-Luce model for incomplete evaluation data.

This module implements ranking for citation verification tools inspired by ONEBench's
approach to handle heterogeneous and incomplete evaluation data. The Plackett-Luce
model allows ranking tools even when not all tools have been evaluated on all entries.

The implementation provides:
- Results matrix construction from predictions and ground truth
- Plackett-Luce ranking via the choix library (with fallback to mean scoring)
- Simple mean-score ranking for comparison
- Matrix serialization for reproducibility

Handling Incomplete Evaluations:
--------------------------------

When tools have partial coverage (e.g., due to API rate limits, failures, or selective
evaluation), the Plackett-Luce model enables fair ranking by:

1. **Pairwise Comparisons**: For each benchmark entry, compare tools that both have
   predictions for that entry. Tools are not penalized for missing entries.

2. **Correctness Scoring**: Each prediction receives a score:
   - 1.0 if label matches ground truth, 0.0 otherwise
   - For hallucinated entries: score weighted by confidence (rewards confident correct
     predictions, penalizes overconfident incorrect ones)
   - UNCERTAIN predictions treated as VALID (conservative, per evaluation protocol)

3. **ILSR Ranking**: Iterative Luce Spectral Ranking aggregates pairwise comparisons
   into global tool rankings, accounting for varying coverage and difficulty.

4. **Fallback**: If choix is unavailable or pairwise data is insufficient, falls back
   to mean-score ranking on covered entries.

Usage:
------
    # Single-tool evaluation (standard metrics)
    result = evaluate(entries, predictions, tool_name="my_tool")

    # Multi-tool ranking with partial coverage
    tool_predictions = {
        "tool_a": predictions_a,  # May cover different subsets
        "tool_b": predictions_b,
    }
    rankings = rank_tools(entries, tool_predictions, method="plackett_luce")

References:
    ONEBench: https://github.com/bethgelab/onebench
    Plackett-Luce model: Luce (1959), Plackett (1975)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


def build_results_matrix(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Build entries x tools matrix of correctness scores.

    For each entry-tool pair, computes a correctness score:
    - 1.0 if prediction matches ground truth label
    - 0.0 if prediction does not match
    - For hallucinated entries, weight by confidence (correctness * confidence)
    - None if tool has no prediction for this entry

    Args:
        entries: List of benchmark entries with ground truth labels
        tool_predictions: Dict mapping tool names to their predictions

    Returns:
        Tuple of (entry_keys, tool_names, matrix) where:
        - entry_keys: List of bibtex_key strings (row labels)
        - tool_names: List of tool name strings (column labels)
        - matrix: 2D list where matrix[i][j] is the correctness score of tool j
          on entry i, or None if missing
    """
    # Build index mappings
    entry_map = {entry.bibtex_key: entry for entry in entries}
    entry_keys = [entry.bibtex_key for entry in entries]
    tool_names = sorted(tool_predictions.keys())

    # Build prediction lookup: tool -> bibtex_key -> prediction
    pred_lookup: dict[str, dict[str, Prediction]] = {}
    for tool_name, predictions in tool_predictions.items():
        pred_lookup[tool_name] = {pred.bibtex_key: pred for pred in predictions}

    # Build matrix
    matrix: list[list[float | None]] = []
    for entry_key in entry_keys:
        entry = entry_map[entry_key]
        row: list[float | None] = []

        for tool_name in tool_names:
            if entry_key not in pred_lookup[tool_name]:
                row.append(None)
                continue

            pred = pred_lookup[tool_name][entry_key]

            # Treat UNCERTAIN as VALID (per evaluation protocol)
            pred_label = "VALID" if pred.label == "UNCERTAIN" else pred.label

            # Compute correctness
            is_correct = pred_label == entry.label
            correctness = 1.0 if is_correct else 0.0

            # Weight hallucinated entries by confidence
            # (Penalize overconfident incorrect predictions)
            if entry.label == "HALLUCINATED":
                # Correct: reward by confidence; incorrect: partial credit for low confidence
                score = pred.confidence if is_correct else 1.0 - pred.confidence
            else:
                # For valid entries, simple correctness
                score = correctness

            row.append(score)

        matrix.append(row)

    return entry_keys, tool_names, matrix


def rank_tools_plackett_luce(
    entry_keys: list[str],
    tool_names: list[str],
    matrix: list[list[float | None]],
    alpha: float = 0.01,
) -> list[tuple[str, float]]:
    """Rank tools using Plackett-Luce model via choix library.

    Converts the results matrix into pairwise comparisons and estimates tool
    parameters using iterative Luce spectral ranking (ILSR). Handles sparse and
    incomplete data gracefully.

    Falls back to mean-score ranking if choix is not installed.

    Args:
        entry_keys: List of entry identifiers (row labels)
        tool_names: List of tool names (column labels)
        matrix: Results matrix where matrix[i][j] is correctness of tool j on entry i
        alpha: L2 regularization parameter for ILSR (default: 0.01)

    Returns:
        List of (tool_name, score) tuples sorted descending by score.
        Scores are normalized probabilities summing to 1.
    """
    try:
        import choix
        import numpy as np
    except ImportError:
        logger.warning(
            "choix library not installed. Install with 'pip install choix'. "
            "Falling back to mean-score ranking."
        )
        return rank_tools_mean_score(entry_keys, tool_names, matrix)

    n_tools = len(tool_names)

    # Convert matrix to pairwise comparisons
    # For each entry, compare each pair of tools that have predictions
    comparisons: list[tuple[int, int]] = []

    for row in matrix:
        # Find tools with predictions for this entry
        available_tools = [(idx, score) for idx, score in enumerate(row) if score is not None]

        if len(available_tools) < 2:
            continue  # Need at least 2 tools to compare

        # Compare all pairs
        for i in range(len(available_tools)):
            for j in range(i + 1, len(available_tools)):
                idx_i, score_i = available_tools[i]
                idx_j, score_j = available_tools[j]

                # Winner is the tool with higher score
                # Add epsilon for tie-breaking
                if score_i > score_j + 1e-9:
                    # Tool i wins
                    comparisons.append((idx_i, idx_j))
                elif score_j > score_i + 1e-9:
                    # Tool j wins
                    comparisons.append((idx_j, idx_i))
                # Skip exact ties

    if not comparisons:
        logger.warning("No pairwise comparisons available. Falling back to mean-score ranking.")
        return rank_tools_mean_score(entry_keys, tool_names, matrix)

    # Estimate parameters using ILSR
    try:
        log_params = choix.ilsr_pairwise(n_tools, comparisons, alpha=alpha)
    except Exception as e:
        logger.error(f"ILSR failed: {e}. Falling back to mean-score ranking.")
        return rank_tools_mean_score(entry_keys, tool_names, matrix)

    # Convert log-parameters to normalized probabilities
    params = np.exp(log_params)
    params = params / params.sum()

    # Sort by score descending
    ranked = sorted(
        [(tool_names[idx], float(params[idx])) for idx in range(n_tools)],
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


def rank_tools_mean_score(
    entry_keys: list[str],
    tool_names: list[str],
    matrix: list[list[float | None]],
) -> list[tuple[str, float]]:
    """Rank tools by mean correctness score.

    Computes the average correctness score for each tool across all entries
    where it has predictions. Missing entries are ignored.

    Args:
        entry_keys: List of entry identifiers (row labels)
        tool_names: List of tool names (column labels)
        matrix: Results matrix where matrix[i][j] is correctness of tool j on entry i

    Returns:
        List of (tool_name, mean_score) tuples sorted descending by mean_score.
    """
    tool_scores: list[tuple[str, float]] = []

    for j, tool_name in enumerate(tool_names):
        scores: list[float] = [s for i in range(len(entry_keys)) if (s := matrix[i][j]) is not None]

        mean_score = 0.0 if not scores else sum(scores) / len(scores)

        tool_scores.append((tool_name, mean_score))

    # Sort descending
    tool_scores.sort(key=lambda x: x[1], reverse=True)

    return tool_scores


def rank_tools(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    method: str = "auto",
    alpha: float = 0.01,
) -> list[tuple[str, float]]:
    """Rank citation verification tools on benchmark entries.

    Main entry point for tool ranking. Builds the results matrix and applies
    the specified ranking method.

    Args:
        entries: List of benchmark entries with ground truth
        tool_predictions: Dict mapping tool names to their predictions
        method: Ranking method - "plackett_luce", "mean", or "auto" (default).
            "auto" uses Plackett-Luce if choix is available, else mean-score.
        alpha: L2 regularization for Plackett-Luce (default: 0.01)

    Returns:
        List of (tool_name, score) tuples sorted descending by score.

    Raises:
        ValueError: If method is not one of the supported options.
    """
    if method not in {"auto", "plackett_luce", "mean"}:
        raise ValueError(
            f"Unknown ranking method: {method}. Use 'auto', 'plackett_luce', or 'mean'."
        )

    # Build results matrix
    entry_keys, tool_names, matrix = build_results_matrix(entries, tool_predictions)

    # Apply ranking method
    if method == "mean":
        return rank_tools_mean_score(entry_keys, tool_names, matrix)
    elif method == "plackett_luce":
        return rank_tools_plackett_luce(entry_keys, tool_names, matrix, alpha=alpha)
    else:  # auto
        # Try Plackett-Luce, will auto-fallback to mean if choix unavailable
        return rank_tools_plackett_luce(entry_keys, tool_names, matrix, alpha=alpha)


def save_results_matrix(
    entry_keys: list[str],
    tool_names: list[str],
    matrix: list[list[float | None]],
    path: str | Path,
) -> None:
    """Save results matrix as CSV for reproducibility.

    The CSV has entry keys as the first column and tool names as column headers.

    Args:
        entry_keys: List of entry identifiers (row labels)
        tool_names: List of tool names (column labels)
        matrix: Results matrix where matrix[i][j] is correctness of tool j on entry i
        path: Output CSV file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["entry_key", *tool_names])

        # Data rows
        for i, entry_key in enumerate(entry_keys):
            row = [entry_key]
            for j in range(len(tool_names)):
                value = matrix[i][j]
                row.append("" if value is None else f"{value:.4f}")
            writer.writerow(row)

    logger.info(f"Saved results matrix to {path}")


def load_results_matrix(
    path: str | Path,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Load results matrix from CSV.

    Args:
        path: Input CSV file path (created by save_results_matrix)

    Returns:
        Tuple of (entry_keys, tool_names, matrix)
    """
    path = Path(path)

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        # Read header
        header = next(reader)
        tool_names = header[1:]  # Skip "entry_key" column

        # Read data rows
        entry_keys: list[str] = []
        matrix: list[list[float | None]] = []

        for row in reader:
            entry_keys.append(row[0])
            matrix_row: list[float | None] = []

            for value in row[1:]:
                if value == "":
                    matrix_row.append(None)
                else:
                    matrix_row.append(float(value))

            matrix.append(matrix_row)

    logger.info(
        f"Loaded results matrix from {path} ({len(entry_keys)} entries x {len(tool_names)} tools)"
    )

    return entry_keys, tool_names, matrix
