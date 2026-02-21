"""Ensemble baseline: combine multiple verification strategies.

Combines bibtex-updater + DOI-only + optional LLM verification
using configurable voting/weighting schemes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble baseline."""

    # Strategy weights (higher = more influence)
    weights: dict[str, float] | None = None
    # Voting threshold: fraction of strategies that must agree on HALLUCINATED
    voting_threshold: float = 0.5
    # Method: "weighted_vote" or "max_confidence" or "mean_confidence"
    method: Literal["weighted_vote", "max_confidence", "mean_confidence"] = "weighted_vote"


def ensemble_predict(
    entries: list[BlindEntry],
    strategy_predictions: dict[str, list[Prediction]],
    config: EnsembleConfig | None = None,
) -> list[Prediction]:
    """Combine predictions from multiple strategies.

    Args:
        entries: Benchmark entries.
        strategy_predictions: Dict mapping strategy name -> list of predictions.
        config: Ensemble configuration.

    Returns:
        Combined predictions.
    """
    config = config or EnsembleConfig()

    # Default equal weights
    if config.weights is None:
        config.weights = {name: 1.0 for name in strategy_predictions}

    # Index predictions by bibtex_key for each strategy
    indexed: dict[str, dict[str, Prediction]] = {}
    for name, preds in strategy_predictions.items():
        indexed[name] = {p.bibtex_key: p for p in preds}

    predictions = []
    for entry in entries:
        if config.method == "weighted_vote":
            pred = _weighted_vote(entry, indexed, config)
        elif config.method == "max_confidence":
            pred = _max_confidence(entry, indexed, config)
        elif config.method == "mean_confidence":
            pred = _mean_confidence(entry, indexed, config)
        else:
            raise ValueError(f"Unknown ensemble method: {config.method}")
        predictions.append(pred)

    return predictions


def _weighted_vote(
    entry: BlindEntry,
    indexed: dict[str, dict[str, Prediction]],
    config: EnsembleConfig,
) -> Prediction:
    """Weighted voting: each strategy votes, weighted by config."""
    hall_weight = 0.0
    valid_weight = 0.0
    total_weight = 0.0
    reasons = []
    total_time = 0.0
    total_api_calls = 0

    assert config.weights is not None
    for name, pred_map in indexed.items():
        pred = pred_map.get(entry.bibtex_key)
        w = config.weights.get(name, 1.0)
        total_weight += w

        if pred is None:
            valid_weight += w  # Missing = assume valid (unweighted by confidence)
            continue

        # Confidence-weighted voting: each tool's vote is scaled by its confidence.
        # hall_fraction = sum(w_i * conf_i for HALL) / sum(w_i) is a confidence-weighted
        # average, not a majority vote. The threshold thus represents a minimum
        # confidence-weighted proportion, not a vote count.
        if pred.label == "HALLUCINATED":
            hall_weight += w * pred.confidence
            reasons.append(f"{name}: HALL ({pred.confidence:.2f})")
        else:
            valid_weight += w * pred.confidence
            reasons.append(f"{name}: VALID ({pred.confidence:.2f})")

        total_time += pred.wall_clock_seconds
        total_api_calls += pred.api_calls

    hall_fraction = hall_weight / total_weight if total_weight > 0 else 0.0
    is_hallucinated = hall_fraction >= config.voting_threshold

    confidence = hall_fraction if is_hallucinated else (1.0 - hall_fraction)
    confidence = max(0.0, min(1.0, confidence))

    return Prediction(
        bibtex_key=entry.bibtex_key,
        label="HALLUCINATED" if is_hallucinated else "VALID",
        confidence=confidence,
        reason=f"Ensemble ({config.method}): " + "; ".join(reasons),
        wall_clock_seconds=total_time,
        api_calls=total_api_calls,
    )


def _max_confidence(
    entry: BlindEntry,
    indexed: dict[str, dict[str, Prediction]],
    config: EnsembleConfig,
) -> Prediction:
    """Take the prediction with highest confidence."""
    best_pred = None
    best_confidence = -1.0

    for _name, pred_map in indexed.items():
        pred = pred_map.get(entry.bibtex_key)
        if pred and pred.confidence > best_confidence:
            best_confidence = pred.confidence
            best_pred = pred

    if best_pred is None:
        return Prediction(
            bibtex_key=entry.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason="No predictions available",
        )

    return Prediction(
        bibtex_key=entry.bibtex_key,
        label=best_pred.label,
        confidence=best_pred.confidence,
        reason=f"Ensemble (max_confidence): {best_pred.reason}",
        wall_clock_seconds=best_pred.wall_clock_seconds,
        api_calls=best_pred.api_calls,
    )


def _mean_confidence(
    entry: BlindEntry,
    indexed: dict[str, dict[str, Prediction]],
    config: EnsembleConfig,
) -> Prediction:
    """Average confidence across strategies, threshold at 0.5."""
    hall_confidences = []
    valid_confidences = []
    reasons = []
    total_time = 0.0
    total_api_calls = 0

    for name, pred_map in indexed.items():
        pred = pred_map.get(entry.bibtex_key)
        if pred is None:
            continue

        if pred.label == "HALLUCINATED":
            hall_confidences.append(pred.confidence)
        else:
            valid_confidences.append(pred.confidence)

        reasons.append(f"{name}: {pred.label} ({pred.confidence:.2f})")
        total_time += pred.wall_clock_seconds
        total_api_calls += pred.api_calls

    if not hall_confidences and not valid_confidences:
        return Prediction(
            bibtex_key=entry.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason="No predictions available",
        )

    mean_hall = sum(hall_confidences) / len(hall_confidences) if hall_confidences else 0.0
    mean_valid = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
    n_total = len(hall_confidences) + len(valid_confidences)
    if n_total > 0:
        weighted_hall = (len(hall_confidences) / n_total) * mean_hall
        weighted_valid = (len(valid_confidences) / n_total) * mean_valid
        is_hallucinated = weighted_hall > weighted_valid
    else:
        is_hallucinated = False
    confidence = mean_hall if is_hallucinated else mean_valid

    return Prediction(
        bibtex_key=entry.bibtex_key,
        label="HALLUCINATED" if is_hallucinated else "VALID",
        confidence=confidence,
        reason="Ensemble (mean_confidence): " + "; ".join(reasons),
        wall_clock_seconds=total_time,
        api_calls=total_api_calls,
    )
