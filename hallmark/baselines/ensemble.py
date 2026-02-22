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
            total_weight -= w  # Missing prediction: exclude from vote entirely
            continue

        # Confidence-weighted voting: each tool's vote is scaled by its confidence.
        # hall_fraction = sum(w_i * conf_i for HALL) / sum(w_i) is a confidence-weighted
        # average, not a majority vote. The threshold thus represents a minimum
        # confidence-weighted proportion, not a vote count.
        if pred.label == "UNCERTAIN":
            # UNCERTAIN = tool couldn't decide — exclude from vote entirely
            total_weight -= w  # Don't count this tool's weight
            reasons.append(f"{name}: UNCERTAIN ({pred.confidence:.2f})")
        elif pred.label == "HALLUCINATED":
            hall_weight += w * pred.confidence
            reasons.append(f"{name}: HALL ({pred.confidence:.2f})")
        else:
            valid_weight += w * pred.confidence
            reasons.append(f"{name}: VALID ({pred.confidence:.2f})")

        total_time += pred.wall_clock_seconds
        total_api_calls += pred.api_calls

    if total_weight == 0:
        return Prediction(
            bibtex_key=entry.bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason=f"Ensemble ({config.method}): all strategies missing or UNCERTAIN; "
            + "; ".join(reasons),
            wall_clock_seconds=total_time,
            api_calls=total_api_calls,
        )

    hall_fraction = hall_weight / total_weight
    is_hallucinated = hall_fraction > config.voting_threshold

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
        if pred and pred.label != "UNCERTAIN" and pred.confidence > best_confidence:
            best_confidence = pred.confidence
            best_pred = pred

    if best_pred is None:
        return Prediction(
            bibtex_key=entry.bibtex_key,
            label="UNCERTAIN",
            confidence=0.5,
            reason="Ensemble (max_confidence): all strategies missing or UNCERTAIN",
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

        if pred.label == "UNCERTAIN":
            # UNCERTAIN = tool couldn't decide — exclude from mean computation entirely
            reasons.append(f"{name}: UNCERTAIN ({pred.confidence:.2f})")
            total_time += pred.wall_clock_seconds
            total_api_calls += pred.api_calls
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
            label="UNCERTAIN",
            confidence=0.5,
            reason="Ensemble (mean_confidence): all strategies missing or UNCERTAIN",
        )

    # Compute per-prediction hallucination scores on a common [0, 1] scale:
    # hall_score = confidence      if HALLUCINATED
    # hall_score = 1 - confidence  if VALID
    # Then threshold the mean at 0.5.
    hall_scores = hall_confidences + [1.0 - c for c in valid_confidences]
    mean_hall_score = sum(hall_scores) / len(hall_scores)
    is_hallucinated = mean_hall_score > 0.5
    confidence = mean_hall_score if is_hallucinated else (1.0 - mean_hall_score)

    return Prediction(
        bibtex_key=entry.bibtex_key,
        label="HALLUCINATED" if is_hallucinated else "VALID",
        confidence=confidence,
        reason="Ensemble (mean_confidence): " + "; ".join(reasons),
        wall_clock_seconds=total_time,
        api_calls=total_api_calls,
    )
