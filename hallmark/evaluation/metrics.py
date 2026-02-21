"""Evaluation metrics for HALLMARK.

Primary: Detection Rate (DR), False Positive Rate (FPR), F1-Hallucination, Tier-weighted F1, ECE.
Secondary: detect@k, temporal robustness, cost efficiency, per-type metrics, subtest accuracy.
Inspired by HumanEval (multi-criteria, pass@k) and ONEBench (sample-level atomic eval).

Evaluation Protocol:
--------------------

1. **Prediction Labels**:
   - Tools must return VALID or HALLUCINATED predictions
   - UNCERTAIN is accepted as a valid label — see UNCERTAIN Protocol below
   - Missing predictions are treated as VALID (conservative default)

2. **UNCERTAIN Protocol**:
   - UNCERTAIN predictions are excluded from all classification metrics (DR, FPR, F1, TW-F1).
     The tool did respond, but with insufficient confidence to make a definitive call.
   - UNCERTAIN predictions COUNT toward coverage (the tool processed the entry).
   - UNCERTAIN predictions are excluded from AUROC and AUPRC (same as before).
   - UNCERTAIN predictions are excluded from ECE (no reliable confidence signal).
   - ``num_uncertain`` in EvaluationResult tracks how many were skipped.

3. **Pre-screening**:
   - Baseline wrappers may include pre-screening (lightweight local checks before external APIs)
   - Pre-screening results are included in the tool's predictions and reported transparently
   - The `reason` field indicates pre-screening with `[Pre-screening override]` prefix

4. **Handling Incomplete Evaluations**:
   - Tools may evaluate only a subset of benchmark entries (e.g., due to API limits)
   - For single-tool evaluation: metrics computed only on covered entries
   - For multi-tool ranking: use Plackett-Luce model (see hallmark.evaluation.ranking)
     to fairly rank tools with heterogeneous coverage

5. **Metrics Computation**:
   - UNCERTAIN predictions excluded from classification metrics (see UNCERTAIN Protocol)
   - Missing predictions treated as VALID (conservative default)
   - Tier-weighted F1: harder hallucinations (Tier 3) weighted 3x vs Tier 1
   - ECE (Expected Calibration Error): measures confidence calibration
   - Per-type metrics: detection rate by hallucination type
   - Subtest accuracy: accuracy on individual verification checks (DOI, title, authors, etc.)

6. **Cost Tracking**:
   - wall_clock_seconds: total time per prediction
   - api_calls: number of external API calls made
   - cost_efficiency: entries evaluated per second
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from hallmark.dataset.schema import BenchmarkEntry, EvaluationResult, Prediction, is_canary_entry

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix for hallucination detection."""

    tp: int = 0  # Correctly detected hallucinations
    fp: int = 0  # Valid entries incorrectly flagged as hallucinated
    tn: int = 0  # Correctly identified valid entries
    fn: int = 0  # Missed hallucinations

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Same as Detection Rate (DR)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient — prevalence-invariant metric.

        MCC is insensitive to class imbalance, making it suitable for
        cross-split comparisons where hallucination prevalence differs
        (e.g., dev_public=54.3% vs test_public=64.8%).
        """
        denom_sq = (
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        if denom_sq == 0:
            return 0.0
        return float((self.tp * self.tn - self.fp * self.fn) / denom_sq**0.5)

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 across both classes (HALLUCINATED and VALID).

        Averages the per-class F1 scores, giving equal weight to both classes
        regardless of their prevalence.
        """
        f1_hall = self.f1
        # F1 for VALID class (treating VALID as positive)
        prec_valid = self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0.0
        rec_valid = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        f1_valid = (
            2 * prec_valid * rec_valid / (prec_valid + rec_valid)
            if (prec_valid + rec_valid) > 0
            else 0.0
        )
        return (f1_hall + f1_valid) / 2


def build_confusion_matrix(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> ConfusionMatrix:
    """Build confusion matrix from entries and predictions.

    UNCERTAIN Protocol: UNCERTAIN predictions are excluded from the confusion matrix
    entirely — they do not contribute to TP, FP, TN, or FN. They count toward
    coverage (the tool did respond) but not toward classification metrics.
    Missing predictions are treated as VALID (conservative default).

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).
    """
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}
    cm = ConfusionMatrix()
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        if pred is None:
            # Missing prediction treated as "VALID" (conservative)
            if entry.label == "HALLUCINATED":
                cm.fn += 1
            else:
                cm.tn += 1
            continue

        # UNCERTAIN excluded from classification metrics — skip this entry entirely
        if pred.label == "UNCERTAIN":
            continue

        if entry.label == "HALLUCINATED":
            if pred.label == "HALLUCINATED":
                cm.tp += 1
            else:
                cm.fn += 1
        else:
            if pred.label == "HALLUCINATED":
                cm.fp += 1
            else:
                cm.tn += 1
    return cm


def tier_weighted_f1(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
    tier_weights: dict[int, float] | None = None,
) -> float:
    """Compute F1 weighted by difficulty tier (Tier 3 worth 3x Tier 1).

    Each hallucinated entry contributes to F1 proportionally to its tier weight.
    UNCERTAIN predictions are excluded (not counted toward the confusion matrix).

    FP weighting note:
    - False positives (valid entries incorrectly flagged) are always weighted at 1.0,
      regardless of the tier_weights scheme. This is intentional: valid entries have
      no difficulty tier, so there is no principled basis for tier-weighting them.
    - Consequence: precision is independent of tier weighting; only recall is
      tier-weighted. This differs from standard macro-weighted F1, where both
      precision and recall are uniformly weighted across classes.
    - Use `tier_weight_sensitivity` to assess how robust TW-F1 scores are to
      different weighting schemes.

    Entries with difficulty_tier=None are assigned to tier 1 (the default tier).
    """
    # F-15: accept list as well as dict
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}

    if tier_weights is None:
        tier_weights = {1: 1.0, 2: 2.0, 3: 3.0}

    weighted_tp = 0.0
    weighted_fn = 0.0
    weighted_fp = 0.0

    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        tier = entry.difficulty_tier or 1
        w = tier_weights.get(tier, 1.0)

        # UNCERTAIN predictions are excluded from classification metrics entirely.
        # Missing predictions are treated as VALID (conservative default).
        if pred is not None and pred.label == "UNCERTAIN":
            continue

        pred_label = pred.label if pred is not None else "VALID"

        if entry.label == "HALLUCINATED":
            if pred_label == "HALLUCINATED":
                weighted_tp += w
            else:
                weighted_fn += w
        else:
            # FP weighting: VALID entries are penalized uniformly (weight 1.0)
            # regardless of weighting scheme, affecting only precision, not recall.
            if pred_label == "HALLUCINATED":
                weighted_fp += 1.0

    precision = (
        weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    )
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def union_recall_at_k(
    entries: list[BenchmarkEntry],
    predictions_per_strategy: list[dict[str, Prediction]],
    k: int | None = None,
    strategy_costs: list[float] | None = None,
) -> dict[int, float]:
    """Compute the fraction of hallucinations detected by the union of the first k strategies.

    A hallucination is "detected" if ANY of the first k strategies flags it as HALLUCINATED.
    UNCERTAIN predictions are treated as VALID (conservative, not counted as detections).

    Note on naming: this metric is sometimes called detect@k, but that name is misleading
    because it implies an analogy with HumanEval's pass@k. HumanEval's pass@k uses an
    unbiased stochastic estimator (sampling without replacement from n attempts); this metric
    is deterministic and order-dependent — results change if you reorder the strategy list.
    Use this function directly when you want the deterministic union-recall formulation.

    Canonical ordering: strategies should be sorted by **ascending cost** (cheapest first)
    so that the union-recall curve shows the cost-effective frontier. When strategies have
    equal cost, order alphabetically for reproducibility.

    Args:
        entries: Benchmark entries.
        predictions_per_strategy: List of prediction dicts, one per verification strategy.
            If ``strategy_costs`` is not provided, must already be ordered by ascending cost.
        k: If provided, compute only for this k. Otherwise compute for k=1..len(strategies).
        strategy_costs: Optional list of floats (one per strategy) giving the cost of each
            strategy. If provided, strategies are sorted by ascending cost before computing
            the union-recall curve. If not provided, the given order is used as-is and a
            debug warning is emitted to remind callers to pass strategies in ascending cost
            order.

    Returns:
        Dict mapping k -> fraction of hallucinations detected by at least one of k strategies.
    """
    if strategy_costs is not None:
        if len(strategy_costs) != len(predictions_per_strategy):
            raise ValueError(
                f"strategy_costs length ({len(strategy_costs)}) must match "
                f"predictions_per_strategy length ({len(predictions_per_strategy)})"
            )
        # Sort strategies by ascending cost
        sorted_pairs = sorted(
            zip(strategy_costs, predictions_per_strategy, strict=True), key=lambda x: x[0]
        )
        predictions_per_strategy = [p for _, p in sorted_pairs]
    else:
        logger.debug(
            "union_recall_at_k: no strategy_costs provided. Using given order. "
            "Pass strategy_costs for reproducible cost-ordered union-recall curves."
        )

    hallucinated = [e for e in entries if e.label == "HALLUCINATED"]
    if not hallucinated:
        return {}

    n_strategies = len(predictions_per_strategy)
    ks = [k] if k is not None else list(range(1, n_strategies + 1))
    results = {}

    for ki in ks:
        if ki > n_strategies:
            break
        detected = 0
        for entry in hallucinated:
            for strategy_preds in predictions_per_strategy[:ki]:
                pred = strategy_preds.get(entry.bibtex_key)
                # UNCERTAIN treated as VALID (not a detection)
                if pred is not None and pred.label == "HALLUCINATED":
                    detected += 1
                    break
        results[ki] = detected / len(hallucinated)

    return results


def detect_at_k(
    entries: list[BenchmarkEntry],
    predictions_per_strategy: list[dict[str, Prediction]],
    k: int | None = None,
) -> dict[int, float]:
    """Deprecated alias for `union_recall_at_k`.

    Computes the fraction of hallucinations detected by the union of the first k strategies.
    Unlike HumanEval's pass@k (which uses an unbiased stochastic estimator), this metric
    is deterministic and order-dependent.

    Use `union_recall_at_k` directly for new code.

    Args:
        entries: Benchmark entries.
        predictions_per_strategy: List of prediction dicts, one per verification strategy.
        k: If provided, compute only for this k. Otherwise compute for k=1..len(strategies).

    Returns:
        Dict mapping k -> fraction of hallucinations detected by at least one of k strategies.
    """
    return union_recall_at_k(entries, predictions_per_strategy, k)


def per_tier_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[int, dict[str, float]]:
    """Compute metrics broken down by difficulty tier.

    VALID entries (difficulty_tier=None) are included in ALL tiers' FPR
    denominators, not assigned to tier 1. This ensures each tier reports a
    meaningful FPR rather than tier 1 absorbing all false positives and tiers
    2-3 showing 0.0 FPR.

    For each tier the confusion matrix is built from:
    - hallucinated entries of that tier (for detection rate / recall)
    - ALL valid entries across all tiers (for FPR)
    """
    # Separate valid from hallucinated entries
    valid_entries = [e for e in entries if e.label == "VALID"]
    hall_by_tier: dict[int, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        if entry.label == "HALLUCINATED":
            tier = entry.difficulty_tier or 1
            hall_by_tier[tier].append(entry)

    result = {}
    for tier, hall_e in sorted(hall_by_tier.items()):
        # Each tier uses its own hallucinated entries + all valid entries
        tier_e = hall_e + valid_entries
        cm = build_confusion_matrix(tier_e, predictions)
        result[tier] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "precision": cm.precision,
            "count": len(hall_e),  # report hallucinated-only count per tier
        }
    return result


def per_type_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
    compute_ci: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by hallucination type.

    Args:
        entries: Benchmark entries.
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).
        compute_ci: If True, add Wilson score 95% CI keys ``dr_ci_lower`` and
            ``dr_ci_upper`` for detection rate per type. CI is only meaningful
            for hallucinated types (n > 0); valid entries get 0.0/0.0.

    Returns:
        Dict mapping hallucination type to metrics dict. When ``compute_ci``
        is True, each inner dict also contains ``dr_ci_lower`` and
        ``dr_ci_upper``.
    """
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}
    type_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        type_entries[h_type].append(entry)

    result = {}
    for h_type, type_e in sorted(type_entries.items()):
        cm = build_confusion_matrix(type_e, predictions)
        metrics: dict[str, float] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "count": len(type_e),
        }
        if compute_ci:
            z = 1.96  # 95% CI
            n = cm.tp + cm.fn  # hallucinated entries for this type
            p = cm.detection_rate
            if n > 0:
                denom = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denom
                half = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
                ci_lower = max(0.0, centre - half)
                ci_upper = min(1.0, centre + half)
            else:
                ci_lower = 0.0
                ci_upper = 0.0
            metrics["dr_ci_lower"] = ci_lower
            metrics["dr_ci_upper"] = ci_upper
        result[h_type] = metrics
    return result


def per_generation_method_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> dict[str, dict[str, float | int]]:
    """Compute metrics broken down by generation method.

    Groups entries by their ``generation_method`` field (e.g. ``perturbation``,
    ``llm_generated``, ``adversarial``, ``scraped``, ``real_world``) and
    computes per-group detection rate, FPR, and F1.

    VALID entries (``scraped`` / ``real_world``) have no hallucinations, so
    their ``detection_rate`` and ``f1`` are 0.0 by definition; FPR reflects
    how often the tool incorrectly flags them.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).

    Returns:
        Dict mapping generation method string to a metrics dict with keys:
        ``n`` (int), ``detection_rate``, ``false_positive_rate``, ``f1``.
    """
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}

    method_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        method = entry.generation_method or "unknown"
        method_entries[method].append(entry)

    result: dict[str, dict[str, float | int]] = {}
    for method, method_e in sorted(method_entries.items()):
        cm = build_confusion_matrix(method_e, predictions)
        result[method] = {
            "n": len(method_e),
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
        }
    return result


def cost_efficiency(predictions: list[Prediction]) -> dict[str, float | None]:
    """Compute cost efficiency metrics."""
    if not predictions:
        return {"entries_per_second": 0.0, "mean_api_calls": 0.0, "total_wall_clock_seconds": 0.0}

    total_time = sum(p.wall_clock_seconds for p in predictions)
    total_api_calls = sum(p.api_calls for p in predictions)
    n = len(predictions)

    return {
        "entries_per_second": n / total_time if total_time > 0 else None,
        "mean_api_calls": total_api_calls / n,
        "total_wall_clock_seconds": total_time,
    }


def expected_calibration_error(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
    n_bins: int = 10,
    adaptive: bool = True,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - mean_confidence| per bin,
    returns weighted average by bin size.

    Confidence represents the tool's belief in its own prediction: a tool
    predicting HALLUCINATED with confidence 0.9 claims 90% certainty.

    UNCERTAIN predictions are excluded from ECE entirely (no reliable confidence signal).

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).
        n_bins: Number of bins for calibration estimation.
        adaptive: If True (default), use equal-frequency (quantile) binning — sort
            predictions by confidence and split into n_bins equal-sized groups. This
            avoids pathological empty-bin behaviour for deterministic tools that
            concentrate all predictions in 2 bins (e.g., confidence in {0.0, 1.0}).
            If False, use fixed equal-width bins [0, 1/n_bins), [1/n_bins, 2/n_bins), …
    """
    # F-15: accept list as well as dict
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}

    if not predictions:
        return 0.0

    # Collect (confidence, correctness) pairs — skip UNCERTAIN (no reliable confidence signal)
    pairs: list[tuple[float, bool]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        if pred is None:
            continue

        # UNCERTAIN excluded from ECE (consistent with AUROC/AUPRC treatment)
        if pred.label == "UNCERTAIN":
            continue

        is_correct = pred.label == entry.label
        pairs.append((pred.confidence, is_correct))

    if not pairs:
        return 0.0

    # F-14: warn when the tool uses only 1-2 distinct confidence values — ECE is unreliable
    # because calibration curves require meaningful variation in confidence scores.
    if len(set(conf for conf, _ in pairs)) <= 2:
        logger.warning("ECE is unreliable for tools with <= 2 distinct confidence values.")

    ece = 0.0
    total = len(pairs)

    if adaptive:
        # Equal-frequency (quantile) binning: sort by confidence and split evenly
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        bin_size_base = total // n_bins
        remainder = total % n_bins
        bins_adaptive: list[list[tuple[float, bool]]] = []
        start = 0
        for i in range(n_bins):
            # Distribute remainder among the first `remainder` bins
            end = start + bin_size_base + (1 if i < remainder else 0)
            if start < total:
                bins_adaptive.append(sorted_pairs[start:end])
            start = end

        for bin_data in bins_adaptive:
            if not bin_data:
                continue
            bin_size = len(bin_data)
            avg_conf = sum(conf for conf, _ in bin_data) / bin_size
            accuracy = sum(1 for _, correct in bin_data if correct) / bin_size
            ece += (bin_size / total) * abs(accuracy - avg_conf)
    else:
        # Fixed equal-width bins
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
        for conf, correct in pairs:
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, correct))

        for bin_data in bins:
            if not bin_data:
                continue
            bin_size = len(bin_data)
            avg_conf = sum(conf for conf, _ in bin_data) / bin_size
            accuracy = sum(1 for _, correct in bin_data if correct) / bin_size
            ece += (bin_size / total) * abs(accuracy - avg_conf)

    return ece


def auroc(
    entries: list[BenchmarkEntry], predictions: dict[str, Prediction] | list[Prediction]
) -> float | None:
    """Compute Area Under ROC Curve for hallucination detection.

    Positive class: HALLUCINATED. Score = confidence for HALLUCINATED predictions,
    (1 - confidence) for VALID predictions.

    Missing predictions and UNCERTAIN predictions are excluded from the AUROC
    computation entirely (not assigned a neutral score of 0.5). This avoids
    artificially inflating or deflating AUROC by treating absence of a prediction
    as weak evidence. Only entries with a definite HALLUCINATED or VALID prediction
    are included.

    Returns None if fewer than 2 classes present among the included entries.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).

    Returns:
        AUROC score in [0, 1], or None if insufficient class diversity.
    """
    # F-15: accept list as well as dict
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}

    # Build (score, true_label) pairs where true_label is 1 for HALLUCINATED, 0 for VALID
    # Skip missing and UNCERTAIN predictions entirely.
    pairs: list[tuple[float, int]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)

        # Skip missing predictions and UNCERTAIN (no reliable score available)
        if pred is None or pred.label == "UNCERTAIN":
            continue

        score = pred.confidence if pred.label == "HALLUCINATED" else 1.0 - pred.confidence

        true_label = 1 if entry.label == "HALLUCINATED" else 0
        pairs.append((score, true_label))

    if not pairs:
        return None

    # Check if we have both classes
    unique_labels = set(label for _, label in pairs)
    if len(unique_labels) < 2:
        return None

    # Sort by score descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Compute ROC curve points (TPR, FPR) using trapezoidal integration
    num_positive = sum(label for _, label in pairs)
    num_negative = len(pairs) - num_positive

    if num_positive == 0 or num_negative == 0:
        return None

    # Walk through sorted list computing TPR/FPR at each threshold
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    area = 0.0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1

        # At each threshold change, compute area increment
        # Check if this is the last point or score changes
        is_last = i == len(pairs) - 1
        score_changes = is_last or pairs[i + 1][0] != score

        if score_changes:
            tpr = tp / num_positive
            fpr = fp / num_negative

            # Trapezoidal rule: area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0

            prev_tpr = tpr
            prev_fpr = fpr

    return area


def auprc(
    entries: list[BenchmarkEntry], predictions: dict[str, Prediction] | list[Prediction]
) -> float | None:
    """Compute Area Under Precision-Recall Curve for hallucination detection.

    Same scoring convention as auroc(). Missing predictions and UNCERTAIN predictions
    are excluded from the computation entirely (not assigned a neutral score of 0.5).
    Returns None if no positive examples among the included entries.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).

    Returns:
        AUPRC score in [0, 1], or None if no positive examples.
    """
    # F-15: accept list as well as dict
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}

    # Build (score, true_label) pairs where true_label is 1 for HALLUCINATED, 0 for VALID
    # Skip missing and UNCERTAIN predictions entirely.
    pairs: list[tuple[float, int]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)

        # Skip missing predictions and UNCERTAIN (no reliable score available)
        if pred is None or pred.label == "UNCERTAIN":
            continue

        score = pred.confidence if pred.label == "HALLUCINATED" else 1.0 - pred.confidence

        true_label = 1 if entry.label == "HALLUCINATED" else 0
        pairs.append((score, true_label))

    if not pairs:
        return None

    # Check if we have positive examples
    num_positive = sum(label for _, label in pairs)
    if num_positive == 0:
        return None

    # Sort by score descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Walk through sorted list computing precision/recall at each threshold
    tp = 0
    fp = 0
    prev_recall = 0.0

    # Collect (recall, precision) points at each threshold change
    pr_points: list[tuple[float, float]] = []

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1

        is_last = i == len(pairs) - 1
        score_changes = is_last or pairs[i + 1][0] != score

        if score_changes:
            recall = tp / num_positive
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            pr_points.append((recall, precision))

    if not pr_points:
        return None

    # Interpolated average precision: monotone precision envelope then integrate.
    # Walk right-to-left to enforce precision_interp[i] = max(precision[i:])
    interp_precisions = [p for _, p in pr_points]
    for i in range(len(interp_precisions) - 2, -1, -1):
        interp_precisions[i] = max(interp_precisions[i], interp_precisions[i + 1])

    # Integrate using interpolated precision at each recall change
    area = 0.0
    prev_recall = 0.0
    for idx, (recall, _) in enumerate(pr_points):
        area += interp_precisions[idx] * (recall - prev_recall)
        prev_recall = recall

    return area


def source_stratified_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> dict[str, dict[str, float]]:
    """Compute detection metrics stratified by API sources queried.

    Groups predictions by their api_sources_queried field and computes
    per-source detection rate and false positive rate.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).

    Returns dict mapping source combination string to metrics dict.
    """
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}
    # Group predictions by source combination
    source_groups: dict[str, list[str]] = defaultdict(list)
    for key, pred in predictions.items():
        source_key = (
            ",".join(sorted(pred.api_sources_queried)) if pred.api_sources_queried else "none"
        )
        source_groups[source_key].append(key)

    result = {}
    for source_key, keys in sorted(source_groups.items()):
        # Use set for O(1) membership lookup
        keys_set = set(keys)
        # Filter entries that have predictions in this group
        group_entries = [e for e in entries if e.bibtex_key in keys_set]
        if not group_entries:
            continue

        group_preds = {k: predictions[k] for k in keys}
        cm = build_confusion_matrix(group_entries, group_preds)
        result[source_key] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "count": len(group_entries),
        }

    return result


def generation_method_stratified_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> dict[str, dict[str, float | int]]:
    """Deprecated alias for ``per_generation_method_metrics``.

    Use ``per_generation_method_metrics`` directly for new code.
    """
    return per_generation_method_metrics(entries, predictions)


def subtest_accuracy_table(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> dict[str, dict[str, float]]:
    """Compute per-subtest accuracy across all entries.

    For each subtest name, computes accuracy (fraction of entries where
    the prediction's subtest result matches the ground truth subtest).

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions — either a dict keyed by bibtex_key
            or a list of Prediction objects (converted internally).

    Returns:
        Dict mapping subtest name to {accuracy, count, true_positives, false_positives}.
    """
    if isinstance(predictions, list):
        predictions = {p.bibtex_key: p for p in predictions}
    from hallmark.dataset.schema import SUBTEST_NAMES

    result = {}
    for subtest_name in SUBTEST_NAMES:
        matches = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for entry in entries:
            if subtest_name not in entry.subtests:
                continue
            pred = predictions.get(entry.bibtex_key)
            if pred is None or subtest_name not in pred.subtest_results:
                continue

            gt_value = entry.subtests[subtest_name]
            pred_value = pred.subtest_results[subtest_name]

            if pred_value is None:
                continue

            total += 1
            if pred_value == gt_value:
                matches += 1

            if gt_value and pred_value:
                tp += 1
            elif not gt_value and pred_value:
                fp += 1
            elif not gt_value and not pred_value:
                tn += 1
            elif gt_value and not pred_value:
                fn += 1

        if total > 0:
            result[subtest_name] = {
                "accuracy": matches / total,
                "count": total,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }

    return result


def stratified_bootstrap_ci(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    metric_fn: Callable[[list[BenchmarkEntry], list[Prediction]], float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute stratified bootstrap CI preserving hallucination type distribution.

    Stratification ensures each bootstrap resample maintains the original
    proportion of each hallucination type, preventing bias from underrepresented types.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.
        metric_fn: Function that takes (entries, predictions) and returns a scalar metric.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (lower, upper) percentile-based confidence interval.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for bootstrap CIs: pip install numpy") from e

    rng = np.random.default_rng(seed)

    # Group entries and predictions by hallucination type
    type_groups: dict[str, tuple[list[BenchmarkEntry], list[Prediction]]] = defaultdict(
        lambda: ([], [])
    )
    pred_map = {p.bibtex_key: p for p in predictions}

    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        # Include all entries (not just those with predictions) so CIs reflect
        # uncertainty from missing predictions, consistent with build_confusion_matrix.
        pred = pred_map.get(entry.bibtex_key)
        type_groups[h_type][0].append(entry)
        if pred is not None:
            type_groups[h_type][1].append(pred)
        else:
            # Missing prediction → treated as VALID with default confidence
            type_groups[h_type][1].append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                )
            )

    # Bootstrap resampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        resampled_entries = []
        resampled_predictions = []

        # Resample within each type group
        for entries_in_type, preds_in_type in type_groups.values():
            n = len(entries_in_type)
            if n == 0:
                continue
            indices = rng.choice(n, size=n, replace=True)
            resampled_entries.extend([entries_in_type[i] for i in indices])
            resampled_predictions.extend([preds_in_type[i] for i in indices])

        # Compute metric on resampled data
        if resampled_entries:
            metric_value = metric_fn(resampled_entries, resampled_predictions)
            bootstrap_metrics.append(metric_value)

    if not bootstrap_metrics:
        return (0.0, 0.0)

    # Compute percentile CI
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    lower = float(np.percentile(bootstrap_metrics, lower_percentile))
    upper = float(np.percentile(bootstrap_metrics, upper_percentile))

    return (lower, upper)


def _bootstrap_all_cis(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    n_bootstrap: int,
    seed: int,
    alpha: float,
    tier_weights: dict[int, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap CIs for all 6 primary metrics in a single resampling loop.

    This is ~6x faster than calling ``stratified_bootstrap_ci`` 6 times separately
    because it reuses each bootstrap resample for all metrics.

    Stratification is by hallucination type, matching ``stratified_bootstrap_ci``.
    Missing predictions are filled with VALID/0.5 placeholders, consistent with
    ``build_confusion_matrix``.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        alpha: Significance level (e.g., 0.05 for 95% CIs).
        tier_weights: Optional tier weights for tier-weighted F1.

    Returns:
        Dict mapping metric name to (lower, upper) CI tuple. Keys:
        ``detection_rate``, ``f1_hallucination``, ``tier_weighted_f1``,
        ``false_positive_rate``, ``ece``, ``mcc``.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for bootstrap CIs: pip install numpy") from e

    rng = np.random.default_rng(seed)

    # Group entries and predictions by hallucination type for stratified resampling
    type_groups: dict[str, tuple[list[BenchmarkEntry], list[Prediction]]] = defaultdict(
        lambda: ([], [])
    )
    pred_map = {p.bibtex_key: p for p in predictions}

    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        pred = pred_map.get(entry.bibtex_key)
        type_groups[h_type][0].append(entry)
        if pred is not None:
            type_groups[h_type][1].append(pred)
        else:
            type_groups[h_type][1].append(
                Prediction(bibtex_key=entry.bibtex_key, label="VALID", confidence=0.5)
            )

    # Per-metric accumulators
    dr_samples: list[float] = []
    f1_samples: list[float] = []
    twf1_samples: list[float] = []
    fpr_samples: list[float] = []
    ece_samples: list[float] = []
    mcc_samples: list[float] = []

    for _ in range(n_bootstrap):
        resampled_entries: list[BenchmarkEntry] = []
        resampled_preds: list[Prediction] = []

        for entries_in_type, preds_in_type in type_groups.values():
            n = len(entries_in_type)
            if n == 0:
                continue
            indices = rng.choice(n, size=n, replace=True)
            resampled_entries.extend([entries_in_type[i] for i in indices])
            resampled_preds.extend([preds_in_type[i] for i in indices])

        if not resampled_entries:
            continue

        # F-8: Use resampled_preds as a list (not dict) for tier_weighted_f1 and ECE so that
        # duplicate entries from bootstrap resampling-with-replacement all contribute
        # independently. Building a dict would deduplicate by bibtex_key and narrow CIs.
        # build_confusion_matrix iterates entries in lockstep with predictions via the list
        # path, so duplicates are counted correctly there too.
        cm_boot = build_confusion_matrix(resampled_entries, resampled_preds)
        dr_samples.append(cm_boot.detection_rate)
        f1_samples.append(cm_boot.f1)
        # F-9: Skip FPR when no valid entries in this resample to avoid mixing 0.0 (undefined)
        # with genuine 0.0 (no false positives), which would bias the CI downward.
        if cm_boot.tn + cm_boot.fp > 0:
            fpr_samples.append(cm_boot.false_positive_rate)
        mcc_samples.append(cm_boot.mcc)
        twf1_samples.append(tier_weighted_f1(resampled_entries, resampled_preds, tier_weights))
        ece_samples.append(
            expected_calibration_error(resampled_entries, resampled_preds, adaptive=True)
        )

    def _ci(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return (0.0, 0.0)
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return (lo, hi)

    return {
        "detection_rate": _ci(dr_samples),
        "f1_hallucination": _ci(f1_samples),
        "tier_weighted_f1": _ci(twf1_samples),
        "false_positive_rate": _ci(fpr_samples),
        "ece": _ci(ece_samples),
        "mcc": _ci(mcc_samples),
    }


def paired_bootstrap_test(
    entries: list[BenchmarkEntry],
    predictions_a: list[Prediction],
    predictions_b: list[Prediction],
    metric_fn: Callable[[list[BenchmarkEntry], list[Prediction]], float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
    two_sided: bool = True,
) -> tuple[float, float, float]:
    """Paired bootstrap significance test with stratified resampling by hallucination type.

    Returns (observed_diff, p_value, effect_size_cohens_h).
    H0: metric_A <= metric_B.

    When ``two_sided=True`` (default), the returned p-value is two-sided:
    p = 2 * fraction of bootstrap resamples where delta (metric_A - metric_B) <= 0,
    capped at 1.0. This tests whether the two tools differ in either direction.

    When ``two_sided=False``, the p-value is one-sided: p = fraction of bootstrap
    resamples where delta <= 0. This tests the directional hypothesis that tool A
    is better than tool B.

    Resampling is stratified by hallucination type, matching the approach in
    ``stratified_bootstrap_ci``. Each bootstrap iteration resamples within each
    type group and concatenates, preserving the original type distribution.

    Args:
        entries: Benchmark entries (ground truth).
        predictions_a: Predictions from tool A.
        predictions_b: Predictions from tool B.
        metric_fn: Function that takes (entries, predictions) and returns a scalar metric.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        two_sided: If True (default), return a two-sided p-value (one-sided * 2, capped at 1.0).
            If False, return the one-sided p-value directly.

    Returns:
        Tuple (observed_diff, p_value, effect_size_cohens_h).
        - observed_diff: metric_A - metric_B on original data
        - p_value: two-sided p-value if ``two_sided=True``, one-sided otherwise
        - effect_size_cohens_h: Cohen's h effect size (2 * (arcsin(sqrt(p_a)) - arcsin(sqrt(p_b))))
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for bootstrap tests: pip install numpy") from e

    rng = np.random.default_rng(seed)

    # Compute observed difference
    metric_a = metric_fn(entries, predictions_a)
    metric_b = metric_fn(entries, predictions_b)
    observed_diff = metric_a - metric_b

    # Build prediction maps
    pred_map_a = {p.bibtex_key: p for p in predictions_a}
    pred_map_b = {p.bibtex_key: p for p in predictions_b}

    # Group entries (and corresponding paired predictions) by hallucination type for
    # stratified resampling — same grouping logic as stratified_bootstrap_ci().
    type_groups: dict[str, list[tuple[BenchmarkEntry, Prediction, Prediction]]] = defaultdict(list)
    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        pa = pred_map_a.get(entry.bibtex_key) or Prediction(
            bibtex_key=entry.bibtex_key, label="VALID", confidence=0.5
        )
        pb = pred_map_b.get(entry.bibtex_key) or Prediction(
            bibtex_key=entry.bibtex_key, label="VALID", confidence=0.5
        )
        type_groups[h_type].append((entry, pa, pb))

    # Bootstrap resampling: resample within each type group, then concatenate (paired)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        resampled_entries = []
        resampled_preds_a: list[Prediction] = []
        resampled_preds_b: list[Prediction] = []

        for group in type_groups.values():
            n = len(group)
            if n == 0:
                continue
            indices = rng.choice(n, size=n, replace=True)
            for i in indices:
                entry, pa, pb = group[i]
                resampled_entries.append(entry)
                resampled_preds_a.append(pa)
                resampled_preds_b.append(pb)

        if resampled_entries:
            metric_a_boot = metric_fn(resampled_entries, resampled_preds_a)
            metric_b_boot = metric_fn(resampled_entries, resampled_preds_b)
            bootstrap_diffs.append(metric_a_boot - metric_b_boot)

    # One-sided p-value: fraction of bootstrap samples where delta <= 0.
    # Note: this implementation is conservative — it counts bootstrap diffs
    # against the raw null (delta <= 0) rather than the null-centered approach
    # (d - observed_diff <= 0). The null-centered method would be unbiased
    # (matching the permutation-test philosophy), but the conservative approach
    # is acceptable for a benchmark leaderboard where we prefer to under-report
    # significance rather than over-report it. This is a deliberate design choice.
    p_value_one_sided = (
        float(np.mean([d <= 0 for d in bootstrap_diffs])) if bootstrap_diffs else 1.0
    )
    p_value = min(1.0, 2.0 * p_value_one_sided) if two_sided else p_value_one_sided

    # Cohen's h effect size
    if 0.0 <= metric_a <= 1.0 and 0.0 <= metric_b <= 1.0:
        cohens_h = float(2 * (np.arcsin(np.sqrt(metric_a)) - np.arcsin(np.sqrt(metric_b))))
    else:
        cohens_h = metric_a - metric_b  # fallback for non-proportion metrics

    return (observed_diff, p_value, float(cohens_h))


def tier_weight_sensitivity(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    weight_schemes: dict[str, dict[int, float]] | None = None,
) -> dict[str, float]:
    """Compute tier-weighted F1 under different weighting schemes.

    Default schemes: uniform {1,1,1}, linear {1,2,3}, quadratic {1,4,9},
    log {1,1.6,2.1}, inverse_difficulty (based on mean DR across entries).
    Returns dict mapping scheme name to TW-F1 value.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.
        weight_schemes: Optional custom weight schemes. If None, uses default schemes.

    Returns:
        Dict mapping scheme name to tier-weighted F1 value.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for tier weight sensitivity: pip install numpy") from e

    pred_map = {p.bibtex_key: p for p in predictions}

    if weight_schemes is None:
        weight_schemes = {
            "uniform": {1: 1.0, 2: 1.0, 3: 1.0},
            "linear": {1: 1.0, 2: 2.0, 3: 3.0},
            "quadratic": {1: 1.0, 2: 4.0, 3: 9.0},
            "log": {1: 1.0, 2: 1.585, 3: 2.0},  # log2(2), log2(3), log2(4)
        }

        # Compute inverse difficulty weights based on mean DR per tier
        tier_dr: dict[int, list[float]] = defaultdict(list)
        for entry in entries:
            if entry.label != "HALLUCINATED":
                continue
            tier = entry.difficulty_tier or 1
            pred = pred_map.get(entry.bibtex_key)
            if pred is not None:
                is_detected = pred.label == "HALLUCINATED"
                tier_dr[tier].append(float(is_detected))

        inverse_difficulty = {}
        for tier in [1, 2, 3]:
            tier_data = tier_dr.get(tier, [])
            if tier_data:
                mean_dr = np.mean(tier_data)
                # Weight inversely proportional to detection rate (harder = higher weight)
                inverse_difficulty[tier] = 1.0 / (mean_dr + 0.01)  # +0.01 to avoid div by zero
            else:
                inverse_difficulty[tier] = 1.0

        # Normalize so Tier 1 = 1.0
        base_weight = inverse_difficulty.get(1, 1.0)
        inverse_difficulty = {k: v / base_weight for k, v in inverse_difficulty.items()}
        weight_schemes["inverse_difficulty"] = inverse_difficulty

    results = {}
    for scheme_name, weights in weight_schemes.items():
        tw_f1 = tier_weighted_f1(entries, pred_map, tier_weights=weights)
        results[scheme_name] = tw_f1

    return results


def equivalence_test(
    entries_a: list[BenchmarkEntry],
    entries_b: list[BenchmarkEntry],
    predictions: list[Prediction],
    epsilon: float = 0.02,
    n_permutations: int = 10_000,
    seed: int = 42,
    metric_fn: Callable[[list[BenchmarkEntry], dict[str, Prediction]], float] | None = None,
) -> tuple[bool, float, float]:
    """TOST-based equivalence test for dataset scaling validation.

    Uses Two One-Sided Tests (TOST) with permutation distributions to test:
    - H0: |metric(A) - metric(B)| >= epsilon (groups are NOT equivalent)
    - H1: |metric(A) - metric(B)| < epsilon (groups ARE equivalent)

    The TOST procedure performs two one-sided tests:
    1. Upper bound test: H0_upper: diff >= epsilon
    2. Lower bound test: H0_lower: diff <= -epsilon

    Equivalence is concluded if BOTH one-sided tests reject (p_tost < alpha).

    Args:
        entries_a: First set of benchmark entries.
        entries_b: Second set of benchmark entries.
        predictions: Predictions covering both entry sets.
        epsilon: Equivalence margin (default 0.02 = 2 percentage points).
        n_permutations: Number of permutation resamples.
        seed: Random seed for reproducibility.
        metric_fn: Function (entries, pred_map) -> float. Defaults to F1
            (consistent with ``paired_bootstrap_test``). The pred_map argument
            is a dict keyed by bibtex_key, matching ``build_confusion_matrix``.

    Returns:
        Tuple (is_equivalent, observed_diff, p_tost).
        - is_equivalent: True if p_tost < 0.05 (reject non-equivalence)
        - observed_diff: metric_A - metric_B on original split
        - p_tost: max(p_upper, p_lower) from TOST procedure
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for equivalence tests: pip install numpy") from e

    rng = np.random.default_rng(seed)

    # Build prediction map
    pred_map = {p.bibtex_key: p for p in predictions}

    # Default metric: F1 on the HALLUCINATED class
    if metric_fn is None:

        def metric_fn(ents: list[BenchmarkEntry], pmap: dict[str, Prediction]) -> float:
            return build_confusion_matrix(ents, pmap).f1

    # Compute observed difference
    observed_diff = metric_fn(entries_a, pred_map) - metric_fn(entries_b, pred_map)

    # Permutation test: shuffle entry assignment to groups, predictions stay
    # mapped by bibtex_key. The null hypothesis is that group assignment
    # (A vs B) doesn't affect the metric.
    combined_entries = entries_a + entries_b
    n_a = len(entries_a)

    permuted_diffs = []
    for _ in range(n_permutations):
        perm_indices = rng.permutation(len(combined_entries))
        perm_entries = [combined_entries[i] for i in perm_indices]

        perm_entries_a = perm_entries[:n_a]
        perm_entries_b = perm_entries[n_a:]

        # Use the shared pred_map — predictions are looked up by bibtex_key
        if perm_entries_a and perm_entries_b:
            permuted_diffs.append(
                metric_fn(perm_entries_a, pred_map) - metric_fn(perm_entries_b, pred_map)
            )

    # TOST: Two One-Sided Tests
    # The permutation distribution is centered at 0 (null of no difference).
    # We shift it to test against the equivalence bounds ±epsilon.
    if permuted_diffs:
        # Upper bound test: H0_upper: true_diff >= epsilon
        # p_upper = P(perm_diff <= observed_diff - epsilon)
        p_upper = float(np.mean([d <= observed_diff - epsilon for d in permuted_diffs]))

        # Lower bound test: H0_lower: true_diff <= -epsilon
        # p_lower = P(perm_diff >= observed_diff + epsilon)
        p_lower = float(np.mean([d >= observed_diff + epsilon for d in permuted_diffs]))

        # TOST p-value: reject non-equivalence only if BOTH tests reject
        p_tost = max(p_upper, p_lower)
    else:
        p_tost = 1.0

    # Equivalence: reject H0 of non-equivalence if p_tost < alpha
    is_equivalent = p_tost < 0.05

    return (is_equivalent, observed_diff, p_tost)


def apply_multiple_comparison_correction(
    p_values: dict[str, float],
    method: str = "holm",
) -> dict[str, float]:
    """Apply multiple comparison correction to a family of p-values.

    When comparing tools across multiple hallucination types or tiers,
    the family-wise error rate (FWER) inflates beyond the nominal alpha. This
    function adjusts p-values to control either the FWER or the false discovery
    rate (FDR), depending on the chosen method.

    Args:
        p_values: Dict mapping comparison label to raw p-value.
        method: Correction method.
            - "holm" (Holm-Bonferroni, recommended): step-down procedure that
              controls the FWER while being less conservative than Bonferroni.
            - "bonferroni": single-step FWER control, most conservative.
            - "bh" (Benjamini-Hochberg): step-up procedure that controls the
              FDR (expected fraction of false discoveries among rejections).
              Less conservative than FWER methods; preferred when many
              simultaneous tests are performed.

    Returns:
        Dict mapping comparison label to adjusted p-value.

    Example:
        >>> raw = {"type_a": 0.03, "type_b": 0.01, "type_c": 0.08}
        >>> apply_multiple_comparison_correction(raw)
        {'type_b': 0.03, 'type_a': 0.06, 'type_c': 0.08}
    """
    if not p_values:
        return {}

    items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(items)
    adjusted: dict[str, float] = {}

    if method == "bonferroni":
        for label, p in items:
            adjusted[label] = min(1.0, p * n)
    elif method == "holm":
        max_so_far = 0.0
        for rank, (label, p) in enumerate(items):
            adj = min(1.0, p * (n - rank))
            max_so_far = max(max_so_far, adj)
            adjusted[label] = max_so_far
    elif method == "bh":
        # Benjamini-Hochberg FDR control
        sorted_items = items  # already sorted by p-value ascending
        for rank, (label, p) in enumerate(sorted_items, 1):
            adjusted[label] = min(1.0, p * n / rank)
        # Enforce monotonicity (step-up): traverse from largest to smallest
        prev = 1.0
        for label, _ in reversed(sorted_items):
            adjusted[label] = min(adjusted[label], prev)
            prev = adjusted[label]
    else:
        raise ValueError(
            f"Unknown correction method: {method!r}. Use 'holm', 'bonferroni', or 'bh'."
        )

    return adjusted


def compare_tools(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    metric_fn: Callable[[list[BenchmarkEntry], list[Prediction]], float] | None = None,
    correction_method: str = "holm",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Run pairwise paired bootstrap tests across all tool pairs with multiple comparison correction.

    For C(n,2) tool pairs, runs ``paired_bootstrap_test`` and then applies
    ``apply_multiple_comparison_correction`` to the collected p-values.

    Note: ``paired_bootstrap_test`` is called with ``two_sided=True`` (the default).
    The ``p_value_raw`` and ``p_value_adjusted`` keys in the output are two-sided p-values.

    Args:
        entries: Benchmark entries (ground truth).
        tool_predictions: Dict mapping tool_name -> list of Predictions.
        metric_fn: Metric function for comparison. Defaults to F1.
        correction_method: Correction method for ``apply_multiple_comparison_correction``.
        n_bootstrap: Number of bootstrap resamples per pair.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: tool_a, tool_b, observed_diff, p_value_raw,
        p_value_adjusted, effect_size, significant.
        p_value_raw and p_value_adjusted are two-sided p-values.
    """
    if metric_fn is None:

        def metric_fn(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            cm_inner = build_confusion_matrix(ents, {p.bibtex_key: p for p in preds})
            return cm_inner.f1

    tool_names = sorted(tool_predictions.keys())
    if len(tool_names) < 2:
        return []

    # Run pairwise tests
    raw_results: list[dict[str, object]] = []
    raw_p_values: dict[str, float] = {}

    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            name_a, name_b = tool_names[i], tool_names[j]
            preds_a = tool_predictions[name_a]
            preds_b = tool_predictions[name_b]

            # Derive a unique seed per pair so each pair uses a distinct
            # resampling pattern. Without this, every pair uses identical
            # bootstrap samples (same seed), which, while not technically wrong
            # (pairs are independent), is statistically undesirable.
            pair_seed = (seed + hash((name_a, name_b))) % (2**31)

            obs_diff, p_value_two_sided, effect_size = paired_bootstrap_test(
                entries,
                preds_a,
                preds_b,
                metric_fn,
                n_bootstrap=n_bootstrap,
                seed=pair_seed,
                two_sided=True,
            )

            pair_key = f"{name_a}_vs_{name_b}"
            raw_p_values[pair_key] = p_value_two_sided
            raw_results.append(
                {
                    "tool_a": name_a,
                    "tool_b": name_b,
                    "observed_diff": obs_diff,
                    "p_value_raw": p_value_two_sided,
                    "effect_size": effect_size,
                    "_pair_key": pair_key,
                }
            )

    # Apply multiple comparison correction
    adjusted = apply_multiple_comparison_correction(raw_p_values, method=correction_method)

    for result in raw_results:
        pair_key = str(result["_pair_key"])
        result["p_value_adjusted"] = adjusted[pair_key]
        result["significant"] = adjusted[pair_key] < 0.05
        del result["_pair_key"]

    return raw_results


def evaluate(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    tool_name: str = "unknown",
    split_name: str = "unknown",
    predictions_per_strategy: list[dict[str, Prediction]] | None = None,
    compute_ci: bool = False,
    n_bootstrap: int = 10_000,
    ci_seed: int = 42,
) -> EvaluationResult:
    """Run full evaluation and return aggregated results.

    Evaluation Protocol:
    - UNCERTAIN predictions are excluded from classification metrics (DR, FPR, F1, TW-F1).
      They count toward coverage but not toward the confusion matrix.
    - Missing predictions are treated as VALID (conservative default)
    - Pre-screening results (if any) are included in the tool's predictions
    - Incomplete evaluations (partial coverage) can be aggregated using Plackett-Luce
      ranking (see hallmark.evaluation.ranking module)

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.
        tool_name: Name of the tool being evaluated.
        split_name: Name of the benchmark split.
        predictions_per_strategy: Optional list of prediction dicts for detect@k.
        compute_ci: If True, compute bootstrap confidence intervals (requires numpy).
        n_bootstrap: Number of bootstrap resamples for CIs (default 10_000).
        ci_seed: Random seed for bootstrap CI computation (default 42).

    Returns:
        EvaluationResult with primary metrics (DR, FPR, F1, TW-F1, ECE),
        per-tier and per-type breakdowns, count of UNCERTAIN predictions,
        and optionally bootstrap CIs for primary metrics.
    """
    # Defense-in-depth: filter canaries even if load_entries() already did.
    entries = [e for e in entries if not is_canary_entry(e)]

    # Warn on duplicate bibtex_keys before dict conversion (last prediction wins)
    seen_keys: set[str] = set()
    duplicate_keys: list[str] = []
    for p in predictions:
        if p.bibtex_key in seen_keys:
            duplicate_keys.append(p.bibtex_key)
        seen_keys.add(p.bibtex_key)
    if duplicate_keys:
        logger.warning(
            "Duplicate bibtex_key(s) found in predictions: %s. Last prediction wins.",
            duplicate_keys,
        )

    pred_map = {p.bibtex_key: p for p in predictions}

    # Warn on degenerate prediction distributions
    if pred_map:
        label_counts: dict[str, int] = defaultdict(int)
        for p in pred_map.values():
            label_counts[p.label] += 1
        total_preds = len(pred_map)
        for label, count in label_counts.items():
            pct = 100.0 * count / total_preds
            if pct > 95.0:
                logger.warning(
                    "Degenerate predictions: %.1f%% are %s. Results may not be meaningful.",
                    pct,
                    label,
                )

    entry_keys = {e.bibtex_key for e in entries}
    pred_keys = set(pred_map.keys())
    overlap = entry_keys & pred_keys
    if len(overlap) == 0:
        logger.warning(
            "Zero key overlap between entries (%d) and predictions (%d). "
            "Check that predictions match the correct split.",
            len(entry_keys),
            len(pred_keys),
        )
    elif len(overlap) < len(entry_keys) // 2:
        logger.warning(
            "Low key overlap: %d/%d entries have predictions (%.0f%%). "
            "Missing entries are treated as VALID.",
            len(overlap),
            len(entry_keys),
            100 * len(overlap) / len(entry_keys),
        )

    # F-7: Use intersection of entry_keys and pred_keys so that extra predictions
    # (keys not in entries) do not push coverage above 1.0.
    coverage = len(entry_keys & pred_keys) / len(entries) if entries else 1.0

    cm = build_confusion_matrix(entries, pred_map)
    tw_f1 = tier_weighted_f1(entries, pred_map)
    tier_metrics = per_tier_metrics(entries, pred_map)
    type_metrics = per_type_metrics(entries, pred_map)
    cost = cost_efficiency(predictions)
    ece_score = expected_calibration_error(entries, pred_map, adaptive=True)
    auroc_score = auroc(entries, pred_map)
    auprc_score = auprc(entries, pred_map)

    dat_k: dict[int, float] = {}
    if predictions_per_strategy:
        dat_k = union_recall_at_k(entries, predictions_per_strategy)

    num_hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")
    num_valid = sum(1 for e in entries if e.label == "VALID")
    num_uncertain = sum(1 for p in predictions if p.label == "UNCERTAIN")

    fpr: float | None = cm.false_positive_rate
    if num_valid == 0:
        logger.warning(
            "Split contains zero valid entries. FPR is undefined and "
            "tier-weighted F1 reduces to weighted recall only. "
            "Metrics should be interpreted as recall-only diagnostics."
        )
        fpr = None

    # Compute bootstrap CIs if requested — single resampling loop for all 6 metrics
    detection_rate_ci = None
    f1_hallucination_ci = None
    tier_weighted_f1_ci = None
    fpr_ci = None
    ece_ci = None
    mcc_ci = None

    if compute_ci:
        all_cis = _bootstrap_all_cis(
            entries,
            predictions,
            n_bootstrap=n_bootstrap,
            seed=ci_seed,
            alpha=0.05,
        )
        detection_rate_ci = all_cis["detection_rate"]
        f1_hallucination_ci = all_cis["f1_hallucination"]
        tier_weighted_f1_ci = all_cis["tier_weighted_f1"]
        fpr_ci = all_cis["false_positive_rate"]
        ece_ci = all_cis["ece"]
        mcc_ci = all_cis["mcc"]

    f1_hallucination = cm.f1
    coverage_adjusted_f1 = f1_hallucination * coverage

    return EvaluationResult(
        tool_name=tool_name,
        split_name=split_name,
        num_entries=len(entries),
        num_hallucinated=num_hallucinated,
        num_valid=num_valid,
        detection_rate=cm.detection_rate,
        false_positive_rate=fpr,
        f1_hallucination=f1_hallucination,
        tier_weighted_f1=tw_f1,
        mcc=cm.mcc,
        macro_f1=cm.macro_f1,
        union_recall_at_k=dat_k,
        cost_efficiency=cost.get("entries_per_second"),
        mean_api_calls=cost.get("mean_api_calls"),
        ece=ece_score,
        auroc=auroc_score,
        auprc=auprc_score,
        num_uncertain=num_uncertain,
        per_tier_metrics=tier_metrics,
        per_type_metrics=type_metrics,
        detection_rate_ci=detection_rate_ci,
        f1_hallucination_ci=f1_hallucination_ci,
        tier_weighted_f1_ci=tier_weighted_f1_ci,
        fpr_ci=fpr_ci,
        ece_ci=ece_ci,
        mcc_ci=mcc_ci,
        coverage=coverage,
        coverage_adjusted_f1=coverage_adjusted_f1,
    )
