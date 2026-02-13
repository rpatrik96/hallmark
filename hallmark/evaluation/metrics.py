"""Evaluation metrics for HALLMARK.

Primary: Detection Rate (DR), False Positive Rate (FPR), F1-Hallucination, Tier-weighted F1, ECE.
Secondary: detect@k, temporal robustness, cost efficiency, per-type metrics, subtest accuracy.
Inspired by HumanEval (multi-criteria, pass@k) and ONEBench (sample-level atomic eval).

Evaluation Protocol:
--------------------

1. **Prediction Labels**:
   - Tools must return VALID or HALLUCINATED predictions
   - UNCERTAIN is accepted as a valid label (treated as VALID conservatively)
   - Missing predictions are treated as VALID (conservative default)

2. **Pre-screening**:
   - Baseline wrappers may include pre-screening (lightweight local checks before external APIs)
   - Pre-screening results are included in the tool's predictions and reported transparently
   - The `reason` field indicates pre-screening with `[Pre-screening override]` prefix

3. **Handling Incomplete Evaluations**:
   - Tools may evaluate only a subset of benchmark entries (e.g., due to API limits)
   - For single-tool evaluation: metrics computed only on covered entries
   - For multi-tool ranking: use Plackett-Luce model (see hallmark.evaluation.ranking)
     to fairly rank tools with heterogeneous coverage

4. **Metrics Computation**:
   - UNCERTAIN predictions treated as VALID for all metrics
   - Tier-weighted F1: harder hallucinations (Tier 3) weighted 3x vs Tier 1
   - ECE (Expected Calibration Error): measures confidence calibration
   - Per-type metrics: detection rate by hallucination type
   - Subtest accuracy: accuracy on individual verification checks (DOI, title, authors, etc.)

5. **Cost Tracking**:
   - wall_clock_seconds: total time per prediction
   - api_calls: number of external API calls made
   - cost_efficiency: entries evaluated per second
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from hallmark.dataset.schema import BenchmarkEntry, EvaluationResult, Prediction


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


def build_confusion_matrix(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> ConfusionMatrix:
    """Build confusion matrix from entries and predictions.

    UNCERTAIN predictions are treated as VALID (conservative default).
    Missing predictions are also treated as VALID.
    """
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

        # Treat UNCERTAIN as VALID (conservative)
        pred_label = "VALID" if pred.label == "UNCERTAIN" else pred.label

        if entry.label == "HALLUCINATED":
            if pred_label == "HALLUCINATED":
                cm.tp += 1
            else:
                cm.fn += 1
        else:
            if pred_label == "HALLUCINATED":
                cm.fp += 1
            else:
                cm.tn += 1
    return cm


def tier_weighted_f1(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
    tier_weights: dict[int, float] | None = None,
) -> float:
    """Compute F1 weighted by difficulty tier (Tier 3 worth 3x Tier 1).

    Each hallucinated entry contributes to F1 proportionally to its tier weight.
    UNCERTAIN predictions are treated as VALID (conservative).
    """
    if tier_weights is None:
        tier_weights = {1: 1.0, 2: 2.0, 3: 3.0}

    weighted_tp = 0.0
    weighted_fn = 0.0
    weighted_fp = 0.0
    total_weight = 0.0

    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        tier = entry.difficulty_tier or 1
        w = tier_weights.get(tier, 1.0)

        # Treat UNCERTAIN as VALID
        pred_label = pred.label if pred and pred.label != "UNCERTAIN" else "VALID"

        if entry.label == "HALLUCINATED":
            total_weight += w
            if pred is not None and pred_label == "HALLUCINATED":
                weighted_tp += w
            else:
                weighted_fn += w
        else:
            if pred is not None and pred_label == "HALLUCINATED":
                weighted_fp += w

    precision = (
        weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    )
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def detect_at_k(
    entries: list[BenchmarkEntry],
    predictions_per_strategy: list[dict[str, Prediction]],
    k: int | None = None,
) -> dict[int, float]:
    """Compute detect@k: fraction of hallucinations detected using k strategies.

    Analogous to HumanEval's pass@k. A hallucination is "detected" if ANY of the
    first k strategies flags it.

    UNCERTAIN predictions are treated as VALID (conservative, not counted as detections).

    Args:
        entries: Benchmark entries.
        predictions_per_strategy: List of prediction dicts, one per verification strategy.
        k: If provided, compute only for this k. Otherwise compute for k=1..len(strategies).

    Returns:
        Dict mapping k -> fraction of hallucinations detected by at least one of k strategies.
    """
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


def per_tier_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[int, dict[str, float]]:
    """Compute metrics broken down by difficulty tier."""
    tier_entries: dict[int, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        tier = entry.difficulty_tier or 0
        tier_entries[tier].append(entry)

    result = {}
    for tier, tier_e in sorted(tier_entries.items()):
        cm = build_confusion_matrix(tier_e, predictions)
        result[tier] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "precision": cm.precision,
            "count": len(tier_e),
        }
    return result


def per_type_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by hallucination type."""
    type_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        type_entries[h_type].append(entry)

    result = {}
    for h_type, type_e in sorted(type_entries.items()):
        cm = build_confusion_matrix(type_e, predictions)
        result[h_type] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "count": len(type_e),
        }
    return result


def per_generation_method_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by generation method."""
    method_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        method = entry.generation_method or "unknown"
        method_entries[method].append(entry)

    result = {}
    for method, method_e in sorted(method_entries.items()):
        cm = build_confusion_matrix(method_e, predictions)
        result[method] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "count": len(method_e),
        }
    return result


def cost_efficiency(predictions: list[Prediction]) -> dict[str, float]:
    """Compute cost efficiency metrics."""
    if not predictions:
        return {"entries_per_second": 0.0, "mean_api_calls": 0.0}

    total_time = sum(p.wall_clock_seconds for p in predictions)
    total_api_calls = sum(p.api_calls for p in predictions)
    n = len(predictions)

    return {
        "entries_per_second": n / total_time if total_time > 0 else float("inf"),
        "mean_api_calls": total_api_calls / n,
        "total_wall_clock_seconds": total_time,
    }


def expected_calibration_error(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - mean_confidence| per bin,
    returns weighted average by bin size.

    For HALLUCINATED predictions, confidence means P(hallucinated).
    For VALID predictions, confidence means P(valid), so we use 1-confidence for
    the "correctness" probability when the prediction is VALID.

    UNCERTAIN predictions are treated as VALID with the reported confidence.
    """
    if not predictions:
        return 0.0

    # Collect (confidence, correctness) pairs
    pairs: list[tuple[float, bool]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        if pred is None:
            continue

        # Treat UNCERTAIN as VALID
        pred_label = "VALID" if pred.label == "UNCERTAIN" else pred.label
        is_correct = pred_label == entry.label
        pairs.append((pred.confidence, is_correct))

    if not pairs:
        return 0.0

    # Create bins
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    # Compute ECE
    ece = 0.0
    total = len(pairs)
    for bin_data in bins:
        if not bin_data:
            continue
        bin_size = len(bin_data)
        avg_conf = sum(conf for conf, _ in bin_data) / bin_size
        accuracy = sum(1 for _, correct in bin_data if correct) / bin_size
        ece += (bin_size / total) * abs(accuracy - avg_conf)

    return ece


def auroc(entries: list[BenchmarkEntry], predictions: dict[str, Prediction]) -> float | None:
    """Compute Area Under ROC Curve for hallucination detection.

    Positive class: HALLUCINATED. Score = confidence for HALLUCINATED predictions,
    (1 - confidence) for VALID predictions. UNCERTAIN treated as VALID.
    Missing predictions treated as VALID with confidence 0.5.
    Returns None if fewer than 2 classes present.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.

    Returns:
        AUROC score in [0, 1], or None if insufficient class diversity.
    """
    # Build (score, true_label) pairs where true_label is 1 for HALLUCINATED, 0 for VALID
    pairs: list[tuple[float, int]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)

        # Determine score (probability of HALLUCINATED)
        if pred is None:
            score = 0.5  # Missing prediction -> neutral
        elif pred.label == "UNCERTAIN":
            score = 0.5  # UNCERTAIN treated as VALID but with neutral confidence
        elif pred.label == "HALLUCINATED":
            score = pred.confidence
        else:  # VALID
            score = 1.0 - pred.confidence

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


def auprc(entries: list[BenchmarkEntry], predictions: dict[str, Prediction]) -> float | None:
    """Compute Area Under Precision-Recall Curve for hallucination detection.

    Same scoring convention as auroc(). Returns None if no positive examples.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.

    Returns:
        AUPRC score in [0, 1], or None if no positive examples.
    """
    # Build (score, true_label) pairs where true_label is 1 for HALLUCINATED, 0 for VALID
    pairs: list[tuple[float, int]] = []
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)

        # Determine score (probability of HALLUCINATED)
        if pred is None:
            score = 0.5  # Missing prediction -> neutral
        elif pred.label == "UNCERTAIN":
            score = 0.5  # UNCERTAIN treated as VALID but with neutral confidence
        elif pred.label == "HALLUCINATED":
            score = pred.confidence
        else:  # VALID
            score = 1.0 - pred.confidence

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
    # For PR curve, we compute area using the points we actually visit
    tp = 0
    fp = 0
    area = 0.0
    prev_recall = 0.0

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
            recall = tp / num_positive
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Use the current precision for the entire recall interval
            # This is the standard way to compute AUPRC (right-hand Riemann sum)
            area += precision * (recall - prev_recall)

            prev_recall = recall

    return area


def source_stratified_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, dict[str, float]]:
    """Compute detection metrics stratified by API sources queried.

    Groups predictions by their api_sources_queried field and computes
    per-source detection rate and false positive rate.

    Returns dict mapping source combination string to metrics dict.
    """
    # Group predictions by source combination
    source_groups: dict[str, list[str]] = defaultdict(list)
    for key, pred in predictions.items():
        source_key = (
            ",".join(sorted(pred.api_sources_queried)) if pred.api_sources_queried else "none"
        )
        source_groups[source_key].append(key)

    result = {}
    for source_key, keys in sorted(source_groups.items()):
        # Filter entries that have predictions in this group
        group_entries = [e for e in entries if e.bibtex_key in keys]
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


def subtest_accuracy_table(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, dict[str, float]]:
    """Compute per-subtest accuracy across all entries.

    For each subtest name, computes accuracy (fraction of entries where
    the prediction's subtest result matches the ground truth subtest).

    Returns dict mapping subtest name to {accuracy, count, true_positives, false_positives}.
    """
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

    rng = np.random.RandomState(seed)

    # Group entries and predictions by hallucination type
    type_groups: dict[str, tuple[list[BenchmarkEntry], list[Prediction]]] = defaultdict(
        lambda: ([], [])
    )
    pred_map = {p.bibtex_key: p for p in predictions}

    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        pred = pred_map.get(entry.bibtex_key)
        if pred is not None:
            type_groups[h_type][0].append(entry)
            type_groups[h_type][1].append(pred)

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


def paired_bootstrap_test(
    entries: list[BenchmarkEntry],
    predictions_a: list[Prediction],
    predictions_b: list[Prediction],
    metric_fn: Callable[[list[BenchmarkEntry], list[Prediction]], float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap significance test.

    Returns (observed_diff, p_value, effect_size_cohens_h).
    H0: metric_A <= metric_B. p-value = fraction of resamples where delta <= 0.

    Args:
        entries: Benchmark entries (ground truth).
        predictions_a: Predictions from tool A.
        predictions_b: Predictions from tool B.
        metric_fn: Function that takes (entries, predictions) and returns a scalar metric.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (observed_diff, p_value, effect_size_cohens_h).
        - observed_diff: metric_A - metric_B on original data
        - p_value: P(delta <= 0 | H0) under bootstrap distribution
        - effect_size_cohens_h: Cohen's h effect size (2 * (arcsin(sqrt(p_a)) - arcsin(sqrt(p_b))))
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for bootstrap tests: pip install numpy") from e

    rng = np.random.RandomState(seed)

    # Compute observed difference
    metric_a = metric_fn(entries, predictions_a)
    metric_b = metric_fn(entries, predictions_b)
    observed_diff = metric_a - metric_b

    # Build prediction maps
    pred_map_a = {p.bibtex_key: p for p in predictions_a}
    pred_map_b = {p.bibtex_key: p for p in predictions_b}

    # Bootstrap resampling (paired - same indices for both)
    bootstrap_diffs = []
    n = len(entries)

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        resampled_entries = [entries[i] for i in indices]

        # Resample predictions using same indices (filter out None)
        resampled_preds_a = [
            p for i in indices if (p := pred_map_a.get(entries[i].bibtex_key)) is not None
        ]
        resampled_preds_b = [
            p for i in indices if (p := pred_map_b.get(entries[i].bibtex_key)) is not None
        ]

        if resampled_preds_a and resampled_preds_b:
            metric_a_boot = metric_fn(resampled_entries, resampled_preds_a)
            metric_b_boot = metric_fn(resampled_entries, resampled_preds_b)
            bootstrap_diffs.append(metric_a_boot - metric_b_boot)

    # p-value: fraction of bootstrap samples where delta <= 0
    p_value = float(np.mean([d <= 0 for d in bootstrap_diffs])) if bootstrap_diffs else 1.0

    # Cohen's h effect size
    cohens_h = 2 * (np.arcsin(np.sqrt(metric_a)) - np.arcsin(np.sqrt(metric_b)))

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

    rng = np.random.RandomState(seed)

    # Build prediction map
    pred_map = {p.bibtex_key: p for p in predictions}

    # Filter predictions for each entry set
    preds_a = [pred_map[e.bibtex_key] for e in entries_a if e.bibtex_key in pred_map]
    preds_b = [pred_map[e.bibtex_key] for e in entries_b if e.bibtex_key in pred_map]

    # Compute observed difference (using F1 as default metric)
    cm_a = build_confusion_matrix(entries_a, pred_map)
    cm_b = build_confusion_matrix(entries_b, pred_map)
    observed_diff = cm_a.f1 - cm_b.f1

    # Permutation test
    combined_entries = entries_a + entries_b
    combined_preds = preds_a + preds_b
    n_a = len(entries_a)

    permuted_diffs = []
    for _ in range(n_permutations):
        # Permute combined data
        perm_indices = rng.permutation(len(combined_entries))
        perm_entries = [combined_entries[i] for i in perm_indices]
        perm_preds = [combined_preds[i] for i in perm_indices if i < len(combined_preds)]

        # Split permuted data
        perm_entries_a = perm_entries[:n_a]
        perm_entries_b = perm_entries[n_a:]
        perm_preds_a = perm_preds[:n_a]
        perm_preds_b = perm_preds[n_a:]

        # Compute metric on permuted split
        perm_pred_map_a = {p.bibtex_key: p for p in perm_preds_a}
        perm_pred_map_b = {p.bibtex_key: p for p in perm_preds_b}

        if perm_entries_a and perm_entries_b:
            cm_perm_a = build_confusion_matrix(perm_entries_a, perm_pred_map_a)
            cm_perm_b = build_confusion_matrix(perm_entries_b, perm_pred_map_b)
            permuted_diffs.append(cm_perm_a.f1 - cm_perm_b.f1)

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


def evaluate(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    tool_name: str = "unknown",
    split_name: str = "unknown",
    predictions_per_strategy: list[dict[str, Prediction]] | None = None,
    compute_ci: bool = False,
) -> EvaluationResult:
    """Run full evaluation and return aggregated results.

    Evaluation Protocol:
    - UNCERTAIN predictions are treated as VALID (conservative default) for all metrics
    - Missing predictions are treated as VALID
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

    Returns:
        EvaluationResult with primary metrics (DR, FPR, F1, TW-F1, ECE),
        per-tier and per-type breakdowns, count of UNCERTAIN predictions,
        and optionally bootstrap CIs for primary metrics.
    """
    pred_map = {p.bibtex_key: p for p in predictions}

    cm = build_confusion_matrix(entries, pred_map)
    tw_f1 = tier_weighted_f1(entries, pred_map)
    tier_metrics = per_tier_metrics(entries, pred_map)
    type_metrics = per_type_metrics(entries, pred_map)
    cost = cost_efficiency(predictions)
    ece_score = expected_calibration_error(entries, pred_map)
    auroc_score = auroc(entries, pred_map)
    auprc_score = auprc(entries, pred_map)

    dat_k: dict[int, float] = {}
    if predictions_per_strategy:
        dat_k = detect_at_k(entries, predictions_per_strategy)

    num_hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")
    num_valid = sum(1 for e in entries if e.label == "VALID")
    num_uncertain = sum(1 for p in predictions if p.label == "UNCERTAIN")

    # Compute bootstrap CIs if requested
    detection_rate_ci = None
    f1_hallucination_ci = None
    tier_weighted_f1_ci = None
    fpr_ci = None
    ece_ci = None

    if compute_ci:
        # Detection Rate CI
        def dr_metric(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            cm_boot = build_confusion_matrix(ents, {p.bibtex_key: p for p in preds})
            return cm_boot.detection_rate

        detection_rate_ci = stratified_bootstrap_ci(
            entries, predictions, dr_metric, n_bootstrap=10_000
        )

        # F1-Hallucination CI
        def f1_metric(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            cm_boot = build_confusion_matrix(ents, {p.bibtex_key: p for p in preds})
            return cm_boot.f1

        f1_hallucination_ci = stratified_bootstrap_ci(
            entries, predictions, f1_metric, n_bootstrap=10_000
        )

        # Tier-weighted F1 CI
        def tw_f1_metric(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            return tier_weighted_f1(ents, {p.bibtex_key: p for p in preds})

        tier_weighted_f1_ci = stratified_bootstrap_ci(
            entries, predictions, tw_f1_metric, n_bootstrap=10_000
        )

        # FPR CI
        def fpr_metric(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            cm_boot = build_confusion_matrix(ents, {p.bibtex_key: p for p in preds})
            return cm_boot.false_positive_rate

        fpr_ci = stratified_bootstrap_ci(entries, predictions, fpr_metric, n_bootstrap=10_000)

        # ECE CI
        def ece_metric(ents: list[BenchmarkEntry], preds: list[Prediction]) -> float:
            return expected_calibration_error(ents, {p.bibtex_key: p for p in preds})

        ece_ci = stratified_bootstrap_ci(entries, predictions, ece_metric, n_bootstrap=10_000)

    return EvaluationResult(
        tool_name=tool_name,
        split_name=split_name,
        num_entries=len(entries),
        num_hallucinated=num_hallucinated,
        num_valid=num_valid,
        detection_rate=cm.detection_rate,
        false_positive_rate=cm.false_positive_rate,
        f1_hallucination=cm.f1,
        tier_weighted_f1=tw_f1,
        detect_at_k=dat_k,
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
    )
