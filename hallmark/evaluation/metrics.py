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


def evaluate(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    tool_name: str = "unknown",
    split_name: str = "unknown",
    predictions_per_strategy: list[dict[str, Prediction]] | None = None,
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

    Returns:
        EvaluationResult with primary metrics (DR, FPR, F1, TW-F1, ECE),
        per-tier and per-type breakdowns, and count of UNCERTAIN predictions.
    """
    pred_map = {p.bibtex_key: p for p in predictions}

    cm = build_confusion_matrix(entries, pred_map)
    tw_f1 = tier_weighted_f1(entries, pred_map)
    tier_metrics = per_tier_metrics(entries, pred_map)
    type_metrics = per_type_metrics(entries, pred_map)
    cost = cost_efficiency(predictions)
    ece_score = expected_calibration_error(entries, pred_map)

    dat_k: dict[int, float] = {}
    if predictions_per_strategy:
        dat_k = detect_at_k(entries, predictions_per_strategy)

    num_hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")
    num_valid = sum(1 for e in entries if e.label == "VALID")
    num_uncertain = sum(1 for p in predictions if p.label == "UNCERTAIN")

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
        num_uncertain=num_uncertain,
        per_tier_metrics=tier_metrics,
        per_type_metrics=type_metrics,
    )
