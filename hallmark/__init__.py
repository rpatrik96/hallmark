"""HALLMARK: A benchmark for evaluating citation hallucination detection tools."""

try:
    from importlib.metadata import version

    __version__ = version("hallmark")
except Exception:
    __version__ = "0.0.0.dev0"

from hallmark.baselines.registry import list_baselines
from hallmark.dataset.loader import (
    filter_by_date_range,
    filter_by_tier,
    filter_by_type,
    get_statistics,
    load_split,
)
from hallmark.dataset.schema import (
    BenchmarkEntry,
    BlindEntry,
    EvaluationResult,
    HallucinationType,
    Prediction,
    load_predictions,
    save_predictions,
)
from hallmark.evaluation.metrics import evaluate

__all__ = [
    "BenchmarkEntry",
    "BlindEntry",
    "EvaluationResult",
    "HallucinationType",
    "Prediction",
    "__version__",
    "evaluate",
    "filter_by_date_range",
    "filter_by_tier",
    "filter_by_type",
    "get_statistics",
    "list_baselines",
    "load_predictions",
    "load_split",
    "save_predictions",
]
