"""HALLMARK: A benchmark for evaluating citation hallucination detection tools."""

try:
    from importlib.metadata import version

    __version__ = version("hallmark")
except Exception:
    __version__ = "0.0.0.dev0"

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import (
    BenchmarkEntry,
    BlindEntry,
    EvaluationResult,
    HallucinationType,
    Prediction,
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
    "load_split",
]
