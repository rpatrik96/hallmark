"""HALLMARK: A benchmark for evaluating citation hallucination detection tools."""

try:
    from importlib.metadata import version

    __version__ = version("hallmark")
except Exception:
    __version__ = "0.0.0.dev0"
