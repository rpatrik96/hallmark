"""Generator registry — single lookup replaces 60-line if/elif dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hallmark.dataset.schema import HallucinationType


@dataclass(frozen=True)
class GeneratorSpec:
    """Specification for a hallucination generator."""

    hall_type: HallucinationType
    func_name: (
        str  # dotted import path, e.g. "hallmark.dataset.generators.tier1.generate_fabricated_doi"
    )
    # Extra keyword arguments this generator requires beyond (entry, rng)
    extra_args: tuple[str, ...] = ()
    # Human-readable description
    description: str = ""


# Registry populated by register_generator() calls in tier modules.
# Stores (spec, callable) pairs.
_REGISTRY: dict[HallucinationType, tuple[GeneratorSpec, Callable[..., Any]]] = {}


def register_generator(
    hall_type: HallucinationType,
    *,
    extra_args: tuple[str, ...] = (),
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a generator function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = GeneratorSpec(
            hall_type=hall_type,
            func_name=f"{func.__module__}.{func.__qualname__}",
            extra_args=extra_args,
            description=description,
        )
        _REGISTRY[hall_type] = (spec, func)
        return func

    return decorator


def get_generator(hall_type: HallucinationType) -> GeneratorSpec:
    """Look up a registered generator spec."""
    if hall_type not in _REGISTRY:
        raise ValueError(
            f"No generator registered for {hall_type}. "
            f"Registered: {sorted(t.value for t in _REGISTRY)}"
        )
    return _REGISTRY[hall_type][0]


def get_generator_func(hall_type: HallucinationType) -> Callable[..., Any]:
    """Get the callable for a hallucination type."""
    if hall_type not in _REGISTRY:
        raise ValueError(
            f"No generator registered for {hall_type}. "
            f"Registered: {sorted(t.value for t in _REGISTRY)}"
        )
    return _REGISTRY[hall_type][1]


def all_generators() -> dict[HallucinationType, GeneratorSpec]:
    """Return all registered generator specs."""
    return {ht: spec for ht, (spec, _func) in _REGISTRY.items()}
