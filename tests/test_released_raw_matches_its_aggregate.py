"""A released raw output must describe the aggregate shipped beside it.

``data/v1.2/baseline_results/bibtexupdater_raw_dev_public.jsonl`` was a
0.10.0-era run -- statuses ``hallucinated``, ``partial_match``, ``doi_not_found``
and no ``unconfirmed`` at all -- sitting next to an aggregate whose
``_btu_status_histogram`` records thirteen 1.2.0 statuses including 147
``unconfirmed``. Anyone deriving a figure from the raw file would have computed
it against a different run than the one the paper reports, and nothing in the
release said so.

The aggregate carries its own histogram, so the check costs one pass over the
file. It is the general form of the guard in
``scripts/rescore_btu_from_raw.py``, which caught this by refusing to write.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "data/v1.2/baseline_results"


def _raw_aggregate_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for raw in sorted(RESULTS.glob("*_raw_*.jsonl")):
        tool, _, split = raw.stem.partition("_raw_")
        aggregate = RESULTS / f"{tool}_{split}.json"
        if aggregate.is_file():
            pairs.append((raw, aggregate))
    return pairs


def _histogram(raw: Path) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for line in raw.read_text().splitlines():
        if line.strip():
            counts[json.loads(line).get("status", "")] += 1
    return dict(counts)


@pytest.mark.parametrize(
    "raw,aggregate",
    _raw_aggregate_pairs(),
    ids=lambda p: p.name if isinstance(p, Path) else str(p),
)
def test_raw_output_describes_its_aggregate(raw: Path, aggregate: Path):
    if raw.read_text(errors="ignore").startswith("version https://git-lfs"):
        pytest.skip(f"{raw.name} is an unfetched LFS pointer")
    recorded = json.loads(aggregate.read_text()).get("_btu_status_histogram")
    if recorded is None:
        pytest.skip(f"{aggregate.name} records no status histogram to check against")
    observed = _histogram(raw)
    assert observed == recorded, (
        f"{raw.name} is not the run behind {aggregate.name}.\n"
        f"  raw file : {observed}\n"
        f"  aggregate: {recorded}\n"
        "A figure derived from this file would describe a different run than the "
        "paper reports, and the release would not say so."
    )


def test_at_least_one_pair_is_checked():
    """Otherwise the parametrisation is empty and this file asserts nothing."""
    assert _raw_aggregate_pairs(), "no raw/aggregate pair found -- the guard is inert"
