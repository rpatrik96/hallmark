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
import importlib.util
import json
import sys
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "data/v1.2/baseline_results"
_RESCORE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts/rescore_btu_from_raw.py"
_spec = importlib.util.spec_from_file_location("rescore_btu_from_raw", _RESCORE_SCRIPT)
assert _spec is not None and _spec.loader is not None
rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore)


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


@pytest.mark.parametrize(
    "aggregate", sorted(RESULTS.glob("bibtexupdater_*.json")), ids=lambda p: p.name
)
def test_every_released_bibtexupdater_aggregate_has_a_status_histogram(aggregate: Path):
    recorded = json.loads(aggregate.read_text()).get("_btu_status_histogram")
    assert recorded, f"{aggregate.name} records no status histogram to verify raw output against"


def _write_rescore_fixture(
    tmp_path: Path, *, entries: int, recorded: object
) -> tuple[Path, Path, Path]:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    aggregate = {
        "f1_hallucination": 0.5,
        "coverage": 1.0,
        "num_uncertain": 0,
        "coverage_adjusted_f1": 0.5,
    }
    if recorded is not ...:
        aggregate["_btu_status_histogram"] = recorded
    (results_dir / "bibtexupdater_dev_public.json").write_text(json.dumps(aggregate))

    data_dir = tmp_path / "data"
    split = data_dir / "v1.2" / "dev_public.jsonl"
    split.parent.mkdir(parents=True)
    records = [
        {
            "bibtex_key": f"k{index}",
            "bibtex_type": "article",
            "fields": {"title": "T", "author": "A", "year": "2024"},
            "label": "VALID",
        }
        for index in range(entries)
    ]
    split.write_text("".join(json.dumps(record) + "\n" for record in records))

    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"key":"k0","status":"verified"}\n')
    return results_dir, data_dir, raw


def _run_rescore(monkeypatch, results_dir: Path, data_dir: Path, raw: Path, *extra: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescore_btu_from_raw.py",
            "--split",
            "dev_public",
            "--raw",
            str(raw),
            "--results-dir",
            str(results_dir),
            "--data-dir",
            str(data_dir),
            *extra,
        ],
    )
    return rescore.main()


@pytest.mark.parametrize("recorded", [..., {}], ids=["missing", "empty"])
def test_rescore_refuses_an_unverifiable_raw(monkeypatch, tmp_path, recorded):
    results_dir, data_dir, raw = _write_rescore_fixture(tmp_path, entries=1, recorded=recorded)
    assert _run_rescore(monkeypatch, results_dir, data_dir, raw) == 1


def test_rescore_allows_explicit_unverified_raw_override(monkeypatch, tmp_path):
    results_dir, data_dir, raw = _write_rescore_fixture(tmp_path, entries=1, recorded=...)
    assert _run_rescore(monkeypatch, results_dir, data_dir, raw, "--allow-unverified-raw") == 0


def test_rescore_refuses_a_partial_raw_output(monkeypatch, tmp_path):
    results_dir, data_dir, raw = _write_rescore_fixture(
        tmp_path, entries=2, recorded={"verified": 1}
    )
    assert _run_rescore(monkeypatch, results_dir, data_dir, raw) == 1
