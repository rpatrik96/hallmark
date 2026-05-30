"""Delta-eval DB/deterministic tools on ONLY the changed (relabel) keys.

The tool prediction on a key is independent of that key's ground-truth label,
so we run the registry baseline on just the <=27 dev + <=25 test changed keys to
recover each tool's verdict, then reconstruct the new aggregate by adjusting the
released confusion matrix. We persist the per-key verdicts for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from hallmark.baselines.registry import get_registry
from hallmark.dataset.schema import BenchmarkEntry

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
DELTA = REPO / "results/relabel_delta"


def load_changed_entries(split: str) -> list[BenchmarkEntry]:
    path = DELTA / f"changed_keys_{split}.jsonl"
    ents = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            ents.append(BenchmarkEntry.from_json(line))
    return ents


def run_tool_on_changed(baseline: str, split: str, reference_year: int | None = None, **kw):
    entries = load_changed_entries(split)
    info = get_registry()[baseline]
    if reference_year is not None:
        kw.setdefault("reference_year", reference_year)
    preds = info.runner(entries, **kw)
    out = {}
    for p in preds:
        out[p.bibtex_key] = {
            "label": p.label,
            "confidence": p.confidence,
            "reason": (p.reason or "")[:200],
            "source": p.source,
        }
    return out


def persist(baseline: str, split: str, verdicts: dict) -> Path:
    path = DELTA / f"{baseline}_{split}_changed_predictions.json"
    path.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    return path
