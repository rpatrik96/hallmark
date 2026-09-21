#!/usr/bin/env python3
"""Rescore bibtex-updater v1.2.0 with the fixed pre-screening DOI check.

The published bibtex-updater numbers (2026-05-31 run) predate 440d0d8
(2026-06-12), which normalizes versioned arXiv DataCite DOIs before the doi.org
HEAD check. Before the fix, ``10.48550/arXiv.2602.12271v1`` 404'd at doi.org and
pre-screening overrode the tool's verdict with HALLUCINATED.

Every pre-screening override in the released per-entry files carries a
versioned arXiv DOI (48 dev_public, 31 test_public). This script re-runs only
the fixed ``check_doi_resolves`` on those entries; the tool's own verdicts are
reused unchanged. arXiv DOIs are permanent DataCite registrations, so the lookup
answers as it would have in June with the fix in place.

- fixed check still returns HALLUCINATED (doi.org 404/410): the override stays
- fixed check returns VALID (the DOI resolves): the override is lifted and the
  tool's own verdict applies (btu_confidence / btu_abstained from the same row)
- fixed check returns UNKNOWN (timeout, 429, 5xx): retried; the script refuses
  to score while any lookup is still UNKNOWN, so a network error can never turn
  a detection into a miss

Outputs:
  results/relabel_delta/btu_v1_2_0_prescreen_fix/bibtexupdater_{split}_per_entry.jsonl
  results/relabel_delta/btu_v1_2_0_prescreen_fix/doi_recheck_{split}.jsonl
  results/relabel_delta/btu_v1_2_0_prescreen_fix/summary.json

Usage:
  uv run python scripts/rescore_btu_prescreening_fix.py
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

from hallmark.baselines.cascade import STAGE1_VERIFIED, STATUS_TO_TYPE
from hallmark.baselines.prescreening import check_doi_resolves
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results/relabel_delta/btu_v1_2_0"
OUT = REPO / "results/relabel_delta/btu_v1_2_0_prescreen_fix"
SPLITS = ("dev_public", "test_public")
MAX_ROUNDS = 5


def recheck(entries: list[BenchmarkEntry]) -> dict[str, dict]:
    """Run the fixed DOI check until every entry has a definite answer."""
    results: dict[str, dict] = {}
    pending = list(entries)
    for rnd in range(1, MAX_ROUNDS + 1):
        still = []
        for e in pending:
            r = check_doi_resolves(e.to_blind())
            if r.label == "UNKNOWN":
                still.append(e)
            results[e.bibtex_key] = {
                "bibtex_key": e.bibtex_key,
                "doi": e.fields.get("doi"),
                "label": r.label,
                "reason": r.reason,
                "round": rnd,
            }
        pending = still
        if not pending:
            break
        time.sleep(10 * rnd)
    return results


def status_label(status: str | None) -> str:
    """Status mapping used by Table 25 (``coverage_reporting.btu_status_to_label``)."""
    if status in STAGE1_VERIFIED:
        return "VALID"
    if status in STATUS_TO_TYPE:
        return "HALLUCINATED"
    return "UNCERTAIN"


def metrics(entries: list[BenchmarkEntry], preds: list[Prediction], split: str) -> dict:
    out = {}
    for mode in ("conservative", "aggressive"):
        r = evaluate(entries, preds, "bibtexupdater", split, eval_mode=mode)
        out[mode] = {
            "dr": r.detection_rate,
            "fpr": r.false_positive_rate,
            "f1": r.f1_hallucination,
            "num_uncertain": r.num_uncertain,
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "btu_version": "1.2.0 (verdicts reused from the 2026-05-31 run)",
        "prescreening_fix": "440d0d8",
    }
    for split in SPLITS:
        entries = load_split(split)
        by_key = {e.bibtex_key: e for e in entries}
        rows = [
            json.loads(line) for line in (SRC / f"bibtexupdater_{split}_per_entry.jsonl").open()
        ]
        overrides = [r for r in rows if r.get("source") == "prescreening_override"]

        checks = recheck([by_key[r["bibtex_key"]] for r in overrides])
        with (OUT / f"doi_recheck_{split}.jsonl").open("w") as f:
            for c in checks.values():
                f.write(json.dumps(c) + "\n")
        unknown = [k for k, c in checks.items() if c["label"] == "UNKNOWN"]
        if unknown:
            raise SystemExit(f"[{split}] {len(unknown)} DOI lookups still UNKNOWN: {unknown}")

        lifted = kept = 0
        new_rows = []
        for r in rows:
            c = checks.get(r["bibtex_key"])
            if c is not None and c["label"] == "VALID":
                # Tool's own verdict: verified / unconfirmed both map to VALID in the wrapper.
                r = {
                    **r,
                    "pred_label": "VALID",
                    "pred_confidence": r["btu_confidence"],
                    "pred_type": None,
                    "source": "tool",
                    "prescreen_fix": "override lifted",
                }
                lifted += 1
            elif c is not None:
                r = {**r, "prescreen_fix": "override kept"}
                kept += 1
            new_rows.append(r)
        with (OUT / f"bibtexupdater_{split}_per_entry.jsonl").open("w") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")

        def wrapper_preds(rs: list[dict]) -> list[Prediction]:
            return [
                Prediction(
                    bibtex_key=r["bibtex_key"],
                    label=r["pred_label"],
                    confidence=float(r["pred_confidence"]),
                )
                for r in rs
            ]

        def status_preds(rs: list[dict], with_prescreen: bool) -> list[Prediction]:
            out = []
            for r in rs:
                label = status_label(r.get("btu_status"))
                if with_prescreen and r.get("source") == "prescreening_override":
                    label = "HALLUCINATED"
                conf = float(r.get("btu_confidence") or 0.5)
                out.append(Prediction(bibtex_key=r["bibtex_key"], label=label, confidence=conf))
            return out

        gold = {e.bibtex_key: e.label for e in entries}
        summary[split] = {
            "overrides": len(overrides),
            "override_gold": {
                "VALID": sum(gold[r["bibtex_key"]] == "VALID" for r in overrides),
                "HALLUCINATED": sum(gold[r["bibtex_key"]] == "HALLUCINATED" for r in overrides),
            },
            "lifted": lifted,
            "kept": kept,
            "lifted_by_gold": {
                lab: sum(1 for k, c in checks.items() if c["label"] == "VALID" and gold[k] == lab)
                for lab in ("VALID", "HALLUCINATED")
            },
            "table1_wrapper": {
                "before": metrics(entries, wrapper_preds(rows), split)["conservative"],
                "after": metrics(entries, wrapper_preds(new_rows), split)["conservative"],
            },
            "table25_status_mapping": {
                "published_style_no_prescreening": metrics(
                    entries, status_preds(rows, with_prescreen=False), split
                ),
                "with_fixed_prescreening": metrics(
                    entries, status_preds(new_rows, with_prescreen=True), split
                ),
            },
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
