"""Release the v1.2.0 prescreen-fix bibtex-updater aggregates.

The paper reports bibtex-updater from the rescored per-entry verdicts in
``btu_v1_2_0_prescreen_fix/`` (dev_public DR 0.820 / FPR 0.051 / F1 0.880,
test_public 0.838 / 0.080 / 0.889), while ``data/v1.2/baseline_results/``
still held the 2026-05-31 run those verdicts replaced (0.865 / 0.092 and
0.877 / 0.115). This rewrites the two released aggregates from the per-entry
files, keeping the published key set.

``pred_label`` already resolves the tool's abstentions to VALID, the paper's
convention for this tool, so ``evaluate`` sees full coverage. The released
``coverage`` and ``num_uncertain`` are kept as published: they are what
``scripts/rescore_btu_from_raw.py`` recomputes from the released raw output,
which counts the entries the tool never returned as declined (dev_public
0.8624). ``btu_v1_2_0_prescreen_fix/coverage.json`` does not (0.8686), and the
paper's Cov. column reads that file. ``coverage_adjusted_f1`` is
``f1_hallucination * coverage``, as defined in ``hallmark.evaluation.metrics``.

Usage:
    uv run python results/relabel_delta/release_btu_v120.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _rescore import load_new_entries
from _writer import merge_into_published

from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/relabel_delta/btu_v1_2_0_prescreen_fix"
OUT = REPO / "data/v1.2/baseline_results"


def load_predictions(path: Path) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        preds[d["bibtex_key"]] = Prediction(
            bibtex_key=d["bibtex_key"],
            label=d["pred_label"],
            confidence=float(d.get("pred_confidence") or 0.5),
            reason=d.get("reason", ""),
            source=d.get("source"),
            predicted_hallucination_type=d.get("pred_type"),
            cascade_stage=d.get("cascade_stage"),
        )
    return preds


def main() -> None:
    coverage = json.loads((SRC / "coverage.json").read_text())
    for split in ("dev_public", "test_public"):
        out_path = OUT / f"bibtexupdater_{split}.json"
        published = json.loads(out_path.read_text())
        entries = load_new_entries(split)
        preds = load_predictions(SRC / f"bibtexupdater_{split}_per_entry.jsonl")
        aligned = [preds[e.bibtex_key] for e in entries]
        result = evaluate(entries, aligned, "bibtexupdater", split, eval_mode="conservative")

        merged = merge_into_published(published, result)
        merged["coverage"] = published["coverage"]
        merged["num_uncertain"] = published["num_uncertain"]
        merged["coverage_adjusted_f1"] = merged["f1_hallucination"] * published["coverage"]
        prov = dict(merged.get("_provenance") or {})
        prov["rescored_from"] = str(
            (SRC / f"bibtexupdater_{split}_per_entry.jsonl").relative_to(REPO)
        )
        prov["rescored_input_sha256"] = coverage[split]["input_sha256"]
        merged["_provenance"] = prov
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
        print(
            f"{split}: DR {merged['detection_rate']:.3f} FPR {merged['false_positive_rate']:.3f} "
            f"F1 {merged['f1_hallucination']:.3f} MCC {merged['mcc']:.3f} coverage {merged['coverage']}"
        )


if __name__ == "__main__":
    main()
