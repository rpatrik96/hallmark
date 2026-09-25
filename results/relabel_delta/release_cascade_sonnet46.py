"""Release the 2026-09-21 bibtex-updater -> Sonnet 4.6 cascade aggregates.

The ``cascade_db_diagnosis`` family in ``data/v1.2/baseline_results/`` held the
2026-05-31 run of this cascade. The paper reports the 2026-09-21 rerun in
``results/cascade_sonnet46_shared/``, whose stage 1 is the fixed-prescreening
bibtex-updater pass (``results/cascade_gpt51/*_preds.jsonl``) and whose stage 2
is a fresh Sonnet 4.6 call (test_public MCC 0.938 conservative, 0.941
aggressive). This writes the conservative result to ``cascade_db_diagnosis_*``
and the aggressive one to ``cascade_db_diagnosis_aggressive_*`` for every split
the rerun covers, keeping the rerun's own provenance and recording its source.

Usage:
    uv run python results/relabel_delta/release_cascade_sonnet46.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/cascade_sonnet46_shared"
OUT = REPO / "data/v1.2/baseline_results"
SPLITS = ("dev_public", "test_public", "stress_test")
FAMILY = {"conservative": "cascade_db_diagnosis", "aggressive": "cascade_db_diagnosis_aggressive"}


def main() -> None:
    for split in SPLITS:
        src = SRC / f"sonnet46_{split}.json"
        run = json.loads(src.read_text())
        for mode, stem in FAMILY.items():
            result = dict(run[mode])
            prov = dict(run.get("_provenance") or {})
            prov["eval_mode"] = mode
            prov["released_from"] = str(src.relative_to(REPO))
            result["_provenance"] = prov
            out_path = OUT / f"{stem}_{split}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(
                f"{stem}_{split}: DR {result['detection_rate']:.3f} "
                f"FPR {result.get('false_positive_rate') or 0:.3f} "
                f"F1 {result['f1_hallucination']:.3f} MCC {result.get('mcc') or 0:.3f}"
            )


if __name__ == "__main__":
    main()
