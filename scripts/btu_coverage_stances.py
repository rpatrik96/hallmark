#!/usr/bin/env python3
"""bibtex-updater coverage and scoring stances under the harness's abstention definition.

Reads the rescored per-entry files written by ``rescore_btu_prescreening_fix.py``
and scores each split three ways:

- selective:    declined entries excluded
- conservative: declined entries scored as VALID
- aggressive:   declined entries scored as HALLUCINATED

Declined = ``btu_status`` in ``ABSTENTION_STATUSES`` (hallmark/baselines/bibtexupdater.py,
defined as of commit c608307), imported rather than copied so the numbers follow
the code if the list changes. The CLI's own ``abstained`` flag is not used: it
also marks ``not_found`` records, which are detections.

Rows without a ``btu_status`` (7 dev_public, 2 test_public) are tool VALID
verdicts and count as committed.

Output: results/relabel_delta/btu_v1_2_0_prescreen_fix/coverage.json

Usage:
    uv run python scripts/btu_coverage_stances.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hallmark.baselines.bibtexupdater import ABSTENTION_STATUSES

REPO = Path(__file__).resolve().parent.parent
RESCORE = REPO / "results/relabel_delta/btu_v1_2_0_prescreen_fix"
OUT = RESCORE / "coverage.json"


def metrics(pairs: list[tuple[str, str]]) -> dict:
    assert all(p in "HV" and g in "HV" for p, g in pairs), "unexpected label"
    tp = sum(p == "H" and g == "H" for p, g in pairs)
    fn = sum(p == "V" and g == "H" for p, g in pairs)
    fp = sum(p == "H" and g == "V" for p, g in pairs)
    tn = sum(p == "V" and g == "V" for p, g in pairs)
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return {
        "dr": round(rec, 4),
        "fpr": round(fp / (fp + tn), 4),
        "f1": round(2 * prec * rec / (prec + rec), 4),
        "n": len(pairs),
    }


def score_split(name: str) -> dict:
    path = RESCORE / f"bibtexupdater_{name}_per_entry.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]

    def declined(r: dict) -> bool:
        return r.get("btu_status") in ABSTENTION_STATUSES

    def gold(r: dict) -> str:
        return str(r["gold_label"][0])

    def pred(r: dict) -> str:
        return str(r["pred_label"][0])

    return {
        "input": str(path.relative_to(REPO)),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n": len(rows),
        "n_declined": sum(declined(r) for r in rows),
        "coverage": round(sum(not declined(r) for r in rows) / len(rows), 4),
        "selective": metrics([(pred(r), gold(r)) for r in rows if not declined(r)]),
        "conservative": metrics([("V" if declined(r) else pred(r), gold(r)) for r in rows]),
        "aggressive": metrics([("H" if declined(r) else pred(r), gold(r)) for r in rows]),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    result = {
        "definition": "declined = btu_status in hallmark ABSTENTION_STATUSES",
        "abstention_statuses": sorted(ABSTENTION_STATUSES),
        "abstention_statuses_source": "hallmark/baselines/bibtexupdater.py @ c608307",
        "not_used": "CLI `abstained` flag (also marks not_found detections)",
        "hallmark_commit": git("rev-parse", "--short", "HEAD"),
        # The output file itself is excluded, so a rerun does not mark its own result dirty
        "working_tree_dirty": bool(
            git("status", "--porcelain", "--", ".", f":!{OUT.relative_to(REPO)}")
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dev_public": score_split("dev_public"),
        "test_public": score_split("test_public"),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("dev_public", "test_public")}, indent=2))


if __name__ == "__main__":
    main()
