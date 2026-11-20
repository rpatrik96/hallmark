#!/usr/bin/env python3
"""bibtex-updater 1.2.0 standalone on VerusCite V1 (external evaluation only).

Same code path as scripts/regen_btu_v1_2_0.py (the run behind the paper's
bibtex-updater numbers): bibtex-check 1.2.0 with --rate-limit 120 --workers 8
--academic-only, then HALLMARK pre-screening + backfill. Only the input split
and the output directory differ. Re-running resumes from the bibtex-check cache.

Usage (1.2.0 must be first on PATH):
  PATH=~/.local/share/btu-1.2.0/bin:$PATH uv run python scripts/veruscite_v1/run_btu_v1_2_0.py [--max N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import regen_btu_v1_2_0 as regen  # noqa: E402

from hallmark.dataset.loader import load_split  # noqa: E402

DATASETS = {
    "v1": (REPO / "results/veruscite_v1/hm_data", REPO / "results/veruscite_v1/runs/btu_1.2.0"),
    "hall": (
        REPO / "results/hall_of_hallucinations/eval/hm_data",
        REPO / "results/hall_of_hallucinations/eval/runs/btu_1.2.0",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="v1")
    ap.add_argument("--max", type=int, default=0, help="smoke test on the first N entries")
    ap.add_argument(
        "--score-only",
        action="store_true",
        help="skip bibtex-check; dedupe the existing btu_raw.jsonl (last record per key) and score it",
    )
    a = ap.parse_args()

    exe = shutil.which("bibtex-check")
    ver = subprocess.run(
        [
            str(Path(exe).parent / "python"),
            "-c",
            "import importlib.metadata as m; print(m.version('bibtex-updater'))",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if ver != "1.2.0":
        sys.exit(f"bibtex-check on PATH is {ver!r} ({exe}), need 1.2.0")

    DATA_DIR, OUT = DATASETS[a.dataset]
    out = OUT if not a.max else OUT.parent / f"btu_1.2.0_smoke{a.max}"
    out.mkdir(parents=True, exist_ok=True)
    regen.DELTA_DIR = out
    regen.RESULTS_DIR = out

    def _load(split: str, version: str = "v1.2", **_: object) -> list:
        entries = load_split(split=split, version=version, data_dir=DATA_DIR)
        return entries[: a.max] if a.max else entries

    regen.load_split = _load
    if a.score_only:
        raw = out / "test_public" / "btu_raw.jsonl"
        last: dict[str, str] = {}
        for line in raw.read_text().splitlines():
            try:
                last[json.loads(line)["key"]] = line
            except (json.JSONDecodeError, KeyError):
                continue
        shutil.copy(raw, raw.with_suffix(".appended.jsonl"))
        raw.write_text("".join(v + "\n" for v in last.values()))
        print(f"deduped {raw}: {len(last)} keys")
        regen.run_bibtex_check_cached = lambda _blind, split: out / split / "btu_raw.jsonl"
    res = regen.run_btu_standalone("test_public")
    print(
        f"{a.dataset} btu 1.2.0 ({exe}): DR={res['detection_rate']} FPR={res['false_positive_rate']}"
    )


if __name__ == "__main__":
    main()
