#!/usr/bin/env python3
"""Fetch reference-audit's HALLMARK submissions, pinned to one commit.

reference-audit (constructorfabric/reference-audit, Nikita) publishes its
predictions under ``benchmarks/submissions/hallmark/`` in HALLMARK's
``Prediction`` schema, one manifest per split. This downloads them from raw
GitHub at a fixed commit, so no clone is needed, and refuses any file whose
SHA-256 differs from its manifest. It also checks that the manifest's blind
split hash matches our ``data/v1.2/<split>_blind.jsonl``, so the predictions
are known to cover the same entries we score.

Usage:
    uv run python scripts/fetch_reference_audit.py
    uv run python scripts/fetch_reference_audit.py --score   # also run hallmark evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = "constructorfabric/reference-audit"
PINNED_SHA = "f1cf36dfdffa9946b198c1d3aa45ecf2f33f5c40"
SUBMISSION_DIR = "benchmarks/submissions/hallmark"
MAPPING = "strict"
SPLITS = ("dev_public", "test_public", "stress_test")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{UPSTREAM}/{PINNED_SHA}/{SUBMISSION_DIR}/{name}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def fetch_split(split: str, out_dir: Path) -> Path:
    stem = f"reference-audit-{MAPPING}_{split}"
    manifest_bytes = fetch(f"{stem}.json")
    manifest = json.loads(manifest_bytes)
    preds = fetch(manifest["predictions_file"])

    if sha256(preds) != manifest["predictions_sha256"]:
        raise SystemExit(f"[{split}] predictions SHA-256 does not match the manifest")
    blind = REPO / "data" / "v1.2" / manifest["blind_split_file"]
    if sha256(blind.read_bytes()) != manifest["blind_split_sha256"]:
        raise SystemExit(f"[{split}] {blind.name} differs from the blind split the tool audited")

    (out_dir / f"{stem}.json").write_bytes(manifest_bytes)
    preds_path = out_dir / manifest["predictions_file"]
    preds_path.write_bytes(preds)
    print(
        f"[{split}] ok: pipeline {manifest['pipeline_version']}, {manifest['model']}, "
        f"partial_authors={manifest['partial_authors']}, labels={manifest['labels']}"
    )
    return preds_path


def score(split: str, preds_path: Path, out_dir: Path) -> None:
    subprocess.run(
        [
            "hallmark",
            "evaluate",
            "--split",
            split,
            "--predictions",
            str(preds_path),
            "--tool-name",
            f"reference-audit-{MAPPING}",
            "--eval-mode",
            "both",
            "--output",
            str(out_dir / f"reference_audit_{MAPPING}_{split}.json"),
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--out-dir", default="results/reference_audit/luna_pipeline_0.23", type=Path)
    ap.add_argument("--score", action="store_true", help="run hallmark evaluate on each split")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        preds_path = fetch_split(split, out_dir)
        if args.score:
            score(split, preds_path, out_dir)


if __name__ == "__main__":
    main()
