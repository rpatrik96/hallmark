#!/usr/bin/env python3
"""Refresh the HALLMARK HuggingFace mirror (hallmark-neurips2026/HALLMARK).

Mirrors the LIVE hub layout (which differs from the repo layout):

    jsonl/<split>.jsonl            <- data/v1.0/<split>.jsonl   (6 files)
    data/<split>.parquet           <- built from jsonl           (3 files)
    blind/<split>_blind.parquet    <- built from jsonl           (3 files)
    sources/llm_generated.jsonl    <- data/v1.0/llm_generated.jsonl
    metadata.json                  <- data/v1.0/metadata.json
    source_mapping.json            <- data/v1.0/source_mapping.json
    valid_entry_verification.json  <- data/v1.0/valid_entry_verification.json
    croissant.json                 <- croissant.json
    README.md                      <- hf/README.md (dataset card, tracked here)

Parquet files are rebuilt deterministically (pandas + snappy) and their SHA-256
is verified against croissant.json before anything is uploaded; metadata.json
is verified the same way. A mismatch aborts the run. Everything ships as one
hub commit; unchanged blobs are deduplicated server-side.

baseline_results/ on the hub is a separate artifact set and is not touched.

Usage:
    python scripts/upload_to_hf.py [--repo-name hallmark-neurips2026/HALLMARK]
                                   [--token <hf_token>] [--dry-run]

Without --token the cached `hf auth login` credential is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "v1.0"

JSONL_SPLITS = [
    "dev_public",
    "dev_public_blind",
    "test_public",
    "test_public_blind",
    "stress_test",
    "stress_test_blind",
]

# repo source -> hub path
STATIC_FILES = {
    DATA_DIR / "metadata.json": "metadata.json",
    DATA_DIR / "source_mapping.json": "source_mapping.json",
    DATA_DIR / "valid_entry_verification.json": "valid_entry_verification.json",
    DATA_DIR / "llm_generated.jsonl": "sources/llm_generated.jsonl",
    ROOT / "croissant.json": "croissant.json",
    ROOT / "hf" / "README.md": "README.md",
}

PARQUET_HUB_PATHS = {
    "dev_public": "data/dev_public.parquet",
    "test_public": "data/test_public.parquet",
    "stress_test": "data/stress_test.parquet",
    "dev_public_blind": "blind/dev_public_blind.parquet",
    "test_public_blind": "blind/test_public_blind.parquet",
    "stress_test_blind": "blind/stress_test_blind.parquet",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def croissant_checksums() -> dict[str, str]:
    """Hub path -> expected sha256, from croissant.json's distribution block."""
    spec = json.loads((ROOT / "croissant.json").read_text(encoding="utf-8"))
    expected = {}
    for obj in spec["distribution"]:
        url = obj.get("contentUrl", "")
        marker = "/resolve/main/"
        if marker in url and "sha256" in obj:
            expected[url.split(marker, 1)[1]] = obj["sha256"]
    return expected


def build_parquets(dst: Path) -> dict[Path, str]:
    import pandas as pd

    ops: dict[Path, str] = {}
    for split, hub_path in PARQUET_HUB_PATHS.items():
        records = [
            json.loads(line)
            for line in (DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out = dst / hub_path
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(out, index=False, compression="snappy")
        ops[out] = hub_path
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the HALLMARK HuggingFace mirror")
    parser.add_argument("--repo-name", default="hallmark-neurips2026/HALLMARK")
    parser.add_argument("--token", default=None, help="HF token (default: cached login)")
    parser.add_argument("--dry-run", action="store_true", help="verify + list, upload nothing")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="hallmark-hf-"))
    uploads: dict[Path, str] = dict(build_parquets(tmp))
    for split in JSONL_SPLITS:
        uploads[DATA_DIR / f"{split}.jsonl"] = f"jsonl/{split}.jsonl"
    for src, hub_path in STATIC_FILES.items():
        uploads[src] = hub_path

    missing = [str(p) for p in uploads if not p.exists()]
    if missing:
        sys.exit(f"missing source files: {missing}")

    expected = croissant_checksums()
    by_hub = {hub: src for src, hub in uploads.items()}
    for hub_path, want in expected.items():
        got = sha256(by_hub[hub_path])
        print(f"checksum {'ok' if got == want else 'MISMATCH'}: {hub_path}")
        if got != want:
            sys.exit(
                f"{hub_path}: sha256 {got} does not match croissant.json ({want}); "
                "regenerate croissant.json or investigate non-determinism before uploading"
            )

    for src, hub_path in sorted(uploads.items(), key=lambda kv: kv[1]):
        rel = src.relative_to(ROOT) if src.is_relative_to(ROOT) else src
        print(f"  {hub_path}  <-  {rel}")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=args.token)
    print("uploading as:", api.whoami()["name"])
    version = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))["version"]
    info = api.create_commit(
        repo_id=args.repo_name,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=hub, path_or_fileobj=str(src))
            for src, hub in uploads.items()
        ],
        commit_message=f"Refresh mirror to corpus v{version}",
    )
    print("commit:", info.commit_url)


if __name__ == "__main__":
    main()
