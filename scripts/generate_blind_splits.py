#!/usr/bin/env python3
"""Generate blind versions of HALLMARK benchmark data files.

Blind files contain only the fields visible to participants (bibtex_key,
bibtex_type, fields, raw_bibtex). Ground-truth labels, hallucination types,
generation metadata, and subtests are stripped.

Usage:
    python scripts/generate_blind_splits.py [--data-dir data/v1.0/]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running without installing the package by adding repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hallmark.dataset.schema import BenchmarkEntry  # noqa: E402

# Splits to process; hidden_test.jsonl is deliberately excluded.
TARGET_SPLITS = [
    "dev_public.jsonl",
    "test_public.jsonl",
    "stress_test.jsonl",
]


def generate_blind_file(src: Path, dst: Path) -> int:
    """Convert *src* JSONL to a blind JSONL at *dst*.

    Returns the number of entries written.
    """
    entries_written = 0
    with open(src, encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for lineno, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = BenchmarkEntry.from_json(line)
            except Exception as exc:
                raise ValueError(f"Failed to parse {src}:{lineno}: {exc}") from exc
            blind = entry.to_blind()
            blind_dict = asdict(blind)
            f_out.write(json.dumps(blind_dict, ensure_ascii=False) + "\n")
            entries_written += 1
    return entries_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate blind (label-free) versions of HALLMARK benchmark splits."
    )
    parser.add_argument(
        "--data-dir",
        default="data/v1.0/",
        help="Directory containing the source JSONL files (default: data/v1.0/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Source directory: {data_dir}")
    print()

    generated: list[tuple[str, int]] = []
    for split_filename in TARGET_SPLITS:
        src = data_dir / split_filename
        if not src.exists():
            print(f"  [SKIP] {split_filename} — file not found")
            continue

        stem = src.stem  # e.g. "dev_public"
        dst_filename = f"{stem}_blind.jsonl"
        dst = data_dir / dst_filename

        n = generate_blind_file(src, dst)
        generated.append((dst_filename, n))
        print(f"  {split_filename} -> {dst_filename}  ({n} entries)")

    print()
    print(f"Generated {len(generated)} blind file(s):")
    for filename, count in generated:
        print(f"  {data_dir / filename}  [{count} lines]")


if __name__ == "__main__":
    main()
