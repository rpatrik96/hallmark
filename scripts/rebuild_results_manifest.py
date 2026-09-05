#!/usr/bin/env python3
"""Rebuild a baseline-results manifest so it covers every file present.  [evaluation]

``validate-results`` iterates ``manifest["files"]``, so a result the manifest
does not list is not checksummed and can be edited undetectably. The manifest
covered 16 of 45 released results, and the 29 it missed included every LLM row
in the main table -- they were written by paths that do not update it
(``run_all_baselines.py`` and ad-hoc scripts), while only
``generate_reference_results.py`` maintains it.

This rebuilds the file list from what is on disk, preserving the existing
``version`` and ``environment`` blocks and any per-file metadata already
recorded. It does not run baselines and does not touch the results themselves.

Usage
-----
    uv run python scripts/rebuild_results_manifest.py                 # dry run
    uv run python scripts/rebuild_results_manifest.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.evaluation.validate import compute_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "v1.2" / "baseline_results"


def _entry_for(path: Path) -> dict[str, Any]:
    """Manifest record for one result file, mirroring generate_reference_results."""
    record: dict[str, Any] = {"sha256": compute_sha256(path)}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return record
    # Dual-mode payloads nest {"conservative": ..., "aggressive": ...}.
    probe = data.get("conservative", data) if isinstance(data, dict) else data
    if not isinstance(probe, dict):
        return record
    for src, dst in (
        ("tool_name", "baseline"),
        ("split_name", "split"),
        ("num_entries", "num_entries"),
        ("f1_hallucination", "f1_hallucination"),
        ("split_sha256", "split_sha256"),
        ("tool_version", "tool_version"),
    ):
        value = probe.get(src)
        if value is not None:
            record[dst] = value
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--apply", action="store_true", help="write the manifest (default: dry run)")
    args = ap.parse_args()

    manifest_path = args.results_dir / "manifest.json"
    if not args.results_dir.is_dir():
        ap.error(f"results dir not found: {args.results_dir}")

    manifest: dict[str, Any] = {"version": "unknown", "files": {}, "environment": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("files", {})

    present = sorted(p for p in args.results_dir.glob("*.json") if p.name != "manifest.json")
    before = set(manifest["files"])
    rebuilt: dict[str, Any] = {}
    for path in present:
        record = _entry_for(path)
        # Keep any fields the existing entry had that we cannot recompute.
        existing = manifest["files"].get(path.name, {})
        merged = {**existing, **record}
        rebuilt[path.name] = merged

    added = sorted(set(rebuilt) - before)
    dropped = sorted(before - set(rebuilt))
    changed = sorted(
        n
        for n in set(rebuilt) & before
        if manifest["files"][n].get("sha256") != rebuilt[n].get("sha256")
    )

    print(f"  present on disk : {len(present)}")
    print(f"  manifest before : {len(before)}")
    print(f"  added           : {len(added)}")
    print(f"  dropped         : {len(dropped)} {dropped if dropped else ''}")
    print(f"  checksum changed: {len(changed)}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    manifest["files"] = rebuilt
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  Wrote {manifest_path.relative_to(REPO_ROOT)} covering {len(rebuilt)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
