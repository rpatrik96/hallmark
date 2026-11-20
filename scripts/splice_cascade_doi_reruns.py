#!/usr/bin/env python3
"""Splice Stage-2 DOI re-runs into a cascade's per-entry predictions and rescore.

For ``cascade_<name>`` the re-run rows in
``results/relabel_recheck/doi_suffix_reruns/<rerun_tool>_<split>.jsonl`` replace
the rows of ``results/cascade_<name>/<prefix>_<split>_preds.jsonl`` for the keys
listed in ``doi_rerun_targets.json``; Stage-1 rows are never touched. Both eval
modes of ``<prefix>_<split>.json`` are recomputed (only keys it already has are
overwritten). Old/new labels per key go to
``results/relabel_recheck/doi_suffix_rerun_log.json``.

``--rerun-tool`` lets two cascades that share one Stage-2 run (cascade_gpt54 and
cascade_gpt54_shared) read the same re-run file.

Usage:
    uv run python scripts/splice_cascade_doi_reruns.py cascade_gpt51 [--dry-run]
    uv run python scripts/splice_cascade_doi_reruns.py cascade_gpt54_shared \\
        --rerun-tool cascade_gpt54
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "results/relabel_delta"))
from _writer import merge_into_published  # noqa: E402
from rescore_v1_2_5 import aligned, entries_at, fmt, load_predictions  # noqa: E402

from hallmark.evaluation.metrics import evaluate  # noqa: E402
from hallmark.evaluation.validate import compute_sha256  # noqa: E402

TARGETS = json.loads((REPO / "results/relabel_recheck/doi_rerun_targets.json").read_text())
RERUNS = REPO / "results/relabel_recheck/doi_suffix_reruns"
LOG = REPO / "results/relabel_recheck/doi_suffix_rerun_log.json"
SPLITS = ("dev_public", "test_public", "stress_test")
PREFIX = {
    "cascade_gpt51": "gpt51",
    "cascade_gpt54": "gpt54",
    "cascade_gpt54_shared": "gpt54",
    "cascade_sonnet46_shared": "sonnet46",
    "cascade_luna_shared": "luna",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cascade", choices=sorted(PREFIX))
    ap.add_argument("--rerun-tool", default=None)
    ap.add_argument(
        "--cascade-dir",
        type=Path,
        default=None,
        help="cascade results dir if not results/<cascade> here (luna lives in another checkout)",
    )
    ap.add_argument(
        "--json-suffix",
        default="",
        help="score into <prefix>_<split><suffix>.json, e.g. .v1_2_5 next to a v1.2.4 file",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rerun_tool = args.rerun_tool or args.cascade
    log = json.loads(LOG.read_text())
    now = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    cdir = args.cascade_dir or REPO / "results" / args.cascade
    for split in SPLITS:
        stem = f"{args.cascade}_{split}"
        keys = set(TARGETS.get(stem, {}).get("keys", []))
        if not keys:
            continue
        pred_path = cdir / f"{PREFIX[args.cascade]}_{split}_preds.jsonl"
        out_path = cdir / f"{PREFIX[args.cascade]}_{split}{args.json_suffix}.json"
        rerun_path = RERUNS / f"{rerun_tool}_{split}.jsonl"
        if not rerun_path.exists():
            failures.append(f"{stem}: re-run file missing ({rerun_path.relative_to(REPO)})")
            continue
        base = load_predictions(pred_path, "std")
        rerun = load_predictions(rerun_path, "std")
        if missing := keys - rerun.keys():
            failures.append(f"{stem}: re-run lacks {len(missing)} target keys")
            continue
        if not_s2 := [k for k in keys if base[k].cascade_stage != "stage2_diagnosis"]:
            failures.append(f"{stem}: {len(not_s2)} targets are not Stage-2 rows")
            continue
        spliced = {**base, **{k: rerun[k] for k in keys}}
        entries = entries_at(None, split)
        doc = json.loads(out_path.read_text())
        line = [f"{stem:40s}"]
        for mode in ("conservative", "aggressive"):
            r = evaluate(
                entries, aligned(entries, spliced), doc[mode]["tool_name"], split, eval_mode=mode
            )
            merged = merge_into_published(doc[mode], r)
            if "split_sha256" in merged:
                merged["split_sha256"] = compute_sha256(REPO / f"data/v1.2/{split}.jsonl")
            if "run_timestamp" in merged:
                merged["run_timestamp"] = now
            line.append(
                f"{mode[:4]} DR {fmt(doc[mode]['detection_rate'])}->{fmt(merged['detection_rate'])} "
                f"FPR {fmt(doc[mode]['false_positive_rate'])}->{fmt(merged['false_positive_rate'])}"
            )
            doc[mode] = merged
        moved = {k: (base[k].label, rerun[k].label) for k in sorted(keys)}
        changed = sum(a != b for a, b in moved.values())
        print(*line, f"changed {changed}/{len(keys)}", sep="  ")
        log["results"][stem] = {
            "rerun_keys": "doi_rerun_targets.json",
            "base_predictions": str(pred_path),
            "rerun": str(rerun_path.relative_to(REPO)),
            "n_rerun": len(keys),
            "verdict_changes": changed,
            "per_key": {k: {"before": a, "after": b} for k, (a, b) in moved.items()},
            "spliced_at": now,
        }
        if not args.dry_run:
            # Keep every existing row, in file order (excluded entries included).
            pred_path.write_text("".join(spliced[k].to_json() + "\n" for k in base))
            out_path.write_text(json.dumps(doc, ensure_ascii=False))
        # The released cascade_db_diagnosis{,_aggressive} results are scored from
        # the sonnet46_shared predictions (rescore_v1_2_5.CASCADE); keep them in step.
        if args.cascade == "cascade_sonnet46_shared":
            for released, mode in (
                (f"cascade_db_diagnosis_{split}", "conservative"),
                (f"cascade_db_diagnosis_aggressive_{split}", "aggressive"),
            ):
                rpath = REPO / "data/v1.2/baseline_results" / f"{released}.json"
                published = json.loads(rpath.read_text())
                r = evaluate(
                    entries,
                    aligned(entries, spliced),
                    published["tool_name"],
                    split,
                    eval_mode=mode,
                )
                merged = merge_into_published(published, r)
                if "split_sha256" in merged:
                    merged["split_sha256"] = compute_sha256(REPO / f"data/v1.2/{split}.jsonl")
                if "run_timestamp" in merged:
                    merged["run_timestamp"] = now
                print(
                    f"  {released:46s} DR {fmt(published['detection_rate'])}->"
                    f"{fmt(merged['detection_rate'])} FPR {fmt(published['false_positive_rate'])}->"
                    f"{fmt(merged['false_positive_rate'])}"
                )
                if not args.dry_run:
                    rpath.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    if failures:
        print("\nREFUSED:", *failures, sep="\n  ")
    if not args.dry_run:
        LOG.write_text(json.dumps(log, indent=2) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
