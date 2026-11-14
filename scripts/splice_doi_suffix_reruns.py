#!/usr/bin/env python3
"""Splice re-runs of the 138 DOI-suffix entries into released predictions and rescore.

Until v1.2.5 (``eacc9af``), 138 public entries (dev_public 83, test_public 55)
carried ``10.48550/arXiv.XXXX.XXXXXv1``. The versioned form 404s at doi.org, so
any tool that resolves DOIs judged a citation whose DOI did not exist. Labels
were right (real papers); the input was broken.

For each tool given on the command line, the released per-entry predictions are
taken as the base (they must already be on the v1.2.5 labels via
``scripts/rescore_v1_2_5.py``), every row for a DOI-suffix key is replaced with
the re-run made on the v1.2.5 input, and the aggregate is recomputed with
``evaluate`` and merged into the released JSON (only keys it already has).
All 138 entries are re-run, not only some verdicts: the 404 pushed tools to
HALLUCINATED on hallucinated entries too, so detections there were not earned.

Re-run files: ``results/relabel_recheck/doi_suffix_reruns/<tool>_<split>.jsonl``.
Spliced predictions: ``results/relabel_recheck/doi_suffix_reruns/<tool>_<split>_spliced.jsonl``.
Log: ``results/relabel_recheck/doi_suffix_rerun_log.json``.

Usage:
    uv run python scripts/splice_doi_suffix_reruns.py doi_only [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "results/relabel_delta"))
from _writer import merge_into_published  # noqa: E402
from rescore_v1_2_5 import SOURCES, aligned, entries_at, fmt, load_predictions  # noqa: E402

from hallmark.dataset.schema import Prediction  # noqa: E402
from hallmark.evaluation.metrics import evaluate  # noqa: E402
from hallmark.evaluation.validate import compute_sha256  # noqa: E402

SUFFIX_COMMIT = "eacc9af"
RERUNS = REPO / "results/relabel_recheck/doi_suffix_reruns"
LOG = REPO / "results/relabel_recheck/doi_suffix_rerun_log.json"
TARGETS = json.loads((REPO / "results/relabel_recheck/doi_rerun_targets.json").read_text())
RESULTS = REPO / "data/v1.2/baseline_results"
SPLITS = ("dev_public", "test_public")


def suffix_keys(split: str) -> set[str]:
    """Keys whose ``fields.doi`` the DOI-suffix fix changed."""
    rel = f"data/v1.2/{split}.jsonl"

    def doi_map(rev: str) -> dict[str, str | None]:
        text = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        rows = (json.loads(line) for line in text.splitlines() if line.strip())
        return {r["bibtex_key"]: r["fields"].get("doi") for r in rows}

    before, after = doi_map(f"{SUFFIX_COMMIT}~1"), doi_map(SUFFIX_COMMIT)
    return {k for k in before if k in after and before[k] != after[k]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tools", nargs="+", help="result-file tool stem, e.g. doi_only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = (
        json.loads(LOG.read_text())
        if LOG.exists()
        else {"suffix_commit": SUFFIX_COMMIT, "results": {}}
    )
    now = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    for tool in args.tools:
        for split in SPLITS:
            stem = f"{tool}_{split}"
            if stem not in SOURCES:
                failures.append(f"{stem}: no released prediction source in rescore_v1_2_5.SOURCES")
                continue
            base_rel, mode, loader = SOURCES[stem]
            rerun_path = RERUNS / f"{stem}.jsonl"
            if not rerun_path.exists():
                failures.append(f"{stem}: re-run file missing ({rerun_path.relative_to(REPO)})")
                continue
            # doi_only re-ran all 138 DOI-suffix keys; the agentic tools re-run only
            # their resolve_doi targets (results/relabel_recheck/doi_rerun_targets.json).
            keys = set(TARGETS[stem]["keys"]) if stem in TARGETS else suffix_keys(split)
            base = load_predictions(REPO / base_rel, loader)
            rerun = load_predictions(rerun_path, "std")
            missing = keys - rerun.keys()
            if missing:
                failures.append(f"{stem}: re-run lacks {len(missing)} DOI-suffix keys")
                continue
            moved = {
                k: (base[k].label if k in base else None, rerun[k].label) for k in sorted(keys)
            }
            spliced: dict[str, Prediction] = {**base, **{k: rerun[k] for k in keys}}

            entries = entries_at(None, split)
            out_path = RESULTS / f"{stem}.json"
            published = json.loads(out_path.read_text())
            result = evaluate(
                entries, aligned(entries, spliced), published["tool_name"], split, eval_mode=mode
            )
            merged = merge_into_published(published, result)
            if loader == "btu":
                merged["coverage"] = published["coverage"]
                merged["num_uncertain"] = published["num_uncertain"]
                merged["coverage_adjusted_f1"] = merged["f1_hallucination"] * published["coverage"]
            merged["split_sha256"] = compute_sha256(REPO / f"data/v1.2/{split}.jsonl")
            if "run_timestamp" in merged:
                merged["run_timestamp"] = now

            spliced_path = RERUNS / f"{stem}_spliced.jsonl"
            log["results"][stem] = {
                "rerun_keys": "doi_rerun_targets.json"
                if stem in TARGETS
                else "all DOI-suffix keys",
                "base_predictions": base_rel,
                "rerun": str(rerun_path.relative_to(REPO)),
                "spliced_predictions": str(spliced_path.relative_to(REPO)),
                "eval_mode": mode,
                "n_rerun": len(keys),
                "verdict_changes": sum(old != new for old, new in moved.values()),
                "before": {
                    k: published.get(k)
                    for k in ("detection_rate", "false_positive_rate", "f1_hallucination")
                },
                "after": {
                    k: merged.get(k)
                    for k in ("detection_rate", "false_positive_rate", "f1_hallucination")
                },
                "per_key": {k: {"before": o, "after": n} for k, (o, n) in moved.items()},
            }
            print(
                f"{stem:44s} DR {fmt(published['detection_rate'])}->{fmt(merged['detection_rate'])} "
                f"FPR {fmt(published['false_positive_rate'])}->{fmt(merged['false_positive_rate'])} "
                f"changed {log['results'][stem]['verdict_changes']}/{len(keys)}"
            )
            if not args.dry_run:
                spliced_path.write_text(
                    "".join(
                        spliced[e.bibtex_key].to_json() + "\n"
                        for e in entries
                        if e.bibtex_key in spliced
                    )
                )
                out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")

    if failures:
        print("\nREFUSED:", *failures, sep="\n  ")
    if not args.dry_run:
        LOG.write_text(json.dumps(log, indent=2) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
