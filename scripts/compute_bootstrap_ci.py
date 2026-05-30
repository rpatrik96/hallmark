#!/usr/bin/env python3
"""Compute & persist stratified bootstrap CIs and paired significance.  [evaluation]

This is the reporting-path wiring for task #9. The paper claims that bootstrap
CIs and paired significance p-values underlie every numerical claim, yet the
``*_ci`` fields in the released artifacts are ``null`` because nothing ever
invoked the computing code. This script closes that gap.

It is *principled and deterministic*:

* CIs are computed from **per-entry predictions** (the only honest source) via
  :func:`hallmark.evaluation.metrics.compute_persisted_cis`. Reconstructing
  outcome vectors from aggregate detection rates -- as an earlier version did --
  is lossy and cannot recover the paired structure needed for significance
  tests, so we never do that.
* When a tool has no stored per-entry predictions, its ``*_ci`` fields stay
  ``null`` and a ``ci_provenance`` block records *why*. We do not fabricate.
* Paired significance (observed diff, p-value, Cohen's h) is computed for every
  tool pair sharing the same split.

By default the script writes augmented copies to ``--output-dir`` and leaves the
released result JSONs untouched (table regeneration is a separate, later stage).
Pass ``--in-place`` to overwrite the source result files.

Usage:
    # Augment every result JSON that has a matching per-entry prediction file:
    python scripts/compute_bootstrap_ci.py \
        --results-dir data/v1.0/baseline_results \
        --predictions-dir data/v1.0/baseline_results \
        --output-dir results/ci

    # Single result + explicit prediction file:
    python scripts/compute_bootstrap_ci.py \
        --result data/v1.0/baseline_results/llm_tool_augmented_dev_public.json \
        --predictions data/v1.0/baseline_results/llm_tool_augmented_dev_public.jsonl \
        --split dev_public --output-dir results/ci
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import compute_persisted_cis, paired_significance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KNOWN_SPLITS = ("dev_public", "test_public", "stress_test", "test_hidden")


def infer_split_from_name(name: str) -> str | None:
    """Infer the split a result/prediction filename scores, from its suffix.

    Files are named ``<tool>_<split>.json`` / ``<tool>_<split>.jsonl``. The
    longest matching known split wins (so ``test_public`` is preferred over a
    hypothetical ``public``).
    """
    stem = Path(name).stem
    matches = [s for s in KNOWN_SPLITS if stem == s or stem.endswith(f"_{s}")]
    if not matches:
        return None
    return max(matches, key=len)


def load_predictions_canonical(path: Path) -> list[Prediction] | None:
    """Load a per-entry prediction JSONL file as canonical ``Prediction`` objects.

    Returns ``None`` (with a logged reason) if the file does not parse as the
    canonical Prediction schema -- e.g. tool-native ``*_raw_*.jsonl`` dumps that
    use ``{"key", "status", ...}`` instead of ``{"bibtex_key", "label", ...}``.
    We deliberately do not guess a mapping here; unparseable files yield null CIs.
    """
    preds: list[Prediction] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                preds.append(Prediction.from_json(line))
            except Exception as exc:  # report & bail, do not guess a mapping
                logger.warning(
                    "Prediction file %s line %d not in canonical schema (%s); "
                    "treating tool as having no usable per-entry predictions.",
                    path,
                    lineno,
                    exc,
                )
                return None
    return preds or None


def find_prediction_file(predictions_dir: Path, result_path: Path) -> Path | None:
    """Find a per-entry prediction JSONL matching a result JSON, if any."""
    stem = result_path.stem  # e.g. "llm_tool_augmented_dev_public"
    candidate = predictions_dir / f"{stem}.jsonl"
    if candidate.exists():
        return candidate
    return None


def augment_one(
    result_path: Path,
    predictions: list[Prediction] | None,
    entries: list[BenchmarkEntry],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict:
    """Return an augmented result-JSON payload with persisted ``*_ci`` fields."""
    payload = json.loads(result_path.read_text())
    ci_block = compute_persisted_cis(
        entries,
        predictions,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    payload.update(ci_block)
    return payload


def run(
    *,
    results: list[Path],
    predictions_dir: Path | None,
    explicit_predictions: Path | None,
    explicit_split: str | None,
    output_dir: Path | None,
    in_place: bool,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    version: str,
    data_dir: str | None,
) -> int:
    """Augment result JSONs with CIs and emit a per-split significance table."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Cache entries per split so we load each split once.
    entries_cache: dict[str, list[BenchmarkEntry]] = {}

    def entries_for(split: str) -> list[BenchmarkEntry]:
        if split not in entries_cache:
            entries_cache[split] = load_split(split, version, data_dir)
        return entries_cache[split]

    # Group tools by split for the significance table.
    preds_by_split: dict[str, dict[str, list[Prediction]]] = defaultdict(dict)
    n_with_ci = 0
    n_null_ci = 0

    for result_path in results:
        split = explicit_split or infer_split_from_name(result_path.name)
        if split is None:
            logger.warning("Cannot infer split for %s; skipping.", result_path)
            continue

        entries = entries_for(split)

        # Resolve per-entry predictions.
        preds: list[Prediction] | None
        if explicit_predictions is not None:
            preds = load_predictions_canonical(explicit_predictions)
        elif predictions_dir is not None:
            pred_file = find_prediction_file(predictions_dir, result_path)
            preds = load_predictions_canonical(pred_file) if pred_file else None
        else:
            preds = None

        payload = augment_one(
            result_path,
            preds,
            entries,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence,
        )

        provenance = payload.get("ci_provenance", {})
        if provenance.get("computed"):
            n_with_ci += 1
            tool = str(payload.get("tool_name") or result_path.stem)
            if preds is not None:
                preds_by_split[split][tool] = preds
            logger.info(
                "  %s [%s]: CIs computed from %d predictions (DR CI=%s).",
                result_path.name,
                split,
                len(preds or []),
                payload.get("detection_rate_ci"),
            )
        else:
            n_null_ci += 1
            logger.info(
                "  %s [%s]: CIs left null -- %s",
                result_path.name,
                split,
                provenance.get("reason"),
            )

        # Persist augmented payload.
        if in_place:
            out_path = result_path
        elif output_dir is not None:
            out_path = output_dir / result_path.name
        else:
            continue  # dry run: compute & log only
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # Paired significance per split (only tools with usable predictions).
    significance: dict[str, dict[str, dict[str, float]]] = {}
    for split, tool_preds in preds_by_split.items():
        if len(tool_preds) < 2:
            continue
        logger.info(
            "Computing paired significance for %d tools on %s ...",
            len(tool_preds),
            split,
        )
        significance[split] = paired_significance(
            entries_for(split),
            tool_preds,
            metric="f1_hallucination",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    if significance and output_dir is not None:
        sig_path = output_dir / "paired_significance.json"
        sig_path.write_text(json.dumps(significance, ensure_ascii=False, indent=2))
        logger.info("Paired significance written to %s", sig_path)

    logger.info(
        "Done: %d result(s) with computed CIs, %d left null (no per-entry predictions).",
        n_with_ci,
        n_null_ci,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--result", type=Path, help="Single result JSON to augment.")
    src.add_argument(
        "--results-dir",
        type=Path,
        help="Directory of <tool>_<split>.json result files to augment.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Explicit per-entry prediction JSONL (use with --result).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory of <tool>_<split>.jsonl per-entry prediction files.",
    )
    parser.add_argument(
        "--split",
        choices=KNOWN_SPLITS,
        help="Force the split (otherwise inferred from filename suffix).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write augmented copies here (released JSONs stay untouched).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source result files instead of writing to --output-dir.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    if args.result is not None:
        results = [args.result]
    else:
        results = sorted(
            p
            for p in args.results_dir.glob("*.json")
            if p.name != "manifest.json" and infer_split_from_name(p.name) is not None
        )

    if not args.in_place and args.output_dir is None:
        logger.warning("Neither --in-place nor --output-dir given: dry run (compute & log only).")

    rc = run(
        results=results,
        predictions_dir=args.predictions_dir,
        explicit_predictions=args.predictions,
        explicit_split=args.split,
        output_dir=args.output_dir,
        in_place=args.in_place,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        confidence=args.confidence,
        version=args.version,
        data_dir=args.data_dir,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
