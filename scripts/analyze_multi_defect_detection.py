#!/usr/bin/env python3
"""Compare detection on *multi-defect* vs *single-defect* hallucinated entries.

Motivation
----------
``scripts/analyze_defect_multiplicity.py`` establishes that a HALLMARK entry's
single ``hallucination_type`` names the injected *cause*, not the number of
broken fields: 28-38% of hallucinated entries fail two or more per-field
subtests. ``docs/multi_defect_handling.md`` argues (§4b) that single-defect
entries are the *harder* case, so headline detection rates are a lower bound on
real-world (mostly multi-defect) performance. That argument was never measured.

This script measures it, per zero-shot model, on the per-entry prediction dumps.

What it reports
---------------
1. **Raw split** — detection rate on entries with exactly 1 defective field vs
   >= 2, with Wilson CIs, a Newcombe CI on the difference, and Fisher's exact p.
2. **Type-stratified split** — the raw contrast is badly confounded: defect
   count is nearly a deterministic function of ``hallucination_type``
   (``plausible_fabrication`` is 100% multi-defect, ``wrong_venue`` 100%
   single). Mantel-Haenszel pools the within-type risk differences over the
   few types that carry *both* kinds of entry, which is the only place the
   defect-count effect is identified separately from the type effect.
3. **Generation-method-stratified split** — same idea, coarser strata
   (``perturbation`` and ``llm_generated`` carry both kinds in bulk).
4. **Dose-response** — detection rate by exact defect count (1/2/3/4) with a
   Cochran-Armitage trend test.

Scoring protocol matches ``build_confusion_matrix()``
(``hallmark/evaluation/metrics.py``): a prediction of ``UNCERTAIN`` is dropped
from the denominator entirely; a *missing* prediction counts as a miss.

Usage
-----
    uv run python scripts/analyze_multi_defect_detection.py
    uv run python scripts/analyze_multi_defect_detection.py --splits test_public
    uv run python scripts/analyze_multi_defect_detection.py --json --out results/multi_defect/detection.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_defect_multiplicity import (  # (path shim above must precede this import)
    DEFAULT_DATA_DIR,
    defective_fields,
    load_split,
)

#: Splits carrying ground-truth ``subtests``, longest name first so suffix
#: stripping does not leave a ``_matched`` tail behind.
SPLITS = [
    "test_crossdomain_matched",
    "test_crossdomain",
    "dev_public",
    "test_public",
    "stress_test",
]

#: ``test_crossdomain_matched`` is a re-matched subsample of the same underlying
#: crossdomain references as ``test_crossdomain``; scoring both double-counts the
#: same predictions, so it is opt-in via ``--splits``.
DEFAULT_SPLITS = ["dev_public", "test_public", "stress_test", "test_crossdomain"]

#: Where each split's entries live, relative to the repo root.
SPLIT_PATHS = {
    "test_crossdomain_matched": Path(
        "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl"
    ),
}

#: Per-entry zero-shot prediction dumps available in this checkout. The split is
#: inferred from bibtex_key overlap, not from the path.
PREDICTION_GLOBS = [
    "results/checkpoints/*/*.jsonl",
    "results/new_models/*.jsonl",
    "results/crossdomain_llms/llm_*.jsonl",
    "results/crossdomain_matched_llms/llm_*.jsonl",
    "results/cascade_gpt54/*_preds.jsonl",
    "results/cascade_gpt51/*_preds.jsonl",
    "results/relabel_delta/btu_v1_2_0/*_per_entry.jsonl",
]

#: Checkpoint dir / file stem -> display name for the model.
MODEL_NAMES = {
    "llm_openai": "GPT-5.1",
    "llm_openrouter_claude_sonnet_4_6": "Claude Sonnet 4.6",
    "llm_openrouter_claude_opus_4_7": "Claude Opus 4.7",
    "llm_openrouter_deepseek_r1": "DeepSeek R1",
    "llm_openrouter_deepseek_v3": "DeepSeek V3.2",
    "llm_openrouter_gemini_flash": "Gemini 2.5 Flash",
    "llm_openrouter_gemini_pro": "Gemini 2.5 Pro",
    "llm_openrouter_llama_4_maverick": "Llama 4 Maverick",
    "llm_openrouter_mistral": "Mistral Large",
    "llm_openrouter_qwen": "Qwen3-235B-A22B",
    "llm_openrouter_qwen_max": "Qwen3 Max",
    "llm_hf_qwen3_4b": "Qwen3-4B",
    "llm_hf_qwen3_8b": "Qwen3-8B",
    "llm_hf_qwen3_14b": "Qwen3-14B",
    "llm_hf_qwen3_32b": "Qwen3-32B",
    "llm_openai_gpt54": "GPT-5.4",
    "gpt54": "GPT-5.4 + BTU (cascade)",
    "gpt51": "GPT-5.1 + BTU (cascade)",
    "bibtexupdater": "bibtex-updater",
    "llm_openrouter_claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_pro": "Gemini 2.5 Pro",
    "llama4_maverick": "Llama 4 Maverick",
    "qwen_max": "Qwen3 Max",
}


# --------------------------------------------------------------------------- stats


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid-score CI for p1 - p2 (independent samples)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    lower = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def mantel_haenszel_rd(strata: dict[str, tuple[int, int, int, int]]) -> dict[str, Any]:
    """Pooled risk difference over strata, each ``(k1, n1, k0, n0)``.

    Group 1 is multi-defect, group 0 single-defect. Uses MH weights
    ``w = n1*n0/(n1+n0)`` and the Greenland-Robins variance. Strata missing
    either arm contribute nothing and are dropped (they carry no within-stratum
    information about defect count) — which is most of them, since defect count
    is nearly a function of hallucination type.
    """
    used = {name: cells for name, cells in strata.items() if cells[1] > 0 and cells[3] > 0}
    if not used:
        return {
            "rd": float("nan"),
            "ci": (float("nan"), float("nan")),
            "p": float("nan"),
            "cmh_p": float("nan"),
            "num_strata": 0,
            "num_strata_dropped": len(strata),
            "strata": [],
        }

    cells = list(used.values())
    w_total = sum(n1 * n0 / (n1 + n0) for _, n1, _, n0 in cells)
    rd = sum((n1 * n0 / (n1 + n0)) * (k1 / n1 - k0 / n0) for k1, n1, k0, n0 in cells) / w_total
    var = (
        sum(
            (k1 * (n1 - k1) * n0**3 + k0 * (n0 - k0) * n1**3) / (n1 * n0 * (n1 + n0) ** 2)
            for k1, n1, k0, n0 in cells
        )
        / w_total**2
    )
    se = math.sqrt(var)
    # se == 0 means every surviving stratum is saturated (both arms all-hit or
    # all-miss). The point estimate is real but has no sampling spread to report,
    # so emit NaN rather than a zero-width interval that reads as certainty.
    z = rd / se if se > 0 else float("nan")
    ci = (rd - 1.959964 * se, rd + 1.959964 * se) if se > 0 else (float("nan"), float("nan"))

    # Cochran-Mantel-Haenszel test of the common-odds-ratio null.
    num = sum(k1 - n1 * (k1 + k0) / (n1 + n0) for k1, n1, k0, n0 in cells)
    den = sum(
        (n1 * n0 * (k1 + k0) * ((n1 + n0) - (k1 + k0))) / ((n1 + n0) ** 2 * ((n1 + n0) - 1))
        for k1, n1, k0, n0 in cells
        if (n1 + n0) > 1
    )
    cmh_p = float(stats.chi2.sf(num**2 / den, 1)) if den > 0 else float("nan")

    return {
        "rd": rd,
        "ci": ci,
        "se": se,
        "z": z,
        # Wald p from the Greenland-Robins SE — the one consistent with `ci`.
        # CMH tests the common-odds-ratio null instead and is conservative when
        # strata are sparse, so the two can disagree; both are kept.
        "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan"),
        "cmh_p": cmh_p,
        "num_strata": len(used),
        "num_strata_dropped": len(strata) - len(used),
        # MH weight n1*n0/(n1+n0) says how much each stratum actually moves the
        # pooled estimate — without it a 12-vs-20 stratum looks like a 47-vs-5 one.
        "strata": [
            {
                "name": name,
                "k_multi": k1,
                "n_multi": n1,
                "dr_multi": k1 / n1,
                "k_single": k0,
                "n_single": n0,
                "dr_single": k0 / n0,
                "rd": k1 / n1 - k0 / n0,
                "mh_weight": n1 * n0 / (n1 + n0),
            }
            for name, (k1, n1, k0, n0) in used.items()
        ],
    }


def cochran_armitage(counts: list[tuple[int, int, float]]) -> dict[str, float]:
    """Trend test over ordered dose levels, each ``(detected, n, dose)``."""
    used = [(k, n, d) for k, n, d in counts if n > 0]
    n_total = sum(n for _, n, _ in used)
    k_total = sum(k for k, _, _ in used)
    if n_total == 0 or k_total in (0, n_total) or len(used) < 2:
        return {"z": float("nan"), "p": float("nan")}
    p_bar = k_total / n_total
    d_bar = sum(n * d for _, n, d in used) / n_total
    num = sum(d * (k - n * p_bar) for k, n, d in used)
    var = p_bar * (1 - p_bar) * sum(n * (d - d_bar) ** 2 for _, n, d in used)
    if var <= 0:
        return {"z": float("nan"), "p": float("nan")}
    z = num / math.sqrt(var)
    return {"z": z, "p": float(2 * stats.norm.sf(abs(z)))}


# ------------------------------------------------------------------- data loading


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load a per-entry prediction JSONL, last write per key winning.

    Checkpoint files are append-on-resume, so a key can appear twice; the later
    record is the one the evaluator's dict-comprehension would keep.
    """
    preds: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # bibtex-updater's per-entry dump names its verdict ``pred_label``;
        # everything else uses ``label``. Normalise so one scorer handles both.
        if "label" not in rec and "pred_label" in rec:
            rec = {**rec, "label": rec["pred_label"]}
        if "bibtex_key" in rec and rec.get("label") is not None:
            preds[rec["bibtex_key"]] = rec
    return preds


def discover_runs(root: Path, split_keys: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Find prediction dumps and assign each to the split its keys belong to."""
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in PREDICTION_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            preds = load_predictions(path)
            if not preds:
                continue
            best_split, best_cov = None, 0.0
            for split, keys in split_keys.items():
                cov = len(preds.keys() & keys) / len(preds)
                if cov > best_cov:
                    best_split, best_cov = split, cov
            if best_split is None or best_cov < 0.95:
                continue
            # Checkpoint runs are named by directory; flat dumps by filename.
            stem = path.parent.name if path.parent.name.startswith("llm_") else path.stem
            for extra in ("_predictions", "_preds", "_per_entry"):
                stem = stem.removesuffix(extra)
            for suffix in SPLITS:
                stem = stem.removesuffix(f"_{suffix}")
            runs.append(
                {
                    "model": MODEL_NAMES.get(stem, stem),
                    "split": best_split,
                    "path": path,
                    "predictions": preds,
                }
            )
    return runs


# ---------------------------------------------------------------------- analysis


def score(entries: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> tuple[int, int, int]:
    """Return ``(detected, scored, uncertain)`` under the evaluator's protocol."""
    detected = uncertain = scored = 0
    for e in entries:
        pred = preds.get(e["bibtex_key"])
        label = pred["label"] if pred else "VALID"  # missing == conservative VALID
        if label == "UNCERTAIN":
            uncertain += 1
            continue
        scored += 1
        detected += label == "HALLUCINATED"
    return detected, scored, uncertain


def rate_block(entries: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    detected, scored, unc = score(entries, preds)
    lo, hi = wilson(detected, scored)
    return {
        "n": len(entries),
        "n_scored": scored,
        "n_uncertain": unc,
        "detected": detected,
        "detection_rate": detected / scored if scored else float("nan"),
        "ci95": [lo, hi],
    }


def analyze_run(
    hallucinated: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    preds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_count: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in hallucinated:
        by_count[len(defective_fields(e))].append(e)

    single = [e for e in hallucinated if len(defective_fields(e)) == 1]
    multi = [e for e in hallucinated if len(defective_fields(e)) >= 2]

    s_blk, m_blk = rate_block(single, preds), rate_block(multi, preds)
    diff = m_blk["detection_rate"] - s_blk["detection_rate"]
    _, fisher_p = stats.fisher_exact(
        [
            [m_blk["detected"], m_blk["n_scored"] - m_blk["detected"]],
            [s_blk["detected"], s_blk["n_scored"] - s_blk["detected"]],
        ]
    )

    def strata_for(*keys: str) -> dict[str, tuple[int, int, int, int]]:
        groups: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"single": [], "multi": []}
        )
        for e in hallucinated:
            arm = "multi" if len(defective_fields(e)) >= 2 else "single"
            groups[" x ".join(str(e.get(k)) for k in keys)][arm].append(e)
        out = {}
        for name, arms in sorted(groups.items()):
            k1, n1, _ = score(arms["multi"], preds)
            k0, n0, _ = score(arms["single"], preds)
            out[name] = (k1, n1, k0, n0)
        return out

    dose = []
    for count in sorted(by_count):
        blk = rate_block(by_count[count], preds)
        blk["defect_count"] = count
        dose.append(blk)

    return {
        "coverage": len([e for e in hallucinated if e["bibtex_key"] in preds]) / len(hallucinated),
        "overall_detection_rate": rate_block(hallucinated, preds)["detection_rate"],
        # On valid entries a "detection" is a false alarm, so the same helper gives FPR.
        "fpr": rate_block(valid, preds)["detection_rate"],
        "single": s_blk,
        "multi": m_blk,
        "diff_multi_minus_single": diff,
        "diff_ci95": list(
            newcombe_diff_ci(
                m_blk["detected"], m_blk["n_scored"], s_blk["detected"], s_blk["n_scored"]
            )
        ),
        "fisher_p": float(fisher_p),
        "mh_by_type": mantel_haenszel_rd(strata_for("hallucination_type")),
        "mh_by_generation_method": mantel_haenszel_rd(strata_for("generation_method")),
        "mh_by_type_and_method": mantel_haenszel_rd(
            strata_for("hallucination_type", "generation_method")
        ),
        "by_defect_count": dose,
        "trend_test": cochran_armitage(
            [(b["detected"], b["n_scored"], b["defect_count"]) for b in dose]
        ),
    }


# ------------------------------------------------------------------------ output


def fmt_pct(x: float) -> str:
    return "  n/a " if x != x else f"{100 * x:5.1f}%"


def fmt_p(p: float) -> str:
    if p != p:
        return "  n/a"
    return "<.001" if p < 0.001 else f"{p:.3f}"


#: A model that flags (nearly) everything gets a high detection rate on both arms
#: by construction, so its multi-vs-single contrast is uninformative. Mark these.
DEGENERATE_FPR = 0.50


#: UNCERTAIN predictions leave the denominator (evaluator protocol), so a model
#: that abstains heavily is compared on a small, self-selected subsample.
MIN_SCORED_SHARE = 0.90


def abstains_heavily(analysis: dict[str, Any]) -> bool:
    n = analysis["single"]["n"] + analysis["multi"]["n"]
    scored = analysis["single"]["n_scored"] + analysis["multi"]["n_scored"]
    return bool(n) and scored / n < MIN_SCORED_SHARE


def is_degenerate(analysis: dict[str, Any]) -> bool:
    fpr = float(analysis["fpr"])
    return fpr == fpr and fpr >= DEGENERATE_FPR  # first clause rejects NaN (no valid entries)


def print_split_report(
    split: str, runs: list[dict[str, Any]], gt: dict[str, Any], *, strata: bool
) -> None:
    print(f"\n{'=' * 104}")
    print(
        f"{split}  —  {gt['n_hallucinated']} hallucinated "
        f"({gt['n_single']} single-defect, {gt['n_multi']} multi-defect), {gt['n_valid']} valid"
    )
    print("=" * 104)

    print(
        f"\n{'model':<22} {'FPR':>7} {'DR all':>8} {'DR 1-def':>9} {'DR >=2-def':>11} "
        f"{'diff':>8} {'95% CI':>18} {'Fisher p':>9}"
    )
    print("-" * 104)
    for r in runs:
        a = r["analysis"]
        lo, hi = a["diff_ci95"]
        flags = ("*" if is_degenerate(a) else "") + ("†" if abstains_heavily(a) else "")
        name = f"{r['model']} {flags}".rstrip()
        print(
            f"{name:<22} {fmt_pct(a['fpr']):>7} {fmt_pct(a['overall_detection_rate']):>8} "
            f"{fmt_pct(a['single']['detection_rate']):>9} {fmt_pct(a['multi']['detection_rate']):>11} "
            f"{fmt_pct(a['diff_multi_minus_single']):>8} "
            f"[{fmt_pct(lo)},{fmt_pct(hi)}] {fmt_p(a['fisher_p']):>9} "
            f"{a['single']['n_scored'] + a['multi']['n_scored']:>4}/"
            f"{a['single']['n'] + a['multi']['n']:<3}"
        )
    if any(is_degenerate(r["analysis"]) for r in runs):
        print(
            f"  * FPR >= {DEGENERATE_FPR:.0%}: near-degenerate 'flag everything' behaviour — its detection"
        )
        print("    rates are high on both arms by construction, so read the contrast with care.")

    print(
        "\n  Confound-adjusted — Mantel-Haenszel pooled risk difference (multi - single) within strata."
    )
    print(
        "  Defect count is nearly a function of hallucination type, so single-arm strata drop out;"
    )
    print("  'k' is the number of strata that survive to carry information.")
    groups = [
        ("mh_by_type", "by type"),
        ("mh_by_generation_method", "by gen-method"),
        ("mh_by_type_and_method", "type x method"),
    ]
    head = f"  {'model':<22}"
    for _, title in groups:
        head += f"{title:>9} {'95% CI':>18} {'p':>6} {'k':>3}   "
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for r in runs:
        row = f"  {r['model']:<22}"
        for key, _ in groups:
            m = r["analysis"][key]
            row += (
                f"{fmt_pct(m['rd']):>9} [{fmt_pct(m['ci'][0])},{fmt_pct(m['ci'][1])}] "
                f"{fmt_p(m['p']):>6} {m['num_strata']:>3}   "
            )
        print(row)

    counts = sorted({b["defect_count"] for r in runs for b in r["analysis"]["by_defect_count"]})
    print("\n  Dose-response — detection rate by exact number of defective fields:")
    header = (
        "  "
        + f"{'model':<22}"
        + "".join(f"{f'{c} defect':>16}" for c in counts)
        + f"{'trend p':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in runs:
        blocks = {b["defect_count"]: b for b in r["analysis"]["by_defect_count"]}
        cells = ""
        for c in counts:
            b = blocks.get(c)
            cell = f"{fmt_pct(b['detection_rate'])} (n={b['n']})" if b else "—"
            cells += f"{cell:>16}"
        print(f"  {r['model']:<22}{cells}{fmt_p(r['analysis']['trend_test']['p']):>10}")

    if strata:
        print_strata_detail(runs)

    print_summary(runs)


def print_strata_detail(runs: list[dict[str, Any]]) -> None:
    """Per-stratum detail, so the pooled MH numbers can be audited."""
    for key, title in (
        ("mh_by_type", "hallucination type"),
        ("mh_by_generation_method", "generation method"),
    ):
        print(f"\n  Per-stratum detail — {title} (only strata with both arms):")
        print(
            f"    {'model':<22} {'stratum':<26} {'DR single':>15} {'DR multi':>15} "
            f"{'RD':>8} {'weight':>7}"
        )
        print("    " + "-" * 97)
        for r in runs:
            for st in r["analysis"][key]["strata"]:
                single = f"{fmt_pct(st['dr_single'])} (n={st['n_single']})"
                multi = f"{fmt_pct(st['dr_multi'])} (n={st['n_multi']})"
                print(
                    f"    {r['model']:<22} {st['name']:<26} {single:>15} {multi:>15} "
                    f"{fmt_pct(st['rd']):>8} {st['mh_weight']:>7.1f}"
                )


def print_summary(runs: list[dict[str, Any]]) -> None:
    """Sign test across models: does the multi-defect arm win more often than chance?"""
    discriminating = [r for r in runs if not is_degenerate(r["analysis"])]
    for label, subset in (("all models", runs), ("discriminating models only", discriminating)):
        if not subset:
            continue
        diffs = [r["analysis"]["diff_multi_minus_single"] for r in subset]
        pos = sum(d > 0 for d in diffs)
        nonzero = sum(d != 0 for d in diffs)
        p = float(stats.binomtest(pos, nonzero, 0.5).pvalue) if nonzero else float("nan")
        print(
            f"\n  Summary ({label}, n={len(subset)}): multi-defect arm higher in {pos}/{len(subset)}; "
            f"median diff {fmt_pct(sorted(diffs)[len(diffs) // 2]).strip()}; sign test p={fmt_p(p)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="repo root for prediction globs"
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=SPLITS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON to stdout")
    parser.add_argument(
        "--strata", action="store_true", help="print per-stratum detail behind the MH pooling"
    )
    parser.add_argument("--out", type=Path, help="also write the JSON payload here")
    args = parser.parse_args()

    entries_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        path = SPLIT_PATHS.get(split, args.data_dir / f"{split}.jsonl")
        if not path.exists():
            print(f"[skip] {split}: no data file at {path}", file=sys.stderr)
            continue
        entries_by_split[split] = load_split(path.parent, path.stem)

    split_keys = {s: {e["bibtex_key"] for e in es} for s, es in entries_by_split.items()}
    runs = discover_runs(args.root, split_keys)
    if not runs:
        print("No per-entry prediction dumps found.", file=sys.stderr)
        raise SystemExit(1)

    payload: dict[str, Any] = {}
    for split, entries in entries_by_split.items():
        split_runs = [r for r in runs if r["split"] == split]
        if not split_runs:
            continue
        hallucinated = [e for e in entries if e["label"] == "HALLUCINATED"]
        valid = [e for e in entries if e["label"] == "VALID"]
        counts = Counter(len(defective_fields(e)) for e in hallucinated)
        gt = {
            "n_hallucinated": len(hallucinated),
            "n_valid": len(valid),
            "n_single": counts[1],
            "n_multi": sum(v for k, v in counts.items() if k >= 2),
        }
        for r in split_runs:
            r["analysis"] = analyze_run(hallucinated, valid, r["predictions"])
        split_runs.sort(key=lambda r: r["model"])
        payload[split] = {
            "ground_truth": gt,
            "models": {
                r["model"]: {"predictions_path": str(r["path"]), **r["analysis"]}
                for r in split_runs
            },
        }
        if not args.json:
            print_split_report(split, split_runs, gt, strata=args.strata)

    if args.json:
        print(json.dumps(payload, indent=2, default=float))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
