#!/usr/bin/env python3
"""Generate evaluation figures for HALLMARK.  [evaluation]

Creates publication-quality figures:
1. Tier-wise detection rates (grouped bar chart)
2. Per-type detection heatmap
3. Cost-accuracy tradeoff
4. Overall comparison
5. Temporal robustness comparison
6. DR-FPR operating points (the paper's Fig. 4)

Requires the ``figures`` extra::

    uv run --extra figures python scripts/generate_figures.py \
        --results-dir results/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Colorblind-safe palette (IBM Design Library, extended)
COLORS = [
    "#648FFF",  # blue
    "#785EF0",  # purple
    "#DC267F",  # magenta
    "#FE6100",  # orange
    "#FFB000",  # gold
    "#44AA99",  # teal (Tol)
    "#882255",  # wine (Tol)
    "#DDCC77",  # sand (Tol)
    "#117733",  # green (Tol)
    "#999933",  # olive (Tol)
]

# Display names for tools in figures
DISPLAY_NAMES = {
    "doi_only": "DOI-only",
    "harc": "HaRC",
    "verify_citations": "verify-citations",
    "llm_openai": "GPT-5.1",
    "llm_openai_gpt-5.4": "GPT-5.4",
    "llm_anthropic": "Claude Sonnet 4.5",
    "llm_openrouter_claude_sonnet_4_6": "Sonnet 4.6",
    "llm_openrouter_claude_opus_4_7": "Opus 4.7",
    "llm_openrouter_deepseek_r1": "DeepSeek-R1",
    "llm_openrouter_deepseek_v3": "DeepSeek-V3.2",
    "llm_openrouter_qwen": "Qwen3-235B",
    "llm_openrouter_qwen_max": "Qwen3-VL-235B",
    "llm_openrouter_mistral": "Mistral Large",
    "llm_openrouter_gemini_flash": "Gemini 2.5 Flash",
    "llm_openrouter_gemini_pro": "Gemini 2.5 Pro",
    "llm_openrouter_llama_4_maverick": "Llama 4 Maverick",
    "bibtexupdater": "bibtex-updater",
    "ensemble": "Ensemble",
    "doi_presence_heuristic": "DOI-heuristic",
    "llm_agentic_openai": "GPT-5.1 + DBs",
    "llm_tool_augmented": "GPT-5.1 + BTU (always)",
    "llm_agentic_btu_openai": "GPT-5.1 + BTU (opt)",
    "llm_agentic_btu_sonnet_4_6": "Sonnet 4.6 + BTU (opt)",
    "cascade_gpt54": "GPT-5.4 + BTU (cascade)",
    "cascade_gpt51": "GPT-5.1 + BTU (cascade)",
    "cascade_sonnet_4_6": "Sonnet 4.6 + BTU (cascade)",
}

# Tools excluded from tier detection rate chart (partial coverage, metrics not meaningful)
_PARTIAL_COVERAGE_TOOLS = {"harc", "verify_citations"}

# Rule-based / non-LLM tools, for operating-point marker styling.
_RULE_BASED_TOOLS = {"doi_only", "bibtexupdater", "doi_presence_heuristic"}

# Tools co-designed with the benchmark's own checks — read as an upper bound.
_CODESIGNED_TOOLS = {"bibtexupdater", "doi_presence_heuristic", "llm_tool_augmented"}

# Excluded from the operating-point figure: the DB-diagnosis cascade family is a
# diagnostic variant sweep, not a candidate system, and all three of its result
# files share one internal ``tool_name`` (so they cannot be told apart anyway).
_OPPOINT_EXCLUDED = {"cascade_db_diagnosis"}

# Operating-point figure styling; this script is the generator of record.
_CLUSTER_STYLE = {
    "doi": ("#7F7F7F", "x"),
    "indep": ("#1F77B4", "o"),
    "agentic": ("#D62728", "s"),
    "codesigned": ("#2CA02C", "D"),
}
_CLUSTER_LABEL = {
    "doi": "DOI-only",
    "indep": "Zero-shot LLM (independent)",
    "agentic": "Agentic (LLM + tools)",
    "codesigned": "Co-designed (interpret as upper bound)",
}

# Tools shown in main results table (Table 3) — used to filter figures.
# Mirrors the full independent full-coverage cohort plus the rule-based bibtex-updater.
# HaRC and verify_citations stay here for table consistency but are filtered out of
# figures via _PARTIAL_COVERAGE_TOOLS.
_MAIN_TABLE_TOOLS = {
    "doi_only",
    "harc",
    "verify_citations",
    "llm_openai",
    "llm_openai_gpt-5.4",
    "llm_openrouter_claude_sonnet_4_6",
    "llm_openrouter_claude_opus_4_7",
    "llm_openrouter_deepseek_r1",
    "llm_openrouter_deepseek_v3",
    "llm_openrouter_qwen",
    "llm_openrouter_qwen_max",
    "llm_openrouter_mistral",
    "llm_openrouter_gemini_flash",
    "llm_openrouter_gemini_pro",
    "llm_openrouter_llama_4_maverick",
    "bibtexupdater",
}


def _display_name(tool_name: str) -> str:
    return DISPLAY_NAMES.get(tool_name, tool_name)


def load_results(results_dir: Path) -> list[dict]:
    """Load dev_public evaluation result JSONs (skips CI, test, no-prescreening variants).

    Scans the top-level results dir plus a few known subdirectories that hold
    later-arriving model evaluations (``gpt54/``, ``new_models/``), and the
    canonical artifact directory ``data/v1.2/baseline_results/``.

    Deduplication by ``tool_name``: ``data/v1.2/baseline_results/`` is listed
    *first* so it takes priority as the canonical source; remaining paths
    (results/ subdirs) are skipped for tools already seen.
    """
    results: list[dict] = []
    seen_tools: set[str] = set()

    candidate_paths: list[Path] = []
    # Canonical artifacts — highest priority
    baseline_dir = results_dir.parent / "data" / "v1.2" / "baseline_results"
    if baseline_dir.is_dir():
        candidate_paths.extend(sorted(baseline_dir.glob("*_dev_public.json")))
    # Legacy results directories
    candidate_paths.extend(sorted(results_dir.glob("*_dev_public.json")))
    for subdir in ("gpt54", "new_models"):
        sub = results_dir / subdir
        if sub.is_dir():
            candidate_paths.extend(sorted(sub.glob("*.json")))

    for path in candidate_paths:
        # Skip CI bootstrap, no-prescreening variants, and partial-evaluation smoke runs
        if "_ci." in path.name or "_no_prescreening" in path.name or "smoke" in path.name:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        # Only include standard evaluation results with proper split + tool_name
        tool_name = data.get("tool_name")
        if not tool_name or data.get("split_name") != "dev_public":
            continue
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        results.append(data)
    return results


def load_operating_point_extras(results_dir: Path) -> list[dict]:
    """Load points that ``load_results`` cannot see, for the operating-point figure.

    Three gaps:

    1. **Cascade runs** (``results/cascade_gpt5*/``) nest their metrics under
       ``conservative``/``aggressive`` keys with no top-level ``tool_name``, so
       ``load_results`` skips them. They are the agentic arm of the cohort and
       must be on the plot.
    2. **Superseded zero-shot runs.** ``data/v1.2/baseline_results/`` is
       canonical, but a later full-coverage re-run of the same model on the same
       split supersedes it. The OpenRouter Anthropic endpoint drifts (see
       ``results/relabel_delta/endpoint_drift_probe/drift_summary.json``: 90%
       label agreement for Opus over 26 days on identical inputs), so a stale
       committed aggregate can sit far from where the model actually operates.
       Rescore files are keyed ``<model>_<split>_v<ver>_rescore.json``.

    Returns records shaped like ``load_results`` output plus a ``kind`` field.
    """
    extras: list[dict] = []

    # ``llm_tool_augmented`` (BTU always invoked) has no ``*_dev_public.json``
    # aggregate — its dev per-entry dump is an unfetched git-lfs pointer — so its
    # post-relabel dev numbers are read from the relabel-delta regen instead.
    todo = results_dir / "relabel_delta" / "todo_offline.json"
    if todo.is_file():
        try:
            with open(todo) as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError):
            blob = {}
        block = (
            blob.get("per_source_dr", {})
            .get("tools_offline", {})
            .get("llm_tool_augmented", {})
            .get("overall_new")
        )
        if isinstance(block, dict) and block.get("detection_rate") is not None:
            extras.append(
                {
                    **block,
                    "tool_name": "llm_tool_augmented",
                    "split_name": "dev_public",
                    "kind": "codesigned",
                }
            )

    for name, path in (
        ("cascade_gpt54", results_dir / "cascade_gpt54" / "gpt54_dev_public.json"),
        ("cascade_gpt51", results_dir / "cascade_gpt51" / "gpt51_dev_public.json"),
    ):
        if not path.is_file():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        variants = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and v.get("detection_rate") is not None
        }
        if not variants:
            continue
        # Pick the variant with the fewest abstentions, ties broken on DR.
        # Under the strict (UNCERTAIN-as-miss) convention an abstention is a
        # pure loss, so this is the variant that represents the system best.
        # It is not variant-shopping: GPT-5.4 scores DR 0.990 in BOTH variants,
        # but conservative reaches it only by abstaining on 21 hallucinated
        # entries (0.955 strict) while aggressive resolves all 606 (0.990
        # strict) at an identical FPR. GPT-5.1's two variants are identical, so
        # the choice is a no-op there.
        key = min(
            variants,
            key=lambda k: (variants[k].get("num_uncertain") or 0, -variants[k]["detection_rate"]),
        )
        extras.append({**variants[key], "tool_name": name, "kind": "agentic", "variant": key})

    # Third cascade: bibtex-updater -> Sonnet 4.6. Its results ship under the
    # ``cascade_db_diagnosis`` family, whose three files share one internal
    # ``tool_name`` -- which is why the family is in ``_OPPOINT_EXCLUDED`` and
    # cannot be resolved by tool name. The *aggressive* file is the row the
    # paper reports (Tab. 1: DR .997, FPR .148, "the aggressive stance"), so it
    # is pulled in explicitly by path and relabelled. Without this the figure
    # silently drops one of the three cascades.
    sonnet_cascade = (
        results_dir.parent
        / "data"
        / "v1.2"
        / "baseline_results"
        / "cascade_db_diagnosis_aggressive_dev_public.json"
    )
    if sonnet_cascade.is_file():
        try:
            with open(sonnet_cascade) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and data.get("detection_rate") is not None:
            extras.append({**data, "tool_name": "cascade_sonnet_4_6", "kind": "agentic"})

    rescore_dir = results_dir / "reviewer_experiments"
    if rescore_dir.is_dir():
        for path in sorted(rescore_dir.glob("*_rescore.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or data.get("split_name") != "dev_public":
                continue
            if data.get("detection_rate") is None:
                continue
            extras.append({**data, "kind": "llm", "supersedes": data.get("tool_name")})

    return extras


def strict_dev_scores(results_dir: Path) -> dict[str, tuple[float, float]]:
    """Recompute dev_public DR/FPR with UNCERTAIN counted as a **miss**.

    The published aggregates drop UNCERTAIN from the denominator, which flatters
    any tool that abstains — most visibly the GPT-5.4 cascade, whose DR falls
    from 0.990 to 0.956 once its 21 abstentions are scored as misses. This
    recomputes from per-entry dumps wherever one exists; callers keep the
    published number for tools that have none.

    ``cascade_gpt54`` is absent: its per-entry dump is the conservative run
    while the figure plots the aggressive variant (0 abstentions), whose
    published rate is already strict. ``cascade_gpt51`` stays because its two
    variants are identical.

    Returns ``{tool_name: (detection_rate, false_positive_rate)}``.
    """
    gold_path = results_dir.parent / "data" / "v1.2" / "dev_public.jsonl"
    if not gold_path.is_file():
        return {}
    gold = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
    gold = [e for e in gold if not e["bibtex_key"].startswith("__canary__")]
    hallucinated = [e for e in gold if e["label"] == "HALLUCINATED"]
    valid = [e for e in gold if e["label"] == "VALID"]
    gold_keys = {e["bibtex_key"] for e in gold}

    # tool_name -> per-entry dump. Only full-coverage dev dumps qualify.
    sources = {
        "cascade_gpt51": "cascade_gpt51/gpt51_dev_public_preds.jsonl",
        "bibtexupdater": "relabel_delta/btu_v1_2_0/bibtexupdater_dev_public_per_entry.jsonl",
        "llm_openrouter_claude_opus_4_7": (
            "checkpoints/llm_openrouter_claude_opus_4_7_dev_public/"
            "openrouter_anthropic_claude-opus-4.7.jsonl"
        ),
        "llm_openrouter_claude_sonnet_4_6": (
            "checkpoints/llm_openrouter_claude_sonnet_4_6_dev_public/"
            "openrouter_anthropic_claude-sonnet-4.6.jsonl"
        ),
        "llm_openrouter_claude_haiku_4_5": (
            "checkpoints/llm_openrouter_claude_haiku_4_5_dev_public/"
            "openrouter_anthropic_claude-haiku-4.5.jsonl"
        ),
        "llm_openrouter_gemini_pro": "new_models/gemini_pro.jsonl",
        "llm_openrouter_llama_4_maverick": "new_models/llama4_maverick.jsonl",
        "llm_openrouter_qwen_max": "new_models/qwen_max.jsonl",
    }

    scores: dict[str, tuple[float, float]] = {}
    for tool, rel in sources.items():
        path = results_dir / rel
        if not path.is_file():
            continue
        preds: dict[str, str] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            label = rec.get("label") or rec.get("pred_label")
            if rec.get("bibtex_key") and label:
                preds[rec["bibtex_key"]] = label
        if len(preds.keys() & gold_keys) / len(gold_keys) < 0.95:
            continue  # partial dump: not comparable to a full-split aggregate
        det = sum(1 for e in hallucinated if preds.get(e["bibtex_key"], "VALID") == "HALLUCINATED")
        fp = sum(1 for e in valid if preds.get(e["bibtex_key"], "VALID") == "HALLUCINATED")
        scores[tool] = (det / len(hallucinated), fp / len(valid))
    return scores


def fig_dr_fpr_operating_points(results: list[dict], results_dir: Path, output_dir: Path) -> None:
    """DR vs FPR scatter of every (tool, configuration) operating point.

    No Pareto front is drawn. Once the two-stage cascades are included the
    front degenerates to a single point, so the line carried no information;
    it was dropped when the paper's Fig. 4 became an operating-point plot.

    Cascade/agentic points are included, so the plot reflects the full
    candidate set rather than the zero-shot cohort alone. Agentic and
    co-designed points are drawn in their own series because they are built
    around the rule-based checks they wrap (a BTU cascade *contains*
    bibtex-updater), so they read as an upper bound, not a like-for-like
    competitor.

    This is the generator of record for the shipped figure
    (``hallmark-paper/figures/dr_fpr_operating_points.pdf``): sans-serif,
    untitled, cluster-coloured labels with leader lines.
    """
    extras = load_operating_point_extras(results_dir)
    superseded = {e["supersedes"] for e in extras if e.get("supersedes")}

    strict = strict_dev_scores(results_dir)
    approximated: list[str] = []

    clusters: dict[str, str] = {}
    points: list[tuple[float, float, str]] = []
    for r in results + extras:
        tool = r.get("tool_name")
        dr, fpr = r.get("detection_rate"), r.get("false_positive_rate")
        if not tool or dr is None or fpr is None:
            continue
        if tool in superseded and not r.get("kind"):
            continue  # stale committed run, replaced by a rescore
        if tool in _PARTIAL_COVERAGE_TOOLS or tool in _OPPOINT_EXCLUDED:
            continue
        if tool == "doi_only":
            cluster = "doi"
        elif tool in _CODESIGNED_TOOLS:
            cluster = "codesigned"
        elif r.get("kind") == "agentic" or tool.startswith("cascade") or "agentic" in tool:
            cluster = "agentic"
        else:
            cluster = "indep"
        if tool in strict:
            dr, fpr = strict[tool]
        elif r.get("num_uncertain"):
            # No per-entry dump, so UNCERTAIN cannot be reassigned; the published
            # (UNCERTAIN-dropped) rate is used and reported as approximate.
            approximated.append(f"{_display_name(tool)} (n_uncertain={r['num_uncertain']})")
        clusters[tool] = cluster
        points.append((float(fpr), float(dr), tool))

    if not points:
        logger.warning("No points for operating-point figure; skipping")
        return

    if approximated:
        logger.warning(
            "Strict (UNCERTAIN-as-miss) scoring unavailable for: %s "
            "— published UNCERTAIN-dropped rates used for these points.",
            "; ".join(sorted(approximated)),
        )

    # Local rcParams: the module default is serif, the paper figure is sans.
    with plt.rc_context({"font.family": "sans-serif", "font.size": 8}):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))

        for cluster, (color, marker) in _CLUSTER_STYLE.items():
            xs = [f for f, _, n in points if clusters[n] == cluster]
            ys = [d for _, d, n in points if clusters[n] == cluster]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                color=color,
                marker=marker,
                s=60,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
                zorder=3,
                label=_CLUSTER_LABEL[cluster],
            )

        # Greedy label placement measured against REAL text bounding boxes.
        # Character-count width estimates are too coarse in dense regions, so
        # each candidate slot is rendered, measured, and kept only if its box
        # clears every already-placed label and every marker.
        xr = (max(f for f, _, _ in points) - min(f for f, _, _ in points)) or 1.0
        yr = (max(d for _, d, _ in points) - min(d for _, d, _ in points)) or 1.0
        ymax_dr = max(d for _, d, _ in points)

        side_slots = [
            (0.014 * xr, 0.0, "left", "center"),
            (-0.014 * xr, 0.0, "right", "center"),
            (0.0, 0.045 * yr, "center", "bottom"),
            (0.0, -0.045 * yr, "center", "top"),
        ]
        diag_slots = [
            (
                sx * 0.030 * xr,
                sy * 0.050 * yr,
                "left" if sx > 0 else "right",
                "bottom" if sy > 0 else "top",
            )
            for sx in (1, -1)
            for sy in (1, -1)
        ]
        stack_slots = [
            (0.0, k * sgn * 0.055 * yr, "center", "bottom" if sgn > 0 else "top")
            for k in (1, 2, 3, 4)
            for sgn in (1, -1)
        ]

        # Limits must be final BEFORE any label is measured: every bbox below is
        # taken in display coords, so a later set_xlim would invalidate them.
        # Extra left/top room is where the outermost labels go.
        xs_all = [f for f, _, _ in points]
        ys_all = [d for _, d, _ in points]
        ax.set_xlim(min(xs_all) - 0.20 * xr, max(xs_all) + 0.16 * xr)
        ax.set_ylim(min(ys_all) - 0.10 * yr, max(ys_all) + 0.24 * yr)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_box = ax.get_window_extent(renderer)
        pad = 1.5  # display-unit breathing room around each label box

        def _overlaps(a: Bbox, boxes: list[Bbox]) -> float:
            """Total overlap area of box *a* against *boxes* (0.0 == clear)."""
            total = 0.0
            for b in boxes:
                dx = min(a.x1, b.x1) - max(a.x0, b.x0) + 2 * pad
                dy = min(a.y1, b.y1) - max(a.y0, b.y0) + 2 * pad
                if dx > 0 and dy > 0:
                    total += dx * dy
            return total

        # Markers occupy space too: labels must dodge points, not just labels.
        occupied = [
            Bbox.from_bounds(px - 4, py - 4, 8, 8)
            for px, py in (ax.transData.transform((f, d)) for f, d, _ in points)
        ]

        for fpr, dr, name in sorted(points, key=lambda t: (-t[1], t[0])):
            label = _display_name(name)
            color = _CLUSTER_STYLE[clusters[name]][0]
            candidates = (
                stack_slots + diag_slots + side_slots
                if dr >= ymax_dr - 0.06 * yr
                else side_slots + diag_slots + stack_slots
            )

            best = None
            for cand in candidates:
                dx, dy, ha, va = cand
                probe = ax.annotate(
                    label,
                    xy=(fpr, dr),
                    xytext=(fpr + dx, dr + dy),
                    fontsize=7,
                    ha=ha,
                    va=va,
                    alpha=0.0,
                )
                box = probe.get_window_extent(renderer)
                # Spilling outside the axes looks worse than a near-miss with
                # another label, so weight it heavily rather than forbidding it
                # outright (a forbidden slot could leave nowhere to go).
                spill = (
                    max(0.0, ax_box.x0 - box.x0)
                    + max(0.0, box.x1 - ax_box.x1)
                    + max(0.0, ax_box.y0 - box.y0)
                    + max(0.0, box.y1 - ax_box.y1)
                )
                area = _overlaps(box, occupied) + 500.0 * spill
                probe.remove()
                if area == 0.0:
                    best = (cand, 0.0)
                    break
                if best is None or area < best[1]:
                    best = (cand, area)

            assert best is not None
            (dx, dy, ha, va), _ = best
            far = (dx / xr) ** 2 + (dy / yr) ** 2 > 0.028**2
            ann = ax.annotate(
                label,
                xy=(fpr, dr),
                xytext=(fpr + dx, dr + dy),
                fontsize=7,
                color=color,
                ha=ha,
                va=va,
                zorder=4,
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": color,
                        "linewidth": 0.4,
                        "alpha": 0.55,
                        "shrinkA": 0.5,
                        "shrinkB": 2.0,
                    }
                    if far
                    else None
                ),
            )
            occupied.append(ann.get_window_extent(renderer))

        ax.set_xlabel(r"FPR $\downarrow$", fontsize=10)
        ax.set_ylabel(r"DR $\uparrow$", fontsize=10)
        ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(
            loc="lower right", fontsize=7.5, framealpha=0.95, borderpad=0.5, handletextpad=0.5
        )

        fig.tight_layout()
        path = output_dir / "dr_fpr_operating_points.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Wrote {path}")
    logger.info(
        "Operating points: "
        + ", ".join(f"{_display_name(n)} (FPR={f:.3f}, DR={d:.3f})" for f, d, n in sorted(points))
    )


def fig_tier_detection_rates(results: list[dict], output_dir: Path) -> None:
    """Horizontal grouped bar chart: detection rate per tier, one row per tool.

    Excludes partial-coverage tools (HaRC, verify-citations) whose per-tier
    metrics on a small subset are not meaningful.
    """
    # Filter to main-table full-coverage tools only
    _tier_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    filtered = [r for r in results if r["tool_name"] in _tier_tools]
    if not filtered:
        logger.warning("No full-coverage results for tier chart")
        return

    tiers = [1, 2, 3]
    tier_labels = ["Tier 1 (Easy)", "Tier 2 (Medium)", "Tier 3 (Hard)"]
    # Distinct colors for the three tiers (colorblind-safe)
    tier_colors = ["#648FFF", "#FE6100", "#DC267F"]

    # Sort tools by Tier-3 detection rate descending for visual readability
    def _t3(r: dict) -> float:
        return r.get("per_tier_metrics", {}).get("3", {}).get("detection_rate", 0.0)

    filtered = sorted(filtered, key=_t3, reverse=True)

    tools = [_display_name(r["tool_name"]) for r in filtered]
    n_tools = len(tools)

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.42 * n_tools + 1.2)))

    y = np.arange(n_tools)
    height = 0.78 / len(tiers)

    for i, (t, label, color) in enumerate(zip(tiers, tier_labels, tier_colors, strict=True)):
        rates = []
        for r in filtered:
            m = r.get("per_tier_metrics", {}).get(str(t), {})
            rates.append(m.get("detection_rate", 0.0))
        offset = (i - len(tiers) / 2 + 0.5) * height
        bars = ax.barh(
            y + offset,
            rates,
            height * 0.9,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, rate in zip(bars, rates, strict=True):
            if rate > 0:
                ax.text(
                    rate + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{rate:.0%}",
                    ha="left",
                    va="center",
                    fontsize=6.5,
                )

    ax.set_xlabel("Detection Rate")
    ax.set_title("Detection Rate by Hallucination Difficulty")
    ax.set_yticks(y)
    ax.set_yticklabels(tools)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()  # best Tier-3 model on top
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "tier_detection_rates.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_per_type_heatmap(results: list[dict], output_dir: Path) -> None:
    """Heatmap: detection rate per hallucination type per tool.

    Excludes partial-coverage tools (HaRC, verify-citations) whose per-type
    metrics on a small subset are not meaningful — consistent with the tier chart.
    """
    _heatmap_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    results = [r for r in results if r["tool_name"] in _heatmap_tools]
    if not results:
        return

    # Collect all types across results
    all_types = set()
    for r in results:
        all_types.update(r.get("per_type_metrics", {}).keys())
    all_types.discard("valid")
    types = sorted(all_types)

    if not types:
        logger.warning("No per-type metrics found")
        return

    tools = [_display_name(r["tool_name"]) for r in results]
    matrix = np.zeros((len(tools), len(types)))

    for i, result in enumerate(results):
        type_metrics = result.get("per_type_metrics", {})
        for j, t in enumerate(types):
            m = type_metrics.get(t, {})
            matrix[i, j] = m.get("detection_rate", 0.0)

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(tools) * 0.55 + 1.2)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", "\n") for t in types], rotation=0, fontsize=7)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools)

    # Add text annotations
    for i in range(len(tools)):
        for j in range(len(types)):
            val = matrix[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Detection Rate by Hallucination Type")
    fig.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)

    fig.tight_layout()
    path = output_dir / "per_type_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_cost_accuracy(results: list[dict], output_dir: Path) -> None:
    """Scatter plot: F1 vs throughput (entries/second)."""
    results = [r for r in results if r["tool_name"] in _MAIN_TABLE_TOOLS]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # Collect points for label adjustment
    points = []
    for i, result in enumerate(results):
        f1 = result.get("f1_hallucination", 0)
        cost = result.get("cost_efficiency", None)
        if cost is None or cost == 0:
            continue

        name = _display_name(result["tool_name"])
        ax.scatter(
            cost,
            f1,
            s=140,
            color=COLORS[i % len(COLORS)],
            edgecolors="black",
            linewidth=0.7,
            zorder=3,
        )
        points.append((cost, f1, name, i))

    # Greedy collision-free label placement: try candidate offsets in order,
    # accept the first whose approximate label box overlaps no placed box and no marker.
    renderer_pts = []  # placed label boxes in display points: (x0, y0, x1, y1)
    marker_r = 9.0 * (plt.gcf().dpi / 72.0) if False else 12.0  # marker radius in display px

    def to_disp(x, y):
        return ax.transData.transform((x, y))

    fig.canvas.draw()  # realize transforms
    candidates = [
        (12, 7, "left"),
        (12, -15, "left"),
        (-12, 7, "right"),
        (-12, -15, "right"),
        (14, 22, "left"),
        (14, -30, "left"),
        (-14, 22, "right"),
        (-14, -30, "right"),
        (16, 37, "left"),
        (16, -45, "left"),
        (-16, 37, "right"),
        (-16, -45, "right"),
        (18, 52, "left"),
        (-18, 52, "right"),
    ]

    # place the most crowded (cluster-inner) points first: sort by number of neighbors
    def n_neighbors(p):
        px, py = to_disp(p[0], p[1])
        c = 0
        for q in points:
            if q is p:
                continue
            qx, qy = to_disp(q[0], q[1])
            if abs(px - qx) < 70 and abs(py - qy) < 40:
                c += 1
        return c

    marker_boxes = []
    for cost, f1, _name, _idx in points:
        mx, my = to_disp(cost, f1)
        marker_boxes.append((mx - marker_r, my - marker_r, mx + marker_r, my + marker_r))

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    ppt = fig.dpi / 72.0  # display pixels per point (offsets/boxes are in points)
    for cost, f1, name, _idx in sorted(points, key=n_neighbors, reverse=True):
        px, py = to_disp(cost, f1)
        w = (4.6 * len(name) + 4) * ppt  # approx label width in display px
        h = 10.0 * ppt
        chosen = candidates[-1]
        for dx, dy, ha in candidates:
            dxp, dyp = dx * ppt, dy * ppt
            if ha == "left":
                box = (px + dxp, py + dyp - h / 2, px + dxp + w, py + dyp + h / 2)
            else:
                box = (px + dxp - w, py + dyp - h / 2, px + dxp, py + dyp + h / 2)
            if any(overlaps(box, b) for b in renderer_pts):
                continue
            if any(overlaps(box, m) for m in marker_boxes):
                continue
            ax_box = ax.get_window_extent()
            if (
                box[0] < ax_box.x0 + 2
                or box[2] > ax_box.x1 - 2
                or box[1] < ax_box.y0 + 2
                or box[3] > ax_box.y1 - 2
            ):
                continue
            chosen = (dx, dy, ha)
            renderer_pts.append(box)
            break
        else:
            renderer_pts.append(box)
        dx, dy, ha = chosen
        far = abs(dx) > 13 or abs(dy) > 16
        ax.annotate(
            name,
            (cost, f1),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            ha=ha,
            va="center",
            zorder=4,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, shrinkB=4) if far else None,
        )

    ax.set_xlabel("Throughput (entries/sec)")
    ax.set_ylabel("F1 (Hallucination)")
    ax.set_title("Cost\u2013Accuracy Tradeoff")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "cost_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_overall_comparison(results: list[dict], output_dir: Path) -> None:
    """Horizontal grouped bar chart: primary metrics comparison across tools.

    Excludes partial-coverage tools (HaRC, verify-citations) whose metrics
    on a small subset are not meaningful for visual comparison.
    """
    _comparison_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    results = [r for r in results if r["tool_name"] in _comparison_tools]
    if not results:
        return

    tools = [_display_name(r["tool_name"]) for r in results]
    metrics = ["detection_rate", "f1_hallucination", "tier_weighted_f1"]
    metric_labels = ["Detection Rate", "F1", "Tier-weighted F1"]

    n_tools = len(tools)
    fig, ax = plt.subplots(figsize=(7, max(3.5, n_tools * 0.55 + 1.2)))

    y = np.arange(n_tools)
    height = 0.8 / len(metrics)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
        values = [r.get(metric, 0) for r in results]
        offset = (i - len(metrics) / 2 + 0.5) * height
        bars = ax.barh(
            y + offset,
            values,
            height * 0.88,
            label=label,
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values, strict=True):
            if val > 0:
                ax.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                )

    ax.set_xlabel("Score")
    ax.set_title("Baseline Comparison")
    ax.set_yticks(y)
    ax.set_yticklabels(tools)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # best tool at top

    fig.tight_layout()
    path = output_dir / "overall_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_temporal_robustness(results_dir: Path, output_dir: Path) -> None:
    """Two-panel figure: DR and FPR (baseline vs probe) across all models."""
    # Discover all temporal_probe_*.json files. They live under
    # ``results/archive/``: a probe report scores no benchmark split, so the
    # freshness gate over ``results/`` cannot judge it. The top level is still
    # read, for a probe written by an older run.
    probe_files = sorted(
        list(results_dir.glob("temporal_probe_*.json"))
        + list((results_dir / "archive").glob("temporal_probe_*.json"))
    )
    # Exclude the probe set JSONL
    probe_files = [p for p in probe_files if p.suffix == ".json"]
    if not probe_files:
        # Fallback: try legacy single-model file
        legacy = results_dir / "temporal_probe.json"
        if legacy.exists():
            probe_files = [legacy]
        else:
            logger.warning("No temporal_probe_*.json found, skipping temporal figure")
            return

    # Load all model results
    models: list[dict] = []
    for path in probe_files:
        with open(path) as f:
            data = json.load(f)
        probe = data.get("probe_metrics", {})
        baseline = data.get("full_baseline", {})
        # Derive display name
        display = data.get("display_name", None)
        if display is None:
            # Legacy format: extract from filename
            stem = path.stem.replace("temporal_probe_", "").replace("temporal_probe", "GPT-5.1")
            display = stem
        models.append(
            {
                "name": display,
                "dr_base": baseline.get("detection_rate", 0),
                "dr_probe": probe.get("detection_rate", 0),
                "fpr_base": baseline.get("false_positive_rate", 0),
                "fpr_probe": probe.get("false_positive_rate", 0),
                "ece_base": baseline.get("ece", 0),
                "ece_probe": probe.get("ece", 0),
            }
        )

    # Sort by baseline FPR (ascending)
    models.sort(key=lambda m: m["fpr_base"])

    n = len(models)
    names = [m["name"] for m in models]

    # Colorblind-safe viridis-derived palette
    cmap = plt.colormaps.get_cmap("viridis").resampled(max(n, 2))
    model_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, (ax_dr, ax_fpr) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    x = np.arange(n)
    width = 0.35

    # ── Left panel: Detection Rate ────────────────────────────────────
    ax_dr.bar(
        x - width / 2,
        [m["dr_base"] for m in models],
        width,
        label="Baseline (2021\u20132023)",
        color=[(*c[:3], 0.5) for c in model_colors],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_probe = ax_dr.bar(
        x + width / 2,
        [m["dr_probe"] for m in models],
        width,
        label="Probe (2024\u20132026)",
        color=model_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars_probe, [m["dr_probe"] for m in models], strict=True):
        if val > 0:
            ax_dr.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_dr.set_ylabel("Detection Rate")
    ax_dr.set_title("(a) Detection Rate")
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_dr.set_ylim(0, 1.15)
    ax_dr.legend(loc="lower right", fontsize=8)
    ax_dr.spines["top"].set_visible(False)
    ax_dr.spines["right"].set_visible(False)

    # ── Right panel: False Positive Rate ──────────────────────────────
    ax_fpr.bar(
        x - width / 2,
        [m["fpr_base"] for m in models],
        width,
        label="Baseline (2021\u20132023)",
        color=[(*c[:3], 0.5) for c in model_colors],
        edgecolor="white",
        linewidth=0.5,
    )
    ax_fpr.bar(
        x + width / 2,
        [m["fpr_probe"] for m in models],
        width,
        label="Probe (2024\u20132026)",
        color=model_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    # Annotate FPR multiplier
    for i, m in enumerate(models):
        base_fpr = m["fpr_base"]
        probe_fpr = m["fpr_probe"]
        if base_fpr > 0.01:
            mult = probe_fpr / base_fpr
            ax_fpr.text(
                i + width / 2,
                probe_fpr + 0.02,
                f"{mult:.1f}\u00d7",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="red" if mult > 2.0 else "black",
            )
        elif probe_fpr > 0:
            ax_fpr.text(
                i + width / 2,
                probe_fpr + 0.02,
                f"{probe_fpr:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax_fpr.set_ylabel("False Positive Rate")
    ax_fpr.set_title("(b) False Positive Rate")
    ax_fpr.set_xticks(x)
    ax_fpr.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_fpr.set_ylim(0, 1.15)
    ax_fpr.legend(loc="upper left", fontsize=8)
    ax_fpr.spines["top"].set_visible(False)
    ax_fpr.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "temporal_robustness.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of tool names to include (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Only include results from this split (e.g., dev_public)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        logger.error(f"No results found in {results_dir}")
        return

    if args.tools:
        tool_set = {t.strip() for t in args.tools.split(",")}
        results = [r for r in results if r.get("tool_name") in tool_set]
    if args.split:
        results = [r for r in results if r.get("split_name") == args.split]

    logger.info(f"Loaded {len(results)} evaluation results")

    fig_tier_detection_rates(results, output_dir)
    fig_per_type_heatmap(results, output_dir)
    fig_cost_accuracy(results, output_dir)
    fig_overall_comparison(results, output_dir)
    fig_temporal_robustness(results_dir, output_dir)
    fig_dr_fpr_operating_points(results, results_dir, output_dir)

    print(f"\nGenerated 6 figures in {output_dir}/")


if __name__ == "__main__":
    main()
