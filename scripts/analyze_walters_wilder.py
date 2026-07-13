"""Analyze a bibtexupdater run on the Walters & Wilder ChatGPT-citation supplement.

Produces HALLMARK metrics (overall, per-tier, per-type) plus dataset-specific
breakdowns (by GPT version and subject field) and a short error analysis, and
writes both a metrics JSON and a Markdown report.

Usage:
    python scripts/analyze_walters_wilder.py \
        --supplement data/v1.0/supplement_chatgpt_citations.jsonl \
        --results <bibtex-check .jsonl> \
        --xlsx appendix3_walters_wilder.xlsx \
        --out-json results/walters_wilder/bibtexupdater_metrics.json \
        --out-md docs/walters_wilder_bibtexupdater_analysis.md
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from hallmark.baselines.bibtexupdater import _parse_jsonl_output
from hallmark.dataset.schema import Prediction, load_entries
from hallmark.evaluation.metrics import evaluate


def load_meta(xlsx_path: Path) -> dict[str, dict[str, Any]]:
    """Map bibtex_key -> {gpt_version, subject_field} from Appendix 3."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Works cited in those papers"]
    rows = [r for r in ws.iter_rows(values_only=True) if r[4]][1:]
    meta: dict[str, dict[str, Any]] = {}
    field_names = {"H": "Humanities", "S": "Social sciences", "N": "Natural sciences"}
    for r in rows:
        if str(r[8]).strip() != "A":
            continue
        gpt = str(r[0]).strip()
        cnum = f"{float(r[3]):.2f}".replace(".", "")
        key = f"ww_{gpt.replace('.', '')}_t{r[2]}_c{cnum}"
        meta[key] = {
            "gpt_version": gpt,
            "subject_field": field_names.get(str(r[1]).strip(), str(r[1]).strip()),
        }
    return meta


def status_of(pred: Prediction | None) -> str:
    if pred is None:
        return "MISSING"
    reason = pred.reason or ""
    if reason.startswith("Status: "):
        return reason.split(";")[0].removeprefix("Status: ").strip()
    return "prescreening_override" if "Pre-screening" in reason else "?"


def confusion(
    entries: list[Any], preds: dict[str, Prediction]
) -> dict[str, tuple[int, int, int, int]]:
    """Return {'subset': (tp, fn, fp, tn)} for the overall set."""
    tp = fn = fp = tn = 0
    for e in entries:
        p = preds.get(e.bibtex_key)
        flagged = bool(p and p.label == "HALLUCINATED")
        if e.label == "HALLUCINATED":
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    return {"all": (tp, fn, fp, tn)}


def dr_fpr(
    entries: list[Any], preds: dict[str, Prediction]
) -> tuple[float, float, tuple[int, int, int, int]]:
    tp = fn = fp = tn = 0
    for e in entries:
        p = preds.get(e.bibtex_key)
        flagged = bool(p and p.label == "HALLUCINATED")
        if e.label == "HALLUCINATED":
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    dr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return dr, fpr, (tp, fn, fp, tn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    entries = load_entries(str(args.supplement))
    preds_list = _parse_jsonl_output(args.results, 1.0, len(entries))
    preds = {p.bibtex_key: p for p in preds_list}
    meta = load_meta(args.xlsx)

    result = evaluate(
        entries,
        [preds[e.bibtex_key] for e in entries if e.bibtex_key in preds],
        tool_name="bibtexupdater",
        split_name="walters_wilder_chatgpt_citations",
        compute_ci=True,
    )

    # Breakdowns
    by_gpt: dict[str, list[Any]] = collections.defaultdict(list)
    by_field: dict[str, list[Any]] = collections.defaultdict(list)
    for e in entries:
        m = meta.get(e.bibtex_key, {})
        by_gpt[m.get("gpt_version", "?")].append(e)
        by_field[m.get("subject_field", "?")].append(e)

    status_dist = collections.Counter(status_of(preds.get(e.bibtex_key)) for e in entries)

    metrics: dict[str, Any] = {
        "tool": "bibtexupdater",
        "split": "walters_wilder_chatgpt_citations",
        "n_entries": len(entries),
        "coverage": result.coverage,
        "overall": {
            "detection_rate": result.detection_rate,
            "false_positive_rate": result.false_positive_rate,
            "f1_hallucination": result.f1_hallucination,
            "tier_weighted_f1": result.tier_weighted_f1,
            "ece": result.ece,
            "num_uncertain": result.num_uncertain,
            "detection_rate_ci": result.detection_rate_ci,
            "fpr_ci": result.fpr_ci,
        },
        "per_tier": result.per_tier_metrics,
        "per_type": result.per_type_metrics,
        "by_gpt_version": {},
        "by_subject_field": {},
        "status_distribution": dict(status_dist.most_common()),
        "confusion_all": confusion(entries, preds)["all"],
    }
    for g, subset in sorted(by_gpt.items()):
        dr, fpr, cm = dr_fpr(subset, preds)
        metrics["by_gpt_version"][g] = {
            "n": len(subset),
            "detection_rate": dr,
            "fpr": fpr,
            "cm": cm,
        }
    for fld, subset in sorted(by_field.items()):
        dr, fpr, cm = dr_fpr(subset, preds)
        metrics["by_subject_field"][fld] = {
            "n": len(subset),
            "detection_rate": dr,
            "fpr": fpr,
            "cm": cm,
        }

    # Error analysis
    false_positives = [
        (e.bibtex_key, status_of(preds.get(e.bibtex_key)), e.fields.get("title", ""))
        for e in entries
        if e.label == "VALID" and (p := preds.get(e.bibtex_key)) and p.label == "HALLUCINATED"
    ]
    missed = [
        (
            e.bibtex_key,
            e.hallucination_type,
            status_of(preds.get(e.bibtex_key)),
            e.fields.get("title", ""),
        )
        for e in entries
        if e.label == "HALLUCINATED"
        and not ((p := preds.get(e.bibtex_key)) and p.label == "HALLUCINATED")
    ]
    metrics["false_positives"] = false_positives
    metrics["missed"] = missed

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2, default=str))

    _write_markdown(args.out_md, metrics, result)
    print(f"Wrote {args.out_json} and {args.out_md}")
    print(
        f"DR={result.detection_rate:.3f} FPR={result.false_positive_rate:.3f} "
        f"F1={result.f1_hallucination:.3f} coverage={result.coverage:.3f}"
    )


def _write_markdown(path: Path, m: dict[str, Any], result: Any) -> None:
    o = m["overall"]
    lines = [
        "# bibtexupdater on the Walters & Wilder ChatGPT-citation supplement",
        "",
        f"- Entries: **{m['n_entries']}** (supported type = journal articles); "
        f"coverage {m['coverage']:.1%}",
        f"- Detection rate: **{o['detection_rate']:.3f}**  ",
        f"- False-positive rate: **{o['false_positive_rate']:.3f}**  ",
        f"- F1 (hallucination): **{o['f1_hallucination']:.3f}**  ",
        f"- Tier-weighted F1: **{o['tier_weighted_f1']:.3f}**  ",
        f"- ECE: {o['ece']}",
        "",
        "## Per tier",
        "",
        "| Tier | n_hall | n_valid | detection_rate | FPR | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for tier, d in sorted(m["per_tier"].items()):
        lines.append(
            f"| {tier} | {d['num_hallucinated']} | {d['num_valid']} | "
            f"{d['detection_rate']:.3f} | {d['false_positive_rate']:.3f} | {d['f1']:.3f} |"
        )
    lines += [
        "",
        "## Per hallucination type",
        "",
        "| Type | count | detection_rate |",
        "|---|---|---|",
    ]
    for t, d in sorted(m["per_type"].items(), key=lambda kv: -kv[1]["count"]):
        lines.append(f"| {t} | {d['count']} | {d['detection_rate']:.3f} |")
    lines += [
        "",
        "## By GPT version",
        "",
        "| Version | n | detection_rate | FPR |",
        "|---|---|---|---|",
    ]
    for g, d in m["by_gpt_version"].items():
        lines.append(f"| GPT-{g} | {d['n']} | {d['detection_rate']:.3f} | {d['fpr']:.3f} |")
    lines += [
        "",
        "## By subject field",
        "",
        "| Field | n | detection_rate | FPR |",
        "|---|---|---|---|",
    ]
    for f, d in m["by_subject_field"].items():
        lines.append(f"| {f} | {d['n']} | {d['detection_rate']:.3f} | {d['fpr']:.3f} |")
    lines += ["", "## Tool status distribution", "", "| Status | count |", "|---|---|"]
    for s, c in m["status_distribution"].items():
        lines.append(f"| {s} | {c} |")
    lines += [
        "",
        f"## False positives on real articles ({len(m['false_positives'])})",
        "",
    ]
    for key, st, title in m["false_positives"][:40]:
        lines.append(f"- `{key}` [{st}] {title[:80]}")
    lines += ["", f"## Missed hallucinations ({len(m['missed'])})", ""]
    for key, htype, st, title in m["missed"][:40]:
        lines.append(f"- `{key}` [{htype}, tool={st}] {title[:80]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
