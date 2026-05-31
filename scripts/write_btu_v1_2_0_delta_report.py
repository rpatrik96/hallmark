#!/usr/bin/env python3
"""Write results/relabel_delta/btu_v1_2_0_regen.md from the fresh 1.2.0 aggregates.

OLD = the committed (HEAD) aggregates, which are the 2026-05-04 published
0.10.0 numbers reconstructed onto the corrected v1.1.1 labels.
NEW = the freshly regenerated bibtex-updater 1.2.0 aggregates (this session).

The standalone btu row is a faithful re-run (DOI/metadata resolution is not
LLM-drift-prone). The cascade row's Stage-2 (Sonnet via OpenRouter) IS
drift-prone, so the cascade NEW numbers are a dated snapshot, not a reproduction.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "data/v1.0/baseline_results"
OUT_MD = REPO / "results/relabel_delta/btu_v1_2_0_regen.md"

FIELDS = [
    ("detection_rate", "DR"),
    ("false_positive_rate", "FPR"),
    ("f1_hallucination", "F1"),
    ("tier_weighted_f1", "TW-F1"),
    ("mcc", "MCC"),
    ("auroc", "AUROC"),
    ("tier3_f1", "T3-F1"),
    ("ece", "ECE"),
]


def load_old(rel: str) -> dict:
    out: dict = json.loads(subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO))
    return out


def load_new(rel: str) -> dict:
    out: dict = json.loads((REPO / rel).read_text())
    return out


def fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def delta(old: float | None, new: float | None) -> str:
    if old is None or new is None:
        return "—"
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def row_block(title: str, rel: str) -> str:
    old = load_old(rel)
    new = load_new(rel)
    lines = [f"### {title}", "", f"`{rel}`", ""]
    lines.append("| Metric | OLD (0.10.0, published-faithful) | NEW (1.2.0, fresh) | Δ |")
    lines.append("|---|---|---|---|")
    for key, label in FIELDS:
        lines.append(
            f"| {label} | {fmt(old.get(key))} | {fmt(new.get(key))} | {delta(old.get(key), new.get(key))} |"
        )
    # composition row
    lines.append(
        f"| n (H/V/U) | {old.get('num_hallucinated')}/{old.get('num_valid')}/"
        f"{old.get('num_uncertain', 0)} | {new.get('num_hallucinated')}/"
        f"{new.get('num_valid')}/{new.get('num_uncertain', 0)} | — |"
    )
    prov = new.get("_provenance", {})
    if prov:
        lines.append("")
        lines.append(
            f"_NEW provenance: btu {prov.get('btu_version')}, "
            f"snapshot {prov.get('snapshot_date')}, endpoint: {prov.get('endpoint_btu')}"
            + (
                f"; Stage-2 endpoint: {prov.get('endpoint_stage2')}"
                if prov.get("endpoint_stage2")
                else ""
            )
            + "._"
        )
    hist = new.get("_btu_status_histogram")
    if hist:
        lines.append("")
        lines.append("_btu 1.2.0 raw status histogram:_ `" + json.dumps(hist) + "`")
    lines.append("")
    return "\n".join(lines)


def per_type_block(title: str, rel: str) -> str:
    old = load_old(rel).get("per_type_metrics", {})
    new = load_new(rel).get("per_type_metrics", {})
    keys = sorted(set(old) | set(new))
    lines = [f"#### per-type DR/F1 — {title}", ""]
    lines.append("| type | DR old→new | F1 old→new | n |")
    lines.append("|---|---|---|---|")
    for k in keys:
        o = old.get(k, {})
        n = new.get(k, {})
        lines.append(
            f"| {k} | {fmt(o.get('detection_rate'))}→{fmt(n.get('detection_rate'))} | "
            f"{fmt(o.get('f1'))}→{fmt(n.get('f1'))} | {n.get('count', o.get('count', '?'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def stage2_composition_block() -> str:
    """Summarise the fresh cascade Stage-2 verdict composition per split.

    Reads the persisted per-entry cascade files and reports how the deferred
    bucket resolved (Stage-1-decided vs Stage-2 VALID/HALLUCINATED/UNCERTAIN),
    so the reader can see how much of the fresh snapshot was shaped by the
    OpenAlex / Semantic-Scholar HTTP 429 conditions on this run's IP.
    """
    delta_dir = REPO / "results/relabel_delta/btu_v1_2_0"
    lines = ["### Fresh cascade Stage-routing composition", ""]
    lines.append("| split | Stage-1 decided | Stage-2 VALID | Stage-2 HALL | Stage-2 UNCERTAIN |")
    lines.append("|---|---|---|---|---|")
    for split in ("dev_public", "test_public", "stress_test"):
        path = delta_dir / f"cascade_db_diagnosis_{split}_per_entry.jsonl"
        if not path.exists():
            lines.append(f"| {split} | (per-entry file not found) | | | |")
            continue
        s1 = 0
        s2 = {"VALID": 0, "HALLUCINATED": 0, "UNCERTAIN": 0}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stage = r.get("cascade_stage")
            lbl = r.get("pred_label")
            if stage == "stage2_diagnosis":
                s2[lbl] = s2.get(lbl, 0) + 1
            else:
                s1 += 1
        lines.append(
            f"| {split} | {s1} | {s2['VALID']} | {s2['HALLUCINATED']} | {s2['UNCERTAIN']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    today = datetime.date.today().isoformat()
    parts: list[str] = []
    parts.append("# bibtex-updater 0.10.0 → 1.2.0 regeneration delta")
    parts.append("")
    parts.append(f"_Generated {today}._")
    parts.append("")
    parts.append(
        "**Scope.** Fresh re-run of bibtex-updater 1.2.0 (CLI `bibtex-check`) on "
        "`dev_public` + `test_public` for the co-designed standalone row, and the "
        "DB-first cascade (Stage-1 btu 1.2.0 + Stage-2 Sonnet via OpenRouter on the "
        "could-not-verify bucket) on `dev_public` + `test_public` + `stress_test`. "
        "OLD = the committed aggregates (the 2026-05-04 published 0.10.0 numbers "
        "reconstructed onto the corrected v1.1.1 labels). NEW = this session's fresh "
        "1.2.0 run."
    )
    parts.append("")
    parts.append(
        "**Drift policy.** The standalone btu row is a faithful re-run — DOI/metadata "
        "resolution is not LLM-drift-prone. The cascade Stage-2 (Sonnet via OpenRouter) "
        "IS drift-prone, so the cascade NEW numbers are a **dated snapshot**, not a "
        "reproduction of the published cascade row. The GPT-5.1 + btu always-call row "
        "could NOT be refreshed (OpenAI quota exhausted) and is left at its published "
        "value — flagged below."
    )
    parts.append("")

    parts.append("## Standalone co-designed bibtex-updater")
    parts.append("")
    parts.append(
        row_block("dev_public", "data/v1.0/baseline_results/bibtexupdater_dev_public.json")
    )
    parts.append(
        row_block("test_public", "data/v1.0/baseline_results/bibtexupdater_test_public.json")
    )

    parts.append("## DB-first cascade (Stage-1 btu 1.2.0 + Stage-2 OpenRouter-Sonnet)")
    parts.append("")
    parts.append(
        row_block("dev_public", "data/v1.0/baseline_results/cascade_db_diagnosis_dev_public.json")
    )
    parts.append(
        row_block("test_public", "data/v1.0/baseline_results/cascade_db_diagnosis_test_public.json")
    )
    parts.append(
        row_block("stress_test", "data/v1.0/baseline_results/cascade_db_diagnosis_stress_test.json")
    )

    parts.append("## Per-type breakdown (standalone btu)")
    parts.append("")
    parts.append(
        per_type_block("dev_public", "data/v1.0/baseline_results/bibtexupdater_dev_public.json")
    )
    parts.append(
        per_type_block("test_public", "data/v1.0/baseline_results/bibtexupdater_test_public.json")
    )

    parts.append("## Fresh cascade Stage-2 composition")
    parts.append("")
    parts.append(stage2_composition_block())

    parts.append("## Notes")
    parts.append("")
    parts.append(
        "- **Endpoint degradation on this snapshot (IMPORTANT).** During the fresh "
        "cascade Stage-2 run, OpenAlex and the Semantic Scholar graph `/paper/search` "
        "endpoint both returned HTTP 429 (rate-limited) for this run's shared IP, while "
        "CrossRef and arXiv stayed available. The agentic diagnoser therefore burned its "
        "fixed 5-tool-call budget retrying the two down endpoints on a non-trivial share "
        "of the deferred bucket and returned UNCERTAIN (`Max tool calls exceeded`). This "
        "inflates the Stage-2 UNCERTAIN rate and depresses the fresh cascade's detection "
        "rate / F1 relative to a run with all four search endpoints healthy. The "
        "**standalone btu row is unaffected** — `bibtex-check`'s own lookups "
        "(CrossRef/OpenAlex/S2/arXiv, used 971/630/.../127 times respectively) completed "
        "while those endpoints were healthy. We did NOT alter the published Stage-2 "
        "baseline config (`MAX_TOOL_CALLS=5`, per-tool `max_retries=3`) to compensate; "
        "the degradation is a documented condition of this dated snapshot, not a "
        "reproduction of the published cascade. See the Stage-2 composition table above."
    )
    parts.append(
        "- **AUROC / T3-F1 (F7 limitation).** The fresh cascade aggregates carry a "
        "self-consistent AUROC and Tier-3 F1 computed by the same `evaluate()` pass "
        "that produced the rest of the row, closing the F7 AUROC gap **provided** the "
        "fresh cascade is internally consistent (same per-entry predictions feed every "
        "metric). Per-entry predictions are persisted under "
        "`results/relabel_delta/btu_v1_2_0/` for audit."
    )
    parts.append(
        "- **New 1.2.0 statuses.** The wrapper map was extended for the statuses "
        "`bibtex-check` 1.2.0 emits that 0.10.0 did not: `arxiv_id_mismatch`, "
        "`doi_mismatch`, `given_name_substitution` → HALLUCINATED; `unconfirmed` → "
        "abstention (conservative VALID standalone / route-to-Stage-2 in cascade); "
        "the `--strict` statuses (`title_near_miss`, `author_truncated`, "
        "`strict_warn_*`) mapped for completeness though `--strict` was not enabled."
    )
    parts.append(
        "- **S2 key scope.** The provided Semantic Scholar key is accepted by "
        "`bibtex-check`'s lookup endpoints (which is why the standalone btu row is "
        "clean) but is rejected (HTTP 403) by the agentic harness's graph "
        "`/paper/search` endpoint; the search tool falls back to keyless, which was "
        "additionally HTTP 429 on this run's IP (see the endpoint-degradation note)."
    )
    parts.append(
        "- **GPT-5.1 + btu always-call row: NOT refreshed** (OpenAI quota). Uses the "
        "existing persisted predictions / published value."
    )
    parts.append("")

    OUT_MD.write_text("\n".join(parts))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
