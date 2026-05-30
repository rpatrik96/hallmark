"""Offline re-score harness for the dev/test_public mislabel relabel.

For each tool that has a full-coverage per-entry prediction file, we recompute
the aggregate via the repo's `evaluate()` against (a) the pre-relabel labels at
commit 7a52362 (to validate we reproduce the published numbers) and (b) the
current/new labels. No API calls: the tool prediction is fixed per entry; only
the ground-truth label changed.

Schema preservation: we load the OLD published aggregate JSON, recompute a fresh
EvaluationResult on the NEW labels, and overwrite ONLY the keys that already
exist in the published JSON with the recomputed values (per_tier_metrics /
per_type_metrics replaced wholesale). Keys absent from the published JSON
(e.g. tier3_f1 on bibtexupdater) are NOT added.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import evaluate

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OLD_REV = "7a52362"  # pre-relabel state the published aggregates were built against

DATA = {
    "dev_public": REPO / "data/v1.0/dev_public.jsonl",
    "test_public": REPO / "data/v1.0/test_public.jsonl",
    "stress_test": REPO / "data/v1.0/stress_test.jsonl",
}


def load_entries_from_text(text: str) -> list[BenchmarkEntry]:
    ents = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            ents.append(BenchmarkEntry.from_json(line))
    # drop canaries to mirror load_entries
    from hallmark.dataset.schema import is_canary_entry

    return [e for e in ents if not is_canary_entry(e)]


def load_old_entries(split: str) -> list[BenchmarkEntry]:
    rel = str(DATA[split].relative_to(REPO))
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{OLD_REV}:{rel}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return load_entries_from_text(out.stdout)


def load_new_entries(split: str) -> list[BenchmarkEntry]:
    return load_entries_from_text(DATA[split].read_text())


def load_pred_map(path: Path) -> dict[str, Prediction]:
    """Load a per-entry prediction JSONL into {bibtex_key: Prediction}.

    Handles resume duplicates (last write wins) and the bibtexupdater raw schema
    (key/status/confidence) by normalizing to a Prediction.
    """
    pm: dict[str, Prediction] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "label" in d and "bibtex_key" in d:
            pm[d["bibtex_key"]] = Prediction.from_dict(d)
        elif "status" in d and "key" in d:
            # bibtexupdater raw schema
            status = d["status"].lower()
            label = {
                "hallucinated": "HALLUCINATED",
                "verified": "VALID",
                "valid": "VALID",
                "uncertain": "UNCERTAIN",
            }.get(status, "UNCERTAIN")
            pm[d["key"]] = Prediction(
                bibtex_key=d["key"],
                label=label,
                confidence=float(d.get("confidence", 0.5)),
            )
    return pm


def predictions_aligned(
    entries: list[BenchmarkEntry], pm: dict[str, Prediction]
) -> list[Prediction]:
    """Build a predictions list aligned to entry order (missing -> conservative VALID).

    Missing predictions are represented by a VALID@0.5 prediction so the
    confusion matrix treats them as VALID (the repo's conservative default also
    treats absent predictions as VALID). We materialize them explicitly so the
    ECE/AUROC lockstep ordering matches the published run, which evaluated over
    the full entry list.
    """
    out = []
    for e in entries:
        p = pm.get(e.bibtex_key)
        if p is None:
            p = Prediction(bibtex_key=e.bibtex_key, label="VALID", confidence=0.5)
        out.append(p)
    return out


def metrics_row(r) -> dict:
    return {
        "DR": r.detection_rate,
        "FPR": r.false_positive_rate,
        "F1": r.f1_hallucination,
        "MCC": r.mcc,
        "TWF1": r.tier_weighted_f1,
        "ECE": r.ece,
        "nh": r.num_hallucinated,
        "nv": r.num_valid,
        "U": r.num_uncertain,
    }


def rescore(pred_path, split: str, tool_name: str, eval_mode: str = "conservative"):
    pred_path = Path(pred_path)
    pm = load_pred_map(pred_path)
    old_e = load_old_entries(split)
    new_e = load_new_entries(split)
    old_preds = predictions_aligned(old_e, pm)
    new_preds = predictions_aligned(new_e, pm)
    r_old = evaluate(old_e, old_preds, tool_name, split, eval_mode=eval_mode)
    r_new = evaluate(new_e, new_preds, tool_name, split, eval_mode=eval_mode)
    return r_old, r_new, pm
