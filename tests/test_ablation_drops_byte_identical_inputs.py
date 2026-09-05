"""The taxonomy-fold ablation must not count one result twice.

``cascade_db_diagnosis_aggressive_*`` and ``cascade_db_diagnosis_evalmode_aggressive_*``
are byte-identical on all three splits, with ``tool_name`` ``cascade_db_diagnosis``
inside both. Scored as two tools they inflated the README's "21 of 21" and
"20 of 20" counts and every rank-move figure derived from them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import ablate_taxonomy_fold as fold  # noqa: E402


def _payload() -> dict:
    """A result whose published figures the script's reconstruction reproduces."""
    per_type = {
        "fabricated_doi": {"count": 10, "detection_rate": 0.8},
        "wrong_venue": {"count": 5, "detection_rate": 1.0},
        "valid": {"count": 10},
    }
    tp = 10 * 0.8 + 5 * 1.0
    fn = 15 - tp
    cm = fold.Confusion(tp=tp, fp=1.0, fn=fn, tn=9.0)
    return {
        "tool_name": "tool",
        "split_name": "test_public",
        "num_entries": 25,
        "num_hallucinated": 15,
        "num_valid": 10,
        "detection_rate": cm.detection_rate,
        "false_positive_rate": cm.false_positive_rate,
        "f1_hallucination": cm.f1,
        "mcc": cm.mcc,
        "per_type_metrics": per_type,
    }


def test_a_byte_identical_copy_is_dropped_and_named(tmp_path, capsys, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    text = json.dumps(_payload(), indent=2)
    (results / "a_tool_test_public.json").write_text(text)
    (results / "a_tool_z_copy_test_public.json").write_text(text)
    other = _payload()
    other["per_type_metrics"]["fabricated_doi"]["detection_rate"] = 0.5
    cm = fold.Confusion(tp=10 * 0.5 + 5.0, fp=1.0, fn=15 - (10 * 0.5 + 5.0), tn=9.0)
    other.update(
        detection_rate=cm.detection_rate,
        false_positive_rate=cm.false_positive_rate,
        f1_hallucination=cm.f1,
        mcc=cm.mcc,
    )
    (results / "b_tool_test_public.json").write_text(json.dumps(other))

    out = tmp_path / "fold.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ablate", "--split", "test_public", "--results-dir", str(results), "--output", str(out)],
    )
    assert fold.main() == 0

    rows = list(csv.DictReader(out.open()))
    assert [r["tool"] for r in rows] == ["a_tool_test_public", "b_tool_test_public"]
    printed = capsys.readouterr().out
    assert "a_tool_z_copy_test_public" in printed and "byte-identical" in printed
