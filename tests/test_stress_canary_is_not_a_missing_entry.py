"""A split's contamination canary is not a dropped entry.

``stress_test`` holds 121 hallucinated entries plus a single VALID canary
planted to detect contamination. With one valid entry the false-positive rate is
not a measurement, so the cascade rows are scored on the 121 and the paper says
so. ``validate-results --metadata`` had no way to express that and failed all
three released cascade results -- which took the whole ``baselines.yml`` matrix
red the first time its conclusion was allowed to mean anything (run
33915448052).

The exemption is exactly one entry, not a tolerance: an off-by-one allowance
would also pass a run that silently dropped an entry, which is the failure this
check exists to catch.
"""

from __future__ import annotations

import json

from hallmark.dataset.schema import EvaluationResult
from hallmark.evaluation.validate import (
    CANARY_ENTRIES,
    _allowed_entry_counts,
    compute_sha256,
    validate_reference_results,
)


def test_stress_test_may_omit_its_canary():
    assert _allowed_entry_counts("stress_test", 122) == {121, 122}


def test_a_split_with_no_canary_gets_no_slack():
    assert _allowed_entry_counts("dev_public", 1119) == {1119}
    assert 1118 not in _allowed_entry_counts("dev_public", 1119)


def test_the_exemption_is_one_entry_not_a_tolerance():
    """120 is a dropped entry, not a canary, and must still fail."""
    assert 120 not in _allowed_entry_counts("stress_test", 122)


def test_unknown_total_permits_nothing():
    """A missing metadata total must not read as 'any count is fine'."""
    assert _allowed_entry_counts("stress_test", None) == set()


def test_only_stress_test_is_registered():
    """Pins the scope. A second entry here needs its own justification."""
    assert CANARY_ENTRIES == {"stress_test": 1}


# --- The omitted entry must be the canary ------------------------------------------


def _stress_result(tmp_path, *, num_entries: int, num_valid: int):
    result = EvaluationResult(
        tool_name="cascade_db_diagnosis",
        split_name="stress_test",
        num_entries=num_entries,
        num_hallucinated=num_entries - num_valid,
        num_valid=num_valid,
        detection_rate=0.9,
        false_positive_rate=None,
        f1_hallucination=0.9,
        tier_weighted_f1=0.9,
    )
    filename = "cascade_db_diagnosis_stress_test.json"
    path = tmp_path / filename
    path.write_text(result.to_json())
    manifest = {
        "version": "1.0",
        "files": {
            filename: {"sha256": compute_sha256(path), "baseline": "x", "split": "stress_test"}
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"splits": {"stress_test": {"total": 122}}}))
    return validate_reference_results(tmp_path, metadata_path=meta)


def test_omitting_the_canary_passes(tmp_path):
    """121 scored, none of them VALID: the one missing entry is the canary."""
    vr = _stress_result(tmp_path, num_entries=121, num_valid=0)
    assert vr.passed, vr.errors


def test_omitting_a_hallucinated_entry_instead_fails(tmp_path):
    """121 scored but the VALID canary is among them: a real entry was dropped.

    The count allowance alone cannot tell these apart, which made it a
    one-entry tolerance in practice.
    """
    vr = _stress_result(tmp_path, num_entries=121, num_valid=1)
    assert not vr.passed
    assert any("canary" in e for e in vr.errors), vr.errors


def test_the_full_split_with_its_canary_passes(tmp_path):
    vr = _stress_result(tmp_path, num_entries=122, num_valid=1)
    assert vr.passed, vr.errors
