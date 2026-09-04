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

from hallmark.evaluation.validate import CANARY_ENTRIES, _allowed_entry_counts


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
