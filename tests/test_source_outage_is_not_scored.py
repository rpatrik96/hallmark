"""A run the tool disowned must not become a scored result.

``bibtex-check`` exits 5 when failed lookups touched at least 10% of entries,
and prints alongside it: *"Treat this run as incomplete and discard its
could-not-verify verdicts."* The wrapper logged that at ERROR and then parsed
the output and returned predictions anyway.

It happened for real on 2026-09-04. dblp.org was unreachable — every request
timing out, homepage included — and the ablation's first arm produced
``bibtexupdater_dev_public.json`` at DR 0.8185, FPR 0.0312 and **coverage
1.0000**, from a run where 285 of 1,119 entries (25.5%) never got a complete set
of source lookups. Nothing downstream could tell those abstentions from real
ones, and the result recorded full coverage.

This is the same defect as an API failure written into a prediction file as a
verdict, and as a timed-out batch scoring zero: a tool reported a problem
honestly and the consumer discarded the report. It is the fourth instance found
in one day, which is why the fix records the condition rather than only refusing
— availability moves outcomes, so it belongs beside the numbers.
"""

from __future__ import annotations

import pytest

from hallmark.baselines.bibtexupdater import (
    EXIT_SOURCE_OUTAGE,
    parse_source_condition,
)

#: bibtex-check's actual output from the 2026-09-04 dblp outage.
REAL_OUTAGE_OUTPUT = """\
INFO: Loaded 1119 entries from dev_public
WARNING: 285 of 1119 entries (25.5%) had at least one source lookup that did not \
complete: dblp (275), openalex (26). Those entries report api_error, not not_found \
-- a source that never answered is not evidence that a reference is absent.
WARNING: Hosts that could not be reached (DNS / connection / TLS / timeout / 5xx): \
dblp.org (275)
ERROR: Source outage: 25.5% of entries could not be checked against a complete set \
of sources (threshold 10%). Treat this run as incomplete and discard its \
could-not-verify verdicts; exiting 5.
"""


class TestSourceConditionParsing:
    def test_reproduces_the_real_outage(self):
        cond = parse_source_condition(REAL_OUTAGE_OUTPUT)
        assert cond == {
            "entries_with_incomplete_lookups": 285,
            "entries_total": 1119,
            "incomplete_fraction": pytest.approx(0.255),
            "per_source_failures": {"dblp": 275, "openalex": 26},
        }

    def test_a_healthy_run_reports_no_condition(self):
        assert parse_source_condition("INFO: Loaded 1119 entries\nINFO: done") is None

    def test_the_final_summary_wins(self):
        """bibtex-check may report progressively; the last line is the total."""
        progressive = (
            "WARNING: 10 of 100 entries (10.0%) had at least one source lookup "
            "that did not complete: dblp (10)\n"
            "WARNING: 40 of 100 entries (40.0%) had at least one source lookup "
            "that did not complete: dblp (35), openalex (5)\n"
        )
        cond = parse_source_condition(progressive)
        assert cond is not None
        assert cond["entries_with_incomplete_lookups"] == 40
        assert cond["per_source_failures"] == {"dblp": 35, "openalex": 5}

    def test_malformed_counts_do_not_raise(self):
        """Provenance parsing must never break an evaluation."""
        cond = parse_source_condition(
            "WARNING: 5 of 50 entries (10.0%) had at least one source lookup "
            "that did not complete: dblp (not-a-number), openalex (3)"
        )
        assert cond is not None
        assert cond["per_source_failures"] == {"openalex": 3}


def test_the_exit_code_is_the_one_the_tool_documents():
    """Pinned because the whole guard keys on it."""
    assert EXIT_SOURCE_OUTAGE == 5
