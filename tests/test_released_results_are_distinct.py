"""No two released result files may be byte-identical.

``cascade_db_diagnosis_aggressive_*`` and ``cascade_db_diagnosis_evalmode_aggressive_*``
shipped as the same bytes under two names on all three splits, with
``tool_name`` ``cascade_db_diagnosis`` inside both. Every consumer that
enumerates the directory -- the fold ablation, the base-rate table, the
leaderboard -- counted one run as two tools until each was taught to notice.
The release itself should not carry the duplicate.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data/v1.2/baseline_results"


def test_no_two_released_results_are_byte_identical():
    by_digest: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "manifest.json":
            continue
        by_digest[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
    duplicates = {d: names for d, names in by_digest.items() if len(names) > 1}
    assert not duplicates, (
        "byte-identical released results (one run under several names): "
        + "; ".join(", ".join(names) for names in duplicates.values())
    )
