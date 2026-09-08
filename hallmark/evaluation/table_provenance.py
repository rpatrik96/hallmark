"""Freshness of derived tables.  [evaluation]

``scripts/check_results_freshness.py`` guards result JSONs against the split
they scored. Nothing guarded the layer above: the CSV and TeX tables under
``tables/`` are arithmetic over those results, and a table outlives its inputs
silently. ``tables/base_rate_precision.csv`` carried a doi_only false-positive
rate of 0.2788 for as long as it took someone to notice, against the 0.0417 its
source run reports after a transient HTTP 202 stopped being scored as a
fabricated citation -- the released table said a tool hands the user roughly
seven times more false accusations than it does, and its precision column, which
is what a deployed user actually experiences, was derived from that number.

Two checks, mirroring the ones the results guard already makes:

**Provenance.** A generator records the sha256 of every result JSON it read, in
``tables/provenance.json``; the guard rehashes them. This is the analogue of
``split_sha256``, and it is the general check -- it works for a table of any
shape, including the per-tier and per-type ones whose cells correspond to no
top-level result field.

**Values.** A table with a ``tool`` column names its source runs directly, so
its cells can be compared against those runs with no provenance at all. This is
the analogue of the results guard's count check, and it is what makes the guard
bite on a repository where no generator has recorded provenance yet. A guard
that reports nothing until every producer is updated is the always-red guard in
its other costume, and the results guard has already been down that road once.

A table with neither recorded provenance nor a ``tool`` column is
**unverifiable**, reported and never fatal, on the same reasoning that keeps
pre-``split_sha256`` results out of the failure set.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from hallmark.evaluation.validate import compute_sha256

#: Provenance sidecar, one entry per generated table, written by the generators.
PROVENANCE_FILE = "provenance.json"

#: Table columns that name the run a row came from rather than a measured value.
_KEY_COLUMNS = ("tool", "split")


@dataclass
class TableReport:
    """Per-table freshness verdict."""

    table: str
    is_stale: bool = False
    #: No provenance entry and no ``tool`` column: nothing to check against.
    unverifiable: bool = False
    #: True when at least one cell was compared against a current result value.
    value_checked: bool = False
    #: True when the table has a ``tool`` column, whether or not it resolved.
    has_tool_column: bool = False
    #: True when recorded input hashes were rechecked.
    provenance_checked: bool = False
    reasons: list[str] = field(default_factory=list)


def read_provenance(tables_dir: Path) -> dict[str, dict]:
    """Read ``tables/provenance.json``; an absent or unreadable file is empty."""
    path = Path(tables_dir) / PROVENANCE_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def record_table(
    table_path: str | Path,
    inputs: list[str | Path] | tuple[str | Path, ...],
    *,
    generator: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Record which files a generated table was derived from, and their hashes.

    Call this after writing a table. Paths are stored relative to the repository
    root so the record survives a clone, and only hashes are stored -- no
    timestamp, so regenerating unchanged inputs leaves the file byte-identical
    and does not churn the diff.

    Args:
        table_path: The table that was just written.
        inputs: Files it was derived from (result JSONs, splits, raw runs).
        generator: Script that produced it, e.g. ``scripts/compute_x.py``.
        repo_root: Repository root; inferred from this file when omitted.

    Returns:
        Path to the provenance file that was written.
    """
    table_path = Path(table_path).resolve()
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    tables_dir = table_path.parent

    def _rel(p: Path) -> str:
        p = p.resolve()
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)

    record = {
        "generator": generator,
        "inputs": {_rel(Path(p)): compute_sha256(Path(p)) for p in sorted(map(Path, inputs))},
    }

    provenance = read_provenance(tables_dir)
    provenance[table_path.name] = record
    out = tables_dir / PROVENANCE_FILE
    out.write_text(json.dumps(dict(sorted(provenance.items())), indent=2) + "\n")
    return out


def _result_payload(results_dir: Path, tool: str) -> dict | None:
    """Load the result JSON a table row names, unwrapping a dual-mode payload."""
    path = results_dir / f"{tool}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    probe = payload.get("conservative", payload)
    return probe if isinstance(probe, dict) else None


def _check_values(table_path: Path, results_dir: Path, report: TableReport) -> None:
    """Compare a table's cells against the runs its ``tool`` column names.

    Only columns that are also top-level numeric fields of the result JSON are
    compared, at the table's own precision: a cell written as ``0.3873`` is
    checked against the current value rounded to four decimals, so re-deriving
    the table is the only way to make the check pass.

    A ``tool`` cell that names no result file is skipped rather than failed.
    ``baseline_cost_latency.csv`` writes display names ("GPT-5.1 (zero-shot)"),
    and a guard has no business inventing a naming convention to make them
    resolve: an unresolvable table simply goes unchecked here and needs recorded
    provenance instead.
    """
    try:
        with table_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as exc:
        report.is_stale = True
        report.reasons.append(f"unreadable ({exc})")
        return

    if not rows or "tool" not in rows[0]:
        return
    report.has_tool_column = True

    mismatches: list[str] = []
    for row in rows:
        tool = (row.get("tool") or "").strip()
        if not tool:
            continue
        payload = _result_payload(results_dir, tool)
        if payload is None:
            continue
        for column, cell in row.items():
            if column in _KEY_COLUMNS or cell is None:
                continue
            current = payload.get(column)
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                continue
            cell = cell.strip()
            decimals = len(cell.partition(".")[2])
            try:
                recorded = float(cell)
            except ValueError:
                continue
            report.value_checked = True
            if f"{float(current):.{decimals}f}" != f"{recorded:.{decimals}f}":
                entry = f"{tool}.{column}: table {cell} != current {current:.{decimals}f}"
                if entry not in mismatches:
                    mismatches.append(entry)

    if mismatches:
        report.is_stale = True
        report.reasons.extend(mismatches)


def _check_provenance(
    record: dict, root: Path, report: TableReport, hash_cache: dict[str, str]
) -> None:
    """Rehash the inputs a generator recorded for this table."""
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return
    for rel, recorded_hash in sorted(inputs.items()):
        path = root / rel
        if not path.exists():
            report.is_stale = True
            report.reasons.append(f"input no longer exists: {rel}")
            continue
        if rel not in hash_cache:
            hash_cache[rel] = compute_sha256(path)
        if hash_cache[rel] != recorded_hash:
            report.is_stale = True
            report.reasons.append(
                f"derived from {rel} at {str(recorded_hash)[:12]}… but it is now "
                f"{hash_cache[rel][:12]}…"
            )
    report.provenance_checked = True


def check_tables(
    tables_dir: str | Path,
    results_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> list[TableReport]:
    """Check every generated table under ``tables_dir`` against its inputs.

    Args:
        tables_dir: Directory of generated tables (``*.csv`` / ``*.tex``).
        results_dir: Directory of the result JSONs tables are derived from.
        repo_root: Root the recorded provenance paths are relative to.

    Returns:
        One :class:`TableReport` per table, in filename order.
    """
    tables_dir = Path(tables_dir)
    results_dir = Path(results_dir)
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]

    provenance = read_provenance(tables_dir)
    hash_cache: dict[str, str] = {}
    reports: list[TableReport] = []
    seen: set[str] = set()

    for table_path in sorted(tables_dir.iterdir()):
        if table_path.name == PROVENANCE_FILE or table_path.suffix not in {".csv", ".tex"}:
            continue

        seen.add(table_path.name)
        report = TableReport(table=table_path.name)
        record = provenance.get(table_path.name)
        if isinstance(record, dict):
            _check_provenance(record, root, report, hash_cache)
        if table_path.suffix == ".csv":
            _check_values(table_path, results_dir, report)

        if not report.provenance_checked and not report.value_checked:
            report.unverifiable = True
            report.reasons.append(
                "no recorded provenance and its tool column resolves to no result file"
                if report.has_tool_column
                else "no recorded provenance and no tool column — nothing to check it against"
            )
        reports.append(report)

    # A record for a table that is no longer there. Renaming an output leaves
    # one behind -- this file gained an orphan within minutes of shipping -- and
    # it is worth saying so, because the alternative reading is that a table was
    # deleted and nobody noticed. Reported, never fatal: the missing table is
    # not itself evidence of a wrong number anywhere.
    for name in sorted(set(provenance) - seen):
        reports.append(
            TableReport(
                table=name,
                unverifiable=True,
                reasons=[
                    "recorded in provenance.json but no such table exists — remove the entry, "
                    "or regenerate the table"
                ],
            )
        )

    return reports
