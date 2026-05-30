#!/usr/bin/env python3
"""Scan model outputs for verbatim emission of the HALLMARK canary.  [contamination]

The benchmark embeds a high-entropy canary GUID (see
``hallmark.dataset.schema.CANARY_GUID``) in its data files. If a model
reproduces that GUID — or the full canary sentence — in its output, the
benchmark text leaked into the model's training corpus. This scanner flags
such verbatim emission.

It accepts either a predictions JSONL file (every string field of each record
is scanned, e.g. ``reason`` rationales) or any plain-text / log file. It is the
emission-detection counterpart to ``is_canary_entry`` (which only *filters*
canary rows out of evaluation).

Usage::

    python scripts/scan_canary_emission.py predictions.jsonl
    python scripts/scan_canary_emission.py model_dump.txt --text
    cat output.txt | python scripts/scan_canary_emission.py -        # stdin
    python scripts/scan_canary_emission.py preds.jsonl --json

Exit status is ``0`` when the input is clean and ``2`` when any canary
emission is detected (so it can gate a CI/contamination check).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from hallmark.dataset.schema import (
    CANARY_GUID,
    CANARY_STRING,
    scan_record_for_canary,
    scan_text_for_canary,
)


@dataclass
class CanaryHit:
    """A single location where canary text was emitted verbatim."""

    location: str  # "line 12" / "record bibtex_key=foo" / "text"
    markers: list[str]

    def as_dict(self) -> dict:
        return {"location": self.location, "markers": self.markers}


@dataclass
class ScanResult:
    scanned_units: int = 0
    hits: list[CanaryHit] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.hits


def scan_predictions_file(path: Path) -> ScanResult:
    """Scan a JSONL predictions file; one record per non-blank line.

    Lines that fail to parse as JSON are scanned as raw text so a malformed
    file still gets checked rather than silently passing.
    """
    result = ScanResult()
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            result.scanned_units += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                markers = scan_text_for_canary(line)
                loc = f"line {lineno}"
            else:
                markers = scan_record_for_canary(record)
                key = record.get("bibtex_key") if isinstance(record, dict) else None
                loc = f"line {lineno}" + (f" (bibtex_key={key})" if key else "")
            if markers:
                result.hits.append(CanaryHit(location=loc, markers=markers))
    return result


def scan_text_file(path: Path | None) -> ScanResult:
    """Scan a plain-text file (or stdin when *path* is None) as one blob."""
    text = sys.stdin.read() if path is None else Path(path).read_text(errors="replace")
    result = ScanResult(scanned_units=1)
    markers = scan_text_for_canary(text)
    if markers:
        result.hits.append(CanaryHit(location="text", markers=markers))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "input",
        help="Predictions JSONL file, text file, or '-' for stdin (implies --text).",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Treat input as plain text rather than a JSONL predictions file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report.",
    )
    args = parser.parse_args(argv)

    if args.input == "-":
        result = scan_text_file(None)
    elif args.text:
        result = scan_text_file(Path(args.input))
    else:
        path = Path(args.input)
        if not path.exists():
            parser.error(f"input file not found: {path}")
        result = scan_predictions_file(path)

    if args.json:
        print(
            json.dumps(
                {
                    "clean": result.clean,
                    "scanned_units": result.scanned_units,
                    "canary_guid": CANARY_GUID,
                    "hits": [h.as_dict() for h in result.hits],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        if result.clean:
            print(f"CLEAN: no canary emission in {result.scanned_units} unit(s).")
        else:
            print(
                f"CONTAMINATION: canary emitted in {len(result.hits)} of "
                f"{result.scanned_units} unit(s)."
            )
            for hit in result.hits:
                shown = [CANARY_STRING if m == CANARY_STRING else m for m in hit.markers]
                print(f"  {hit.location}: {shown}")

    return 0 if result.clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
