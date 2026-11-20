"""Turn the Hall HALLUCINATED and VALID drafts into runnable HALLMARK data dirs.

Writes three data dirs, each with v1.2/test_public.jsonl (labels) and test_public_blind.jsonl
(what tools see), shuffled with seed 0:
  <out_dir>/all/           932 HALLUCINATED + 932 VALID
  <out_dir>/hallucinated/  HALLUCINATED only (detection rate; FPR undefined)
  <out_dir>/valid/         VALID only (FPR; detection rate undefined)
The CLI only accepts the fixed split names, so each subset is its own data dir with split `test_public`.
Evaluate with `hallmark evaluate --data-dir <out_dir>/<subset> --split test_public`.

Usage: uv run python scripts/hall_of_hallucinations/make_dataset.py <hallucinated.jsonl> <valid.jsonl> <out_dir>
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry


def blind(e: BenchmarkEntry) -> dict:
    """Same four fields as data/v1.2/*_blind.jsonl."""
    b = e.to_blind()
    return {
        "bibtex_key": b.bibtex_key,
        "bibtex_type": b.bibtex_type,
        "fields": b.fields,
        "raw_bibtex": b.raw_bibtex,
    }


def main() -> None:
    hall_f, valid_f, out = map(Path, sys.argv[1:4])
    rows = [json.loads(line) for f in (hall_f, valid_f) for line in f.read_text().splitlines()]
    keys = Counter(r["bibtex_key"] for r in rows)
    if dup := [k for k, v in keys.items() if v > 1]:
        sys.exit(f"duplicate bibtex_key: {dup[:5]}")
    random.Random(0).shuffle(rows)
    entries = [BenchmarkEntry.from_dict(r) for r in rows]
    for name, keep in (("all", None), ("hallucinated", "HALLUCINATED"), ("valid", "VALID")):
        sub = [e for e in entries if keep is None or e.label == keep]
        d = out / name / "v1.2"
        d.mkdir(parents=True, exist_ok=True)
        (d / "test_public.jsonl").write_text(
            "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in sub)
        )
        (d / "test_public_blind.jsonl").write_text(
            "".join(json.dumps(blind(e), ensure_ascii=False) + "\n" for e in sub)
        )
        print(f"{d}: {len(sub)} entries {dict(Counter(e.label for e in sub))}")


if __name__ == "__main__":
    main()
