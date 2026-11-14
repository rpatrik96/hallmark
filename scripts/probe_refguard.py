"""Probe RefGuard (github.com/mggrim/refguard) as a candidate cascade Stage 1.

Draws a stratified sample from a split, feeds the blind BibTeX to the RefGuard
CLI (Bun), and scores its verdicts with the Stage-1 mapping:
blocking verdicts -> HALLUCINATED, CONSISTENT -> VALID, everything else -> escalate.

    uv run python scripts/probe_refguard.py --refguard-dir /path/to/refguard \
        --split dev_public --n-valid 80 --per-type 5 --out-dir results/refguard_probe
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from hallmark.dataset.schema import load_entries

BLOCKING = {"INCONSISTENT", "FABRICATED_CANDIDATE", "RETRACTED"}

# arXiv DataCite DOIs carry no version; the scraper kept one (``...v1``), which
# 404s at doi.org. bibtex-updater and prescreening strip it internally.
_ARXIV_VERSIONED_DOI = re.compile(r"^(10\.48550/arxiv\.[0-9.]+)v\d+$", re.IGNORECASE)


def strip_arxiv_version(doi: str) -> str:
    return _ARXIV_VERSIONED_DOI.sub(r"\1", doi.strip())


def stage1(verdict: str) -> str:
    if verdict in BLOCKING:
        return "HALLUCINATED"
    if verdict == "CONSISTENT":
        return "VALID"
    return "ESCALATE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refguard-dir", required=True)
    ap.add_argument("--split", default="dev_public")
    ap.add_argument(
        "--data-dir",
        default="data",
        help="dir holding v1.2/<split>.jsonl (e.g. an external set such as VerusCite V1)",
    )
    ap.add_argument("--n-valid", type=int, default=80)
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/refguard_probe")
    ap.add_argument("--rescore-only", action="store_true")
    ap.add_argument("--all", action="store_true", help="use every entry in the split (no sampling)")
    ap.add_argument(
        "--strip-arxiv-version",
        action="store_true",
        help="drop the vN suffix from arXiv DOIs before handing entries to RefGuard",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = load_entries(f"{args.data_dir}/v1.2/{args.split}.jsonl")
    rng = random.Random(args.seed)

    if args.all:
        sample = list(entries)
    else:
        by_type: dict[str, list] = defaultdict(list)
        for e in entries:
            by_type[e.hallucination_type or "VALID"].append(e)
        sample = rng.sample(by_type.pop("VALID"), args.n_valid)
        for t in sorted(by_type):
            sample += rng.sample(by_type[t], min(args.per_type, len(by_type[t])))
    truth = {e.bibtex_key: e for e in sample}

    bib = out / "sample.bib"
    report = out / "refguard_report.json"
    if not args.rescore_only:
        blinds = []
        n_stripped = 0
        for e in sample:
            b = e.to_blind()
            if args.strip_arxiv_version and "doi" in b.fields:
                new = strip_arxiv_version(b.fields["doi"])
                n_stripped += new != b.fields["doi"]
                b.fields["doi"] = new
            blinds.append(b)
        if args.strip_arxiv_version:
            print(f"stripped arXiv DOI version on {n_stripped} entries")
        bib.write_text("\n\n".join(b.to_bibtex() for b in blinds) + "\n")
        env = dict(os.environ)
        proc = subprocess.run(
            ["bun", "src/cli.ts", "--bib", str(bib.resolve()), "--out", str(report.resolve())],
            cwd=args.refguard_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        (out / "refguard_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        print(f"refguard exit code {proc.returncode}")

    results = json.loads(report.read_text())["results"]
    seen = {r["key"] for r in results}
    missing = set(truth) - seen
    if missing:
        print(f"WARNING: {len(missing)} sampled entries missing from report: {sorted(missing)[:5]}")

    rows = []
    for r in results:
        e = truth.get(r["key"])
        if e is None:
            continue
        rows.append(
            {
                "key": r["key"],
                "label": e.label,
                "type": e.hallucination_type or "VALID",
                "verdict": r["verdict"],
                "stage1": stage1(r["verdict"]),
                "notes": r["notes"],
                "truth_title": r.get("truth_title"),
                "match_basis": r.get("match_basis"),
            }
        )
    (out / "scored.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

    print(f"\nscored {len(rows)} entries")
    print("\nverdict counts by label:")
    for lab in ("VALID", "HALLUCINATED"):
        c = Counter(x["verdict"] for x in rows if x["label"] == lab)
        print(f"  {lab}: {dict(c.most_common())}")

    print("\nper type: flagged / valid-called / escalated (n)")
    for t in ["VALID", *sorted({x["type"] for x in rows} - {"VALID"})]:
        sub = [x for x in rows if x["type"] == t]
        c = Counter(x["stage1"] for x in sub)
        print(
            f"  {t:24s} H={c['HALLUCINATED']:3d}  V={c['VALID']:3d}  "
            f"E={c['ESCALATE']:3d}  (n={len(sub)})"
        )

    valid = [x for x in rows if x["label"] == "VALID"]
    hall = [x for x in rows if x["label"] == "HALLUCINATED"]
    fp = sum(x["stage1"] == "HALLUCINATED" for x in valid)
    fn = sum(x["stage1"] == "VALID" for x in hall)
    esc = sum(x["stage1"] == "ESCALATE" for x in rows)
    print(f"\nStage-1 FPR (VALID flagged): {fp}/{len(valid)}")
    print(f"Stage-1 miss (HALL passed as VALID): {fn}/{len(hall)}")
    print(f"Escalation rate: {esc}/{len(rows)}")


if __name__ == "__main__":
    main()
