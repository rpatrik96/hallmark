"""A1 prompt-sensitivity: build the stratified n=150 sample (seed 42).

Stratified by (label, difficulty_tier) so the variant/model comparison sees a
fixed, representative slice of dev_public rather than a label-skewed draw. VALID
entries carry difficulty_tier=None; they form their own stratum. Deterministic:
seed 42, sorted strata, sorted keys within a stratum before sampling.

Output: results/ablations/a1_prompt_full/sample_150.jsonl (full entries) and
sample_150_keys.json (the ordered bibtex_key list, for provenance).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
DEV = ROOT / "data/v1.0/dev_public.jsonl"
OUTDIR = ROOT / "results/ablations/a1_prompt_full"
N = 150
SEED = 42


def main() -> None:
    rows = [json.loads(line) for line in DEV.read_text().splitlines() if line.strip()]
    # Stratum key: (label, tier). VALID -> tier bucket "valid"; HALLUCINATED -> its tier.
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] == "VALID":
            key = ("VALID", "valid")
        else:
            key = ("HALLUCINATED", str(r.get("difficulty_tier")))
        strata[key].append(r)

    # Proportional allocation (largest-remainder) so totals sum exactly to N.
    total = len(rows)
    ordered_strata = sorted(strata.items(), key=lambda kv: kv[0])
    raw_alloc = [(k, len(v) * N / total) for k, v in ordered_strata]
    floor_alloc = {k: int(x) for k, x in raw_alloc}
    remainder = N - sum(floor_alloc.values())
    # distribute leftover to largest fractional parts (stable, sorted)
    frac_sorted = sorted(raw_alloc, key=lambda kx: (-(kx[1] - int(kx[1])), kx[0]))
    for i in range(remainder):
        floor_alloc[frac_sorted[i][0]] += 1

    rng = random.Random(SEED)
    sample: list[dict] = []
    alloc_report = {}
    for key, entries in ordered_strata:
        want = floor_alloc[key]
        pool = sorted(entries, key=lambda r: r["bibtex_key"])
        take = rng.sample(pool, min(want, len(pool)))
        sample.extend(take)
        alloc_report[f"{key[0]}/{key[1]}"] = {"pool": len(pool), "sampled": len(take)}

    # Deterministic final order by bibtex_key.
    sample.sort(key=lambda r: r["bibtex_key"])
    assert len(sample) == N, f"expected {N}, got {len(sample)}"

    (OUTDIR / "sample_150.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sample) + "\n"
    )
    (OUTDIR / "sample_150_keys.json").write_text(
        json.dumps([r["bibtex_key"] for r in sample], indent=2)
    )
    n_hall = sum(1 for r in sample if r["label"] == "HALLUCINATED")
    n_valid = sum(1 for r in sample if r["label"] == "VALID")
    meta = {
        "n": N,
        "seed": SEED,
        "stratified_by": ["label", "difficulty_tier"],
        "n_hallucinated": n_hall,
        "n_valid": n_valid,
        "allocation": alloc_report,
        "source": "data/v1.0/dev_public.jsonl",
    }
    (OUTDIR / "sample_150_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
