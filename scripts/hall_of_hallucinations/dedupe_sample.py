"""Flag Hall rows overlapping VerusCite V1 / HALLMARK real-world entries; draw a review sample.

Usage: python3 -I dedupe_sample.py <hall_rows.jsonl> <v1_ground_truth.csv> <hallmark_data_dir> <out_dir>
"""

import csv
import json
import random
import re
import sys
from pathlib import Path


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main() -> None:
    rows_f, v1_f, data_dir, out = map(Path, sys.argv[1:5])
    rows = [json.loads(line) for line in rows_f.open()]

    v1 = list(csv.DictReader(v1_f.open(encoding="utf-8")))
    v1_titles = {norm(r["title"]) for r in v1 if norm(r["title"])}

    hm_titles: dict[str, str] = {}
    for f in sorted(data_dir.rglob("*.jsonl")):
        for line in f.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("generation_method") == "real_world" or "gptzero" in f.name:
                t = norm((e.get("fields") or {}).get("title", ""))
                if t:
                    hm_titles.setdefault(t, f.name)

    seen: dict[str, int] = {}
    for r in rows:
        t = norm(r["cited_title"])
        r["in_veruscite_v1"] = bool(t) and t in v1_titles
        r["in_hallmark"] = hm_titles.get(t, "") if t else ""
        r["dup_within_hall"] = t in seen if t else False
        seen.setdefault(t, 1)

    keep = [
        r for r in rows if not (r["in_veruscite_v1"] or r["in_hallmark"] or r["dup_within_hall"])
    ]
    (out / "hall_hallucinated_rows_flagged.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )

    random.seed(0)
    sample = random.sample(keep, min(100, len(keep)))
    cols = [
        "row_id",
        "paper_title",
        "paper_doi",
        "original_url",
        "share_url",
        "raw",
        "cited_title",
        "cited_venue",
        "match_title",
        "match_venue",
        "match_doi",
        "notes",
    ]
    with (out / "hall_review_sample_100.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([*cols, "is_hallucination (y/n/unsure)", "hallmark_type", "reviewer_note"])
        for r in sample:
            w.writerow([r.get(c, "") for c in cols] + ["", "", ""])

    print(
        f"rows {len(rows)} | in V1 {sum(r['in_veruscite_v1'] for r in rows)} | "
        f"in HALLMARK {sum(bool(r['in_hallmark']) for r in rows)} | dup within Hall {sum(r['dup_within_hall'] for r in rows)} | "
        f"kept {len(keep)} | edited rows {sum(r['edited'] == '1' for r in rows)} | sample {len(sample)}"
    )


if __name__ == "__main__":
    main()
