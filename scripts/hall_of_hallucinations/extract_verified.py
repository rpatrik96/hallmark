"""Extract VerusCite `verified` rows from the scraped Hall share reports, dedupe, draw candidates.

Valid-side counterpart of scrape_hall.py + dedupe_sample.py. Reads the local raw/ pages (no rescrape).
Dedup by normalized title against: VerusCite V1, every HALLMARK entry under <hallmark_data_dir>
(VALID and HALLUCINATED), the Hall hallucinated rows, and earlier verified rows (first occurrence kept).
Candidates: seed-0 shuffle of the kept rows, first <n_candidates> (cut to the final N after autocheck).

Usage: python3 -I extract_verified.py <raw_dir> <hall_index.html> <hall_rows.jsonl> <v1_ground_truth.csv>
           <hallmark_data_dir> <out_dir> [n_candidates]
"""

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_hall import parse_cards, parse_rows


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main() -> None:
    raw, index, hall_rows_f, v1_f, data_dir, out = map(Path, sys.argv[1:7])
    n_cand = int(sys.argv[7]) if len(sys.argv) > 7 else 1100

    rows: list[dict] = []
    for c in parse_cards(index.read_text(encoding="utf-8")):
        f = raw / f"{c['share_url'].rsplit('/', 1)[-1]}.html"
        if not f.exists():
            continue
        for r in parse_rows(f.read_text(encoding="utf-8", errors="ignore"), status="verified"):
            rows.append(
                {**{k if k.startswith("paper_") else f"paper_{k}": v for k, v in c.items()}, **r}
            )

    v1 = {norm(r["title"]) for r in csv.DictReader(v1_f.open(encoding="utf-8"))} - {""}
    hm: dict[str, str] = {}
    for f in sorted(data_dir.rglob("*.jsonl")):
        for line in f.open():
            try:
                t = norm((json.loads(line).get("fields") or {}).get("title", ""))
            except json.JSONDecodeError:
                continue
            if t:
                hm.setdefault(t, f.name)
    hall = [json.loads(line) for line in hall_rows_f.open()]
    hall_titles = {norm(r["cited_title"]) for r in hall} - {""}
    hall_ids = {r["row_id"] for r in hall}

    seen: set[str] = set()
    for r in rows:
        t = norm(r["cited_title"])
        r["no_title"] = not t
        r["in_veruscite_v1"] = t in v1
        r["in_hallmark"] = hm.get(t, "")
        r["in_hall_hallucinated"] = t in hall_titles
        r["dup_within_hall"] = t in seen
        seen.add(t)

    def kept(r: dict) -> bool:
        return not (
            r["no_title"]
            or r["in_veruscite_v1"]
            or r["in_hallmark"]
            or r["in_hall_hallucinated"]
            or r["dup_within_hall"]
        )

    keep = [r for r in rows if kept(r)]
    (out / "hall_verified_rows_flagged.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    rng = random.Random(0)
    cand = keep[:]
    rng.shuffle(cand)
    cand = cand[:n_cand]
    for i, r in enumerate(cand):
        r["cand_rank"] = i
    (out / "hall_verified_candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cand)
    )

    ids = Counter(r["row_id"] for r in rows)
    print(
        f"verified rows {len(rows)} | no title {sum(r['no_title'] for r in rows)} | "
        f"in V1 {sum(r['in_veruscite_v1'] for r in rows)} | in HALLMARK {sum(bool(r['in_hallmark']) for r in rows)} | "
        f"title in Hall hallucinated {sum(r['in_hall_hallucinated'] for r in rows)} | "
        f"dup within verified {sum(r['dup_within_hall'] for r in rows)} | kept {len(keep)} | candidates {len(cand)}"
    )
    print(
        f"edited rows {sum(r['edited'] == '1' for r in rows)} | duplicate row_ids {sum(v > 1 for v in ids.values())} | "
        f"row_id collisions with hallucinated {len(set(ids) & hall_ids)} | papers {len({r['paper_share_url'] for r in keep})}"
    )


if __name__ == "__main__":
    main()
