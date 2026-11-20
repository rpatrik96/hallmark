"""Convert VerusCite V1 ground_truth.csv to HALLMARK entries for external evaluation only.

Not added to HALLMARK data. Writes <out_dir>/v1.2/test_public.jsonl (+ _blind) so that
`hallmark evaluate --data-dir <out_dir> --split test_public` scores V1.

Label mapping: possible_hallucination -> HALLUCINATED, verified -> VALID.
minor_error / not_found / uncertain are excluded (reported separately).
V1 has no HALLMARK type; hallucinated entries get the placeholder type
plausible_fabrication, so only overall recall/FPR are meaningful.

Usage: python3 -I v1_to_hallmark.py <ground_truth.csv> <out_dir>
"""

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BIBTYPE = {
    "journal_article": "article",
    "law_review": "article",
    "conference_paper": "inproceedings",
    "book": "book",
    "book_chapter": "incollection",
    "working_paper": "techreport",
    "dissertation": "phdthesis",
    "web": "misc",
    "unknown": "misc",
}
KEEP = {"possible_hallucination": "HALLUCINATED", "verified": "VALID"}


def fix_author(a: str) -> str:
    """'Deng J' / 'Ayachi Amor Y' (Vancouver) -> 'Deng, J.' so BibTeX reads the surname correctly."""
    a = a.strip()
    toks = a.split()
    if "," not in a and len(toks) >= 2 and re.fullmatch(r"[A-Z]{1,3}", toks[-1]):
        return " ".join(toks[:-1]) + ", " + " ".join(c + "." for c in toks[-1])
    return a


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    stats: Counter = Counter()
    entries = []
    for i, r in enumerate(rows):
        label = KEEP.get(r["expected_status"])
        stats[r["expected_status"]] += 1
        if not label:
            continue
        bt = BIBTYPE.get(r["citation_type"], "misc")
        f = {
            "title": r["title"],
            "author": " and ".join(fix_author(a) for a in r["authors"].split("|") if a.strip()),
        }
        if bt == "inproceedings":
            venue_key = "booktitle"
        elif bt == "incollection":
            venue_key = "journal_unused"
        else:
            venue_key = "journal"
        for k, v in [
            ("year", r["year"]),
            (venue_key, r["journal"]),
            ("booktitle", r["book_title"] if bt == "incollection" else ""),
            ("volume", r["volume"]),
            ("number", r["issue"]),
            ("pages", r["pages"]),
            ("doi", r["doi"]),
            ("isbn", r["isbn"]),
            ("publisher", r["publisher"]),
            ("url", r["url"]),
        ]:
            if v and v.strip() and k != "journal_unused":
                f[k] = v.strip()
        f = {k: v for k, v in f.items() if v}
        key = hashlib.sha1(f"v1:{i}:{r['source_pdf']}:{r['id']}".encode()).hexdigest()[:12]
        e = {
            "bibtex_key": key,
            "bibtex_type": bt,
            "fields": f,
            "label": label,
            "hallucination_type": "plausible_fabrication" if label == "HALLUCINATED" else None,
            "difficulty_tier": 3 if label == "HALLUCINATED" else None,
            "explanation": f"VerusCite V1 ({r['expected_status']}), {r['source_pdf']} #{r['id']} (csv row {i + 2}). External evaluation only.",
            "generation_method": "real_world" if label == "HALLUCINATED" else "scraped",
            "source_conference": None,
            "source": "veruscite_v1",
            "publication_date": f"{r['year']}-01-01" if r["year"][:4].isdigit() else "",
            "added_to_benchmark": "",
            "subtests": {},
            "raw_bibtex": None,
            "schema_version": "1.0",
        }
        entries.append(e)
    d = out / "v1.2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_public.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    )
    blind = [
        {k: e[k] for k in ("bibtex_key", "bibtex_type", "fields", "raw_bibtex")} for e in entries
    ]
    (d / "test_public_blind.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in blind)
    )
    lab = Counter(e["label"] for e in entries)
    print(f"source {dict(stats)} -> wrote {len(entries)} ({dict(lab)}) to {d}")


if __name__ == "__main__":
    main()
