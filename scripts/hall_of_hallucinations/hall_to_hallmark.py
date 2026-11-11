"""Convert Hall of Hallucinations rows to HALLMARK BenchmarkEntry JSONL (draft, evaluation-only).

Type priority: human review decision (verdict y) > notes-rule suggestion. Rows a human marked n/u are dropped.
Rows with no type at all are dropped and counted.

Usage: python3 -I hall_to_hallmark.py <rows_flagged.jsonl> <out.jsonl> [--review decisions.csv] [--autocheck autocheck.jsonl]
Credit: Hall of Hallucinations by Andrew Wheeler / VerusCite, https://veruscite.com/hallucinations (used with permission).
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_html import suggest

TIER = {
    "fabricated_doi": 1,
    "nonexistent_venue": 1,
    "placeholder_authors": 1,
    "future_date": 1,
    "chimeric_title": 2,
    "wrong_venue": 2,
    "swapped_authors": 2,
    "preprint_as_published": 2,
    "hybrid_fabrication": 2,
    "merged_citation": 2,
    "partial_author_list": 2,
    "near_miss_title": 3,
    "plausible_fabrication": 3,
    "arxiv_version_mismatch": 3,
}
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
CREDIT = "Hall of Hallucinations (VerusCite, A. Wheeler), https://veruscite.com/hallucinations"


def parse_authors(s: str) -> tuple[str, str]:
    m = re.search(r"\((\d{4})[a-z]?\)\s*$", s or "")
    year = m.group(1) if m else ""
    s = s[: m.start()] if m else (s or "")
    out = []
    for a in (x.strip() for x in s.split(";")):
        if not a:
            continue
        if re.fullmatch(r"et al\.?", a, re.I):
            out.append("others")
            continue
        etal = re.search(r",?\s+et al\.?$", a, re.I)
        if etal:
            a = a[: etal.start()]
        toks = a.split()
        # "Muraki T" / "Reid MD" -> "Muraki, T."
        if "," not in a and len(toks) >= 2 and re.fullmatch(r"[A-Z]{1,3}", toks[-1]):
            a = " ".join(toks[:-1]) + ", " + ". ".join(toks[-1]) + "."
        out.append(a)
        if etal:
            out.append("others")
    return " and ".join(out), year


def bib_type(venue: str, ids: str) -> str:
    if ids.startswith("ISBN"):
        return "book"
    if re.search(r"proceed|conference|symposium|workshop|\bconf\b", venue, re.I):
        return "inproceedings"
    if re.fullmatch(r"\s*arxiv\s*", venue, re.I) or not venue:
        return "misc"
    return "article"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--review")
    ap.add_argument("--autocheck")
    a = ap.parse_args()

    with open(a.rows) as fh:
        rows = [json.loads(line) for line in fh]
    rows = [
        r
        for r in rows
        if not (r.get("in_veruscite_v1") or r.get("in_hallmark") or r.get("dup_within_hall"))
    ]
    review: dict = {}
    if a.review:
        with open(a.review, encoding="utf-8") as fh:
            review = {d["row_id"]: d for d in csv.DictReader(fh)}
    review = {k: d for k, d in review.items() if d.get("verdict")}
    auto: dict = {}
    if a.autocheck:
        with open(a.autocheck) as fh:
            auto = {(o := json.loads(line))["row_id"]: o for line in fh}

    stats: Counter = Counter()
    entries = []
    for r in rows:
        rv = review.get(r["row_id"])
        if rv and rv["verdict"] != "y":
            stats[f"dropped_review_{rv['verdict']}"] += 1
            continue
        if rv and rv.get("hallmark_type") and rv["hallmark_type"] != rv.get("suggested_type"):
            htype, tsrc = rv["hallmark_type"], "human-reviewed"
        else:
            htype, _ = suggest(r)
            tsrc = "auto-suggested from VerusCite notes"
            if htype not in TIER:  # same fallback as parse_gptzero_csv.classify_hallucination
                htype, tsrc = "plausible_fabrication", "default (no notes rule matched)"
        if htype not in TIER:
            stats["dropped_no_type"] += 1
            continue
        author, year = parse_authors(r["cited_authors_year"])
        venue = r["cited_venue"]
        bt = bib_type(venue, r["cited_ids"])
        fields = {"title": r["cited_title"], "author": author}
        if year:
            fields["year"] = year
        if venue and bt != "misc":
            fields[
                "booktitle"
                if bt == "inproceedings"
                else ("publisher" if bt == "book" else "journal")
            ] = venue
        elif venue:
            fields["howpublished"] = venue
        m = DOI_RE.search(r["cited_ids"])
        if m:
            fields["doi"] = m.group(0).rstrip(".,;)")
        ac = auto.get(r["row_id"], {})
        doi_ok = (
            {"crossref": True, "other_ra": True, "unregistered": False}.get(
                ac.get("doi_status", "")
            )
            if "doi" in fields
            else None
        )
        expl = (
            f"Real-world hallucination flagged by VerusCite in '{r['paper_title']}' ({r['paper_venue']}, {r['paper_year']}). "
            f"Type {tsrc}. VerusCite notes: {r['notes'][:600]} Report: {r['paper_share_url']}. Source: {CREDIT}."
        )
        entries.append(
            {
                "bibtex_key": hashlib.sha1(f"hall:{r['row_id']}".encode()).hexdigest()[:12],
                "bibtex_type": bt,
                "fields": fields,
                "label": "HALLUCINATED",
                "hallucination_type": htype,
                "difficulty_tier": TIER[htype],
                "explanation": expl,
                "generation_method": "real_world",
                "source_conference": None,
                "source": "veruscite_hall",
                "publication_date": f"{year}-01-01" if year else "",
                "added_to_benchmark": date.today().isoformat(),
                "subtests": {
                    "doi_resolves": doi_ok,
                    "title_exists": None,
                    "authors_match": None,
                    "fields_complete": None,
                    "cross_db_agreement": None,
                    "venue_correct": None,
                },
                "raw_bibtex": None,
                "schema_version": "1.0",
            }
        )
        stats[tsrc] += 1
        stats[f"type:{htype}"] += 1
    Path(a.out).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    print(f"wrote {len(entries)} entries")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
