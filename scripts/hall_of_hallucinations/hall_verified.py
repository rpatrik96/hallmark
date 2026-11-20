"""Build the Hall of Hallucinations VALID set from VerusCite `verified` rows.

Pass rule on hall_autocheck.py output (no LLM, no paid API):
  - cited DOI in Crossref and its title matches the cited title (sim >= 0.9)          -> pass
  - no cited DOI, or DOI registered outside Crossref (DataCite etc.): Crossref title search
    sim >= 0.9 and the cited first author is among the hit's authors                    -> pass
  - cited DOI not registered anywhere                                                   -> manual review
  - first-pass "title not found" rows re-checked by hall_autocheck2.py
    (DataCite / OpenAlex / arXiv, title sim >= 0.9 + first author) and matched   -> pass
  - title-only passes (no Crossref DOI match) that bibtex-check 1.2.0 contradicts
    (hall_btu_flags.py)                                                                 -> manual review
  - anything else                                                                       -> manual review

  review  <candidates.jsonl> <autocheck.jsonl> <out_dir> --n N [--audit 50]
      Final set = first N candidates (seed-0 order from extract_verified.py).
      Writes the manual-review queue (all autocheck failures in the set) and a uniform random
      audit sample of the set (seed 1), each as JSONL for build_valid_review_html.py.
  build   <candidates.jsonl> <autocheck.jsonl> <out.jsonl> --n N [--decisions csv ...]
      Walks candidates in order. Human verdict y -> keep; n/u -> drop; no verdict -> keep only if
      autocheck passed. Refuses to run while failures in the first N lack a verdict. Stops at N;
      rejected rows are backfilled from later candidates (unreviewed failures there are skipped).

Credit: Hall of Hallucinations by Andrew Wheeler / VerusCite, https://veruscite.com/hallucinations (used with permission).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hall_to_hallmark import CREDIT, DOI_RE, bib_type, parse_authors

SIM = 0.9


def verdict(ac: dict) -> tuple[bool, str]:
    ok, why = _verdict(ac)
    if ok and ac.get("btu_flag"):
        return False, f"btu_flag ({ac['btu_flag']['btu_status']}; passed on {why})"
    return ok, why


def _verdict(ac: dict) -> tuple[bool, str]:
    st = ac.get("doi_status")
    best = ac.get("search_best") or {}
    if st == "crossref":
        ts = ac.get("doi_title_sim", 0.0)
        if ts >= SIM:
            return True, "doi_title_match"
        return False, f"doi_title_mismatch ({ts:.2f})"
    if st == "unregistered":
        return False, "doi_unregistered"
    if best.get("sim", 0.0) >= SIM and best.get("first_author_hit"):
        return True, f"title_search_match ({st})"
    sec = ac.get("second") or {}
    if sec.get("pass2"):
        return True, f"second_check_{sec['pass2_source']} ({st})"
    if best.get("sim", 0.0) >= SIM:
        return False, f"title_found_first_author_missing ({st})"
    return False, f"title_not_found ({st}, best {best.get('sim', 0.0):.2f})"


def load(
    cand_f: str, ac_f: str, ac2_f: str | None = None, btu_f: str | None = None
) -> tuple[list[dict], dict]:
    with open(cand_f) as fh:
        cand = [json.loads(line) for line in fh]
    with open(ac_f) as fh:
        ac = {(o := json.loads(line))["row_id"]: o for line in fh}
    if ac2_f:
        with open(ac2_f) as fh:
            for line in fh:
                o = json.loads(line)
                if o["row_id"] in ac:
                    ac[o["row_id"]]["second"] = o
    if btu_f:
        with open(btu_f) as fh:
            for line in fh:
                o = json.loads(line)
                if o["row_id"] in ac:
                    ac[o["row_id"]]["btu_flag"] = o
    return cand, ac


def review(a: argparse.Namespace) -> None:
    cand, ac = load(a.candidates, a.autocheck, a.autocheck2, a.btu_flags)
    sel = cand[: a.n]
    missing = [r["row_id"] for r in sel if r["row_id"] not in ac]
    if missing:
        sys.exit(f"{len(missing)} of the first {a.n} candidates have no autocheck result yet")
    out = Path(a.out_dir)
    stats: Counter = Counter()
    queue = []
    for r in sel:
        ok, why = verdict(ac[r["row_id"]])
        stats[("pass: " if ok else "review: ") + re.sub(r" \(.*", "", why)] += 1
        stats[f"doi_status {ac[r['row_id']]['doi_status']} {'pass' if ok else 'review'}"] += 1
        if not ok:
            queue.append({**r, "autocheck": ac[r["row_id"]], "autocheck_why": why})
    audit_ids = {r["row_id"] for r in random.Random(1).sample(sel, a.audit)}
    audit = []
    for r in sel:
        if r["row_id"] in audit_ids:
            ok, why = verdict(ac[r["row_id"]])
            audit.append(
                {**r, "autocheck": ac[r["row_id"]], "autocheck_why": why, "autocheck_pass": ok}
            )
    tag = f"n{a.n}"
    for name, rows in (
        (f"hall_valid_review_queue_{tag}.jsonl", queue),
        (f"hall_valid_audit{a.audit}_{tag}.jsonl", audit),
    ):
        (out / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(
        f"set {len(sel)} | autocheck pass {len(sel) - len(queue)} | manual review {len(queue)} | audit sample {len(audit)} "
        f"(of which in review queue {sum(not x['autocheck_pass'] for x in audit)})"
    )
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


def build(a: argparse.Namespace) -> None:
    cand, ac = load(a.candidates, a.autocheck, a.autocheck2, a.btu_flags)
    dec: dict[str, dict] = {}
    for f in a.decisions or []:
        with open(f, encoding="utf-8") as fh:
            for d in csv.DictReader(fh):
                if d.get("verdict"):
                    dec[d["row_id"]] = d
    pending = [
        r["row_id"]
        for r in cand[: a.n]
        if r["row_id"] in ac and not verdict(ac[r["row_id"]])[0] and r["row_id"] not in dec
    ]
    if pending and not a.allow_unreviewed:
        sys.exit(
            f"{len(pending)} autocheck failures in the first {a.n} candidates have no human verdict; "
            "review them first (skipping them would bias the set toward Crossref-indexed papers) "
            "or pass --allow-unreviewed for a dry run"
        )
    stats: Counter = Counter()
    entries = []
    for r in cand:
        if len(entries) >= a.n:
            break
        if r["row_id"] not in ac:
            stats["stopped_at_unchecked_candidate"] += 1
            break
        ok, why = verdict(ac[r["row_id"]])
        d = dec.get(r["row_id"])
        if d and d["verdict"] != "y":
            stats[f"dropped_human_{d['verdict']}"] += 1
            continue
        if not d and not ok:
            stats["skipped_unreviewed_autocheck_failure"] += 1
            continue
        stats[("y: " + (d.get("source") or "human")) if d else "autocheck_pass_only"] += 1
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
        st = ac[r["row_id"]]["doi_status"]
        doi_ok = (
            {"crossref": True, "other_ra": True, "unregistered": False}.get(st)
            if "doi" in fields
            else None
        )
        how = (d.get("source") or "human-verified") if d else f"autocheck pass ({why})"
        entries.append(
            {
                "bibtex_key": hashlib.sha1(f"hall:{r['row_id']}".encode()).hexdigest()[:12],
                "bibtex_type": bt,
                "fields": fields,
                "label": "VALID",
                "hallucination_type": None,
                "difficulty_tier": None,
                "explanation": (
                    f"Real-world citation marked verified by VerusCite in '{r['paper_title']}' "
                    f"({r['paper_venue']}, {r['paper_year']}); {how}. Report: {r['paper_share_url']}. Source: {CREDIT}."
                ),
                "generation_method": "real_world",
                "source_conference": None,
                "source": "veruscite_hall",
                "publication_date": f"{year}-01-01" if year else "",
                "added_to_benchmark": date.today().isoformat(),
                "subtests": {
                    "doi_resolves": doi_ok,
                    "title_exists": True,
                    "authors_match": None,
                    "fields_complete": None,
                    "cross_db_agreement": None,
                    "venue_correct": None,
                },
                "raw_bibtex": None,
                "schema_version": "1.0",
            }
        )
    if len(entries) < a.n:
        print(
            f"WARNING: only {len(entries)} of {a.n} entries; review more candidates or extract more"
        )
    Path(a.out).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    print(f"wrote {len(entries)} VALID entries to {a.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("review")
    r.add_argument("candidates")
    r.add_argument("autocheck")
    r.add_argument("out_dir")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--audit", type=int, default=50)
    r.add_argument(
        "--btu-flags", help="hall_btu_flags.py output: title-only passes forced to review"
    )
    r.add_argument("--autocheck2", help="hall_autocheck2.py output (OpenAlex/arXiv/DataCite)")
    b = sp.add_parser("build")
    b.add_argument("candidates")
    b.add_argument("autocheck")
    b.add_argument("out")
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--decisions", nargs="*")
    b.add_argument(
        "--btu-flags", help="hall_btu_flags.py output: title-only passes forced to review"
    )
    b.add_argument("--autocheck2")
    b.add_argument("--allow-unreviewed", action="store_true")
    a = ap.parse_args()
    review(a) if a.cmd == "review" else build(a)


if __name__ == "__main__":
    main()
