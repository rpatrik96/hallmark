"""Pick which title-only autocheck passes bibtex-check sends to manual review.

Input: the bibtex-check 1.2.0 raw JSONL (no HALLMARK pre-screening) run on the VALID-side rows that passed
on title + first author alone (no Crossref DOI match). Output: row_ids to force into manual review.

Flagged: author/title/year mismatch, not_found, invalid_year, partial_match, given_name_substitution,
and venue_mismatch unless the cited venue equals or abbreviates the Crossref title-search venue.
Not flagged: verified, venue abbreviations, and statuses where bibtex-check gives no evidence
(skipped = books, unconfirmed = could not verify).

Usage: python3 -I hall_btu_flags.py <candidates.jsonl> <autocheck.jsonl> <btu_raw.jsonl> <out.jsonl>
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

STOP = {"of", "in", "the", "and", "for", "on", "a", "an", "de", "la"}
FLAG = {
    "author_mismatch",
    "title_mismatch",
    "year_mismatch",
    "invalid_year",
    "not_found",
    "partial_match",
    "given_name_substitution",
    "hallucinated",
    "doi_mismatch",
    "doi_not_found",
}


def words(s: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split() if w not in STOP]


def abbrev(a: str, b: str) -> bool:
    """True if every word of a is a prefix of a later word of b, in order."""
    A, B = words(a), words(b)
    if not A or len(A) > len(B):
        return False
    j = 0
    for w in A:
        while j < len(B) and not B[j].startswith(w):
            j += 1
        if j == len(B):
            return False
        j += 1
    return True


def main() -> None:
    cand_f, ac_f, btu_f, out_f = map(Path, sys.argv[1:5])
    rows = {
        hashlib.sha1(f"hall:{r['row_id']}".encode()).hexdigest()[:12]: r
        for r in map(json.loads, cand_f.read_text().splitlines())
    }
    ac = {o["row_id"]: o for o in map(json.loads, ac_f.read_text().splitlines())}
    stats: Counter = Counter()
    out = []
    for b in map(json.loads, btu_f.read_text().splitlines()):
        r = rows[b["key"]]
        st = b["status"]
        flag = st in FLAG
        why = st
        if st == "venue_mismatch":
            best = ac[r["row_id"]].get("search_best") or {}
            ok = (
                best.get("sim", 0) >= 0.9
                and best.get("venue")
                and abbrev(r["cited_venue"], best["venue"])
            )
            flag, why = (not ok), ("venue_mismatch (abbreviation)" if ok else "venue_mismatch")
        stats[f"{'flag' if flag else 'keep'}: {why}"] += 1
        if flag:
            out.append(
                {
                    "row_id": r["row_id"],
                    "btu_status": why,
                    "mismatched_fields": b.get("mismatched_fields", []),
                }
            )
    out_f.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(f"flagged {len(out)} of {sum(stats.values())}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
