"""Scrape Hall of Hallucinations (veruscite.com/hallucinations) hallucination rows.

Permission granted by the site owner (Andrew Wheeler); credit + link required.
Usage: python3 -I scrape_hall.py <hall.html> <out_dir>
"""

import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

UA = "hallmark-benchmark-research (permission granted by site owner)"
DELAY = 1.5


def clean(x: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()


def span(cls: str, row: str) -> str:
    m = re.search(
        rf'class="(?:[^"]*\s)?{re.escape(cls)}(?:\s[^"]*)?"[^>]*>(.*?)</(?:span|strong|small|a|div)>',
        row,
        re.S,
    )
    return clean(m.group(1)) if m else ""


def parse_cards(t: str) -> list[dict]:
    cards = []
    for m in re.finditer(r'<article class="hall-card"(.*?)</article>', t, re.S):
        a, body = m.group(1), m.group(0)
        attr = {
            k: html.unescape(v) for k, v in re.findall(r'data-(\w+)="([^"]*)"', a.split(">")[0])
        }
        share = re.search(r'href="(https://veruscite.com/share/[^"]+)"', body)
        orig = re.search(r'href="([^"]+)"[^>]*>Original paper', body)
        n_h = re.search(r"(\d+) hallucinated reference", body)
        n_c = re.search(r"of (\d+) citations", body)
        cards.append(
            {
                "paper_title": attr.get("title", ""),
                "paper_authors": attr.get("authors", ""),
                "paper_venue": attr.get("venue", ""),
                "paper_doi": attr.get("doi", ""),
                "paper_year": attr.get("year", ""),
                "added": attr.get("added", ""),
                "share_url": share.group(1) if share else "",
                "original_url": orig.group(1) if orig else "",
                "n_hallucinated": int(n_h.group(1)) if n_h else None,
                "n_citations": int(n_c.group(1)) if n_c else None,
            }
        )
    return cards


def parse_rows(t: str) -> list[dict]:
    rows = []
    for m in re.finditer(r'<tr class="citation-row"(.*?)</tr>', t, re.S):
        r = m.group(0)
        attr = dict(re.findall(r'data-([\w-]+)="([^"]*)"', r.split(">")[0]))
        if attr.get("status") != "hallucination":
            continue
        notes = re.search(r'class="notes-text[^"]*"[^>]*>(.*?)</div>', r, re.S)
        rows.append(
            {
                "row_id": attr.get("id"),
                "edited": attr.get("edited"),
                "document_order": attr.get("document-order"),
                "status": attr.get("status"),
                "cited_title": span("cite-title", r),
                "cited_authors_year": span("cite-authors-year", r),
                "cited_venue": span("cite-venue-line", r),
                "cited_ids": span("cite-id-line", r),
                "raw": span("cite-raw-text", r),
                "match_title": span("cite-matched-title", r),
                "match_authors": span("cite-matched-authors", r),
                "match_venue": span("cite-matched-venue", r),
                "match_doi": span("cite-matched-doi", r),
                "notes": clean(notes.group(1)) if notes else "",
            }
        )
    return rows


def main() -> None:
    hall, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cards = parse_cards(hall.read_text(encoding="utf-8"))
    print(
        f"{len(cards)} cards, {sum(c['n_hallucinated'] or 0 for c in cards)} hallucinated per cards"
    )
    all_rows, mism = [], []
    for i, c in enumerate(cards):
        sid = c["share_url"].rsplit("/", 1)[-1]
        f = out / "raw" / f"{sid}.html"
        if not f.exists():
            req = urllib.request.Request(c["share_url"], headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    f.write_bytes(resp.read())
            except Exception as e:
                print(f"[{i}] FAIL {c['share_url']}: {e}")
                continue
            time.sleep(DELAY)
        rows = parse_rows(f.read_text(encoding="utf-8", errors="ignore"))
        if len(rows) != c["n_hallucinated"]:
            mism.append((i, c["n_hallucinated"], len(rows)))
        for r in rows:
            all_rows.append(
                {
                    **{f"paper_{k}" if not k.startswith("paper_") else k: v for k, v in c.items()},
                    **r,
                }
            )
    (out / "hall_hallucinated_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows)
    )
    print(f"rows {len(all_rows)}; count mismatches {len(mism)}: {mism[:10]}")


if __name__ == "__main__":
    main()
