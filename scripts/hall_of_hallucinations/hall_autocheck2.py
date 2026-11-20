"""Second free check for Hall verified rows that failed hall_autocheck.py's Crossref-only test.

Only rows whose first-pass failure is "title not found / first author missing" (no DOI, or DOI outside
Crossref) are re-checked; DOI mismatches and unregistered DOIs stay in manual review.
Sources, in order, stop at first match:
  DataCite DOI lookup (if the cited DOI is registered outside Crossref) -> title sim >= 0.9
  OpenAlex title search (OPENALEX_API_KEY if set; ~1 search per row)  -> title sim >= 0.9 + first author
  arXiv title search                                                   -> title sim >= 0.9 + first author
(DBLP's API sits behind a bot challenge as of 2026-11-18, so it is not used.)

Usage: python3 -I hall_autocheck2.py <candidates.jsonl> <autocheck.jsonl> <out.jsonl> [--limit 1000]
Resumable: rows already in <out.jsonl> are skipped.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hall_autocheck import UA, get_json, norm, sim, surnames
from hall_verified import SIM, verdict

DELAY = 1.0


def get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(5 * (attempt + 1))
    return ""


def hit(cited_title: str, first: str, title: str, authors: list[str]) -> dict:
    s = sim(cited_title, title)
    fa = bool(first) and any(first in norm(a).split() for a in authors)
    return {"title": title, "sim": s, "first_author_hit": fa, "authors": authors[:6]}


def best(cands: list[dict]) -> dict | None:
    return max(
        cands, key=lambda c: (c["sim"] >= SIM and c["first_author_hit"], c["sim"]), default=None
    )


def openalex(t: str, first: str) -> tuple[str, dict | None]:
    key = os.environ.get("OPENALEX_API_KEY", "")
    url = (
        "https://api.openalex.org/works?per-page=5&select=title,authorships,doi&search="
        + urllib.parse.quote(t)
    )
    if key:
        url += "&api_key=" + urllib.parse.quote(key)
    code, js = get_json(url)
    if code != 200 or not js:
        return f"http_{code}", None
    return "ok", best(
        [
            hit(
                t,
                first,
                w.get("title") or "",
                [a["author"].get("display_name", "") for a in w.get("authorships", [])],
            )
            for w in js.get("results", [])
        ]
    )


def arxiv(t: str, first: str) -> tuple[str, dict | None]:
    q = re.sub(r"[^\w\s]", " ", t)
    x = get_text(
        "https://export.arxiv.org/api/query?max_results=5&search_query="
        + urllib.parse.quote(f'ti:"{q}"')
    )
    if not x:
        return "error", None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(x)
    except ET.ParseError:
        return "parse_error", None
    return "ok", best(
        [
            hit(
                t,
                first,
                " ".join((e.findtext("a:title", "", ns) or "").split()),
                [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            )
            for e in root.findall("a:entry", ns)
        ]
    )


def datacite(t: str, doi: str) -> tuple[str, dict | None]:
    code, js = get_json("https://api.datacite.org/dois/" + urllib.parse.quote(doi))
    if code != 200 or not js:
        return f"http_{code}", None
    a = js["data"]["attributes"]
    titles = [x.get("title", "") for x in a.get("titles", [])]
    title = max(titles, key=lambda x: sim(t, x), default="")
    return "ok", {
        "title": title,
        "sim": sim(t, title),
        "first_author_hit": True,
        "authors": [c.get("name", "") for c in a.get("creators", [])][:6],
    }


def check(r: dict, ac: dict) -> dict:
    t = r["cited_title"]
    sn = surnames(r.get("cited_authors_year", ""))
    first = sn[0].split()[-1] if sn else ""
    out: dict = {"row_id": r["row_id"], "tried": {}}
    steps = []
    if ac.get("doi_status") == "other_ra" and ac.get("cited_doi"):
        steps.append(("datacite", lambda: datacite(t, ac["cited_doi"])))
    steps += [
        ("openalex", lambda: openalex(t, first)),
        ("arxiv", lambda: arxiv(t, first)),
    ]
    for name, fn in steps:
        st, b = fn()
        time.sleep(DELAY if name != "arxiv" else 3.0)
        out["tried"][name] = {"status": st, "best": b}
        if b and b["sim"] >= SIM and b["first_author_hit"]:
            out.update(pass2=True, pass2_source=name)
            return out
    out["pass2"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("autocheck")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    cand = [json.loads(line) for line in Path(a.candidates).read_text().splitlines()][: a.limit]
    ac = {
        (o := json.loads(line))["row_id"]: o for line in Path(a.autocheck).read_text().splitlines()
    }
    todo = []
    for r in cand:
        o = ac.get(r["row_id"])
        if not o:
            continue
        ok, why = verdict(o)
        if not ok and why.startswith(("title_not_found", "title_found_first_author_missing")):
            todo.append(r)
    dst = Path(a.out)
    done = {json.loads(line)["row_id"] for line in dst.open()} if dst.exists() else set()
    print(f"{len(todo)} rows to re-check ({len(done)} already done)", flush=True)
    with dst.open("a") as fh:
        for i, r in enumerate(todo):
            if r["row_id"] in done:
                continue
            fh.write(json.dumps(check(r, ac[r["row_id"]]), ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
