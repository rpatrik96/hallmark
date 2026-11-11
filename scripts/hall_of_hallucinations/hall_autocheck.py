"""Free automated checks on Hall rows: does the cited DOI exist / point to the cited paper,
and does Crossref know a paper with the cited title?  No LLM, no paid API.

Usage: python3 -I hall_autocheck.py <hall_rows_flagged.jsonl> <out.jsonl>
Resumable: rows already in <out.jsonl> are skipped.
"""

import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "hallmark-benchmark-research"
DELAY = 0.3
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3) if a and b else 0.0


def get_json(url: str) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return e.code, None
        except Exception:
            if attempt < 2:
                time.sleep(5)
                continue
            return -1, None
    return -1, None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def doi_org_status(doi: str) -> int:
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(
        f"https://doi.org/{urllib.parse.quote(doi)}", method="HEAD", headers={"User-Agent": UA}
    )
    try:
        return opener.open(req, timeout=30).status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def surnames(authors_year: str) -> list[str]:
    s = re.sub(r"\(\d{4}[a-z]?\)\s*$", "", authors_year or "")
    out = []
    for a in s.split(";"):
        a = a.strip()
        if not a or a.lower().startswith("et al"):
            continue
        if "," in a:
            out.append(a.split(",")[0].strip())
        else:
            toks = [t for t in a.split() if len(t.strip(".")) > 2 or not t.isupper()]
            out.append(toks[-1] if toks and "." in a else a.split()[0])
    return [norm(x) for x in out if norm(x)]


def summarize(item: dict) -> dict:
    t = item.get("title") or [""]
    au = [norm(a.get("family", "")) for a in item.get("author", []) or []]
    yr = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
    return {
        "title": t[0] if t else "",
        "doi": item.get("DOI", ""),
        "venue": (item.get("container-title") or [""])[0],
        "year": yr,
        "surnames": au,
    }


def check(r: dict) -> dict:
    out: dict = {"row_id": r["row_id"]}
    m = DOI_RE.search(r.get("cited_ids", "") + " " + r.get("raw", ""))
    doi = m.group(0).rstrip(".,;)") if m else ""
    out["cited_doi"] = doi
    cited_sn = surnames(r.get("cited_authors_year", ""))
    if doi:
        code, js = get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        time.sleep(DELAY)
        if code == 200 and js:
            tgt = summarize(js["message"])
            out.update(
                doi_status="crossref",
                doi_target=tgt,
                doi_title_sim=sim(r["cited_title"], tgt["title"]),
            )
        else:
            hc = doi_org_status(doi)
            time.sleep(DELAY)
            out["doi_status"] = (
                "other_ra"
                if hc in (301, 302, 303)
                else ("unregistered" if hc == 404 else f"error_{code}_{hc}")
            )
    else:
        out["doi_status"] = "none"
    q = r["cited_title"] + (" " + cited_sn[0] if cited_sn else "")
    code, js = get_json(
        "https://api.crossref.org/works?rows=5&select=DOI,title,author,container-title,issued&query.bibliographic="
        + urllib.parse.quote(q)
    )
    time.sleep(DELAY)
    best = None
    if code == 200 and js:
        for it in js["message"].get("items", []):
            s = summarize(it)
            s["sim"] = sim(r["cited_title"], s["title"])
            s["first_author_hit"] = bool(cited_sn) and cited_sn[0] in s["surnames"]
            if best is None or s["sim"] > best["sim"]:
                best = s
    out["search_status"] = code
    out["search_best"] = best
    out["hint"] = hint(out)
    return out


def hint(o: dict) -> str:
    bs = (o.get("search_best") or {}).get("sim", 0.0)
    found = bs >= 0.9
    st = o["doi_status"]
    if st == "unregistered":
        return (
            "DOI not registered; title found in Crossref -> fabricated_doi?"
            if found
            else "DOI not registered; title not found -> fabricated_doi / plausible_fabrication?"
        )
    if st == "crossref":
        ts = o["doi_title_sim"]
        if ts >= 0.9:
            return (
                "DOI points to the cited title -> check authors/venue/year (possible VerusCite FP)"
            )
        if ts < 0.6:
            return (
                "DOI points to another paper; cited title exists -> merged_citation?"
                if found
                else "DOI points to another paper; title not found -> hybrid_fabrication?"
            )
        return "DOI target title partly similar -> check"
    if st == "other_ra":
        return "DOI registered outside Crossref (DataCite etc.) -> open it"
    if found:
        return "title found in Crossref -> check authors/venue/year"
    if bs < 0.7:
        return "title not found in Crossref -> plausible_fabrication? (Crossref misses arXiv/books)"
    return "partial title match in Crossref -> check"


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in src.open()]
    rows = [
        r
        for r in rows
        if not (r.get("in_veruscite_v1") or r.get("in_hallmark") or r.get("dup_within_hall"))
    ]
    done = {json.loads(line)["row_id"] for line in dst.open()} if dst.exists() else set()
    with dst.open("a") as fh:
        for i, r in enumerate(rows):
            if r["row_id"] in done:
                continue
            fh.write(json.dumps(check(r), ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
