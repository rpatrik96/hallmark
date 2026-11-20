"""Self-contained HTML page for reviewing Hall `verified` rows (VALID side).

Merges the autocheck-failure queue and the uniform audit sample into one page; each card is tagged
(QUEUE = autocheck failed, AUDIT = random sample of the final set). Verdicts: y = valid citation of a
real work, n = not valid, u = unsure. Decisions autosave in the browser and export as CSV.

Usage: python3 -I build_valid_review_html.py <queue.jsonl> <audit.jsonl> <out.html>
"""

import json
import sys
from pathlib import Path


def ev(r: dict) -> dict:
    ac = r["autocheck"]
    t = ac.get("doi_target") or {}
    b = ac.get("search_best") or {}
    return {
        "row_id": r["row_id"],
        "paper_title": r["paper_title"],
        "share_url": r["paper_share_url"],
        "original_url": r["paper_original_url"],
        "raw": r["raw"],
        "cited_title": r["cited_title"],
        "cited_authors": r["cited_authors_year"],
        "cited_venue": r["cited_venue"],
        "cited_ids": r["cited_ids"],
        "match_title": r["match_title"],
        "match_authors": r["match_authors"],
        "match_venue": r["match_venue"],
        "match_doi": r["match_doi"],
        "notes": r["notes"],
        "why": r["autocheck_why"],
        "doi": ac.get("cited_doi", ""),
        "doi_status": ac.get("doi_status", ""),
        "doi_target": f"{t.get('title', '')} | {t.get('venue', '')} | {t.get('year', '')} | {', '.join(t.get('surnames', [])[:6])}"
        if t
        else "",
        "search": f"[{b.get('sim', 0)}] {b.get('title', '')} | {b.get('venue', '')} | {b.get('year', '')} | {', '.join(b.get('surnames', [])[:6])} | doi {b.get('doi', '')}"
        if b
        else "",
    }


def main() -> None:
    q_f, a_f, out = map(Path, sys.argv[1:4])
    items: dict[str, dict] = {}
    for f, tag in ((q_f, "QUEUE"), (a_f, "AUDIT")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            it = items.setdefault(r["row_id"], {**ev(r), "tags": []})
            it["tags"].append(tag)
    data = json.dumps(list(items.values()), ensure_ascii=False).replace("</", "<\\/")
    key = "hall_valid_review_" + out.stem
    out.write_text(
        TEMPLATE.replace("__DATA__", data)
        .replace("__KEY__", key)
        .replace("__N__", str(len(items))),
        encoding="utf-8",
    )
    n_q = sum("QUEUE" in i["tags"] for i in items.values())
    n_a = sum("AUDIT" in i["tags"] for i in items.values())
    print(
        f"wrote {out}: {len(items)} cards (queue {n_q}, audit {n_a}, both {n_q + n_a - len(items)})"
    )


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Hall verified rows - VALID review (__N__)</title>
<style>
body{margin:0;font:15px/1.5 -apple-system,Helvetica,sans-serif;color:#1d2330;background:#f6f7f9}
header{position:sticky;top:0;z-index:5;background:#1d2330;color:#fff;padding:10px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header button,header select{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid #556;background:#2b3343;color:#fff;cursor:pointer}
main{max-width:1100px;margin:18px auto;padding:0 16px 80px}
.guide{background:#fff;border:1px solid #e3e6eb;border-radius:10px;padding:10px 16px;font-size:13.5px;margin-bottom:16px}
.card{background:#fff;border:1px solid #e3e6eb;border-radius:10px;padding:16px 20px;margin-bottom:14px}
.card.cur{box-shadow:0 0 0 3px #93c5fd}.card.done-y{border-left:6px solid #1a7f37}.card.done-n{border-left:6px solid #b42318}.card.done-u{border-left:6px solid #b54708}
.top{display:flex;justify-content:space-between;gap:10px;color:#667085;font-size:13px}.top a{color:#2563eb}
.tag{font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:9px;margin-right:4px}.QUEUE{background:#fee4e2;color:#b42318}.AUDIT{background:#e0eaff;color:#1d4ed8}
.raw{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;background:#f2f4f7;border-radius:6px;padding:9px 12px;white-space:pre-wrap;margin:8px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}th,td{border:1px solid #e3e6eb;padding:5px 8px;text-align:left;vertical-align:top}th{width:150px;background:#f8f9fb;color:#667085}
.why{background:#fffaeb;border:1px solid #fedf89;border-radius:6px;padding:6px 10px;font-size:13.5px}
.links a{font-size:13px;margin-right:10px;color:#2563eb}
.dec{display:flex;gap:10px;align-items:center;margin-top:10px}.dec button{font:inherit;padding:6px 14px;border-radius:7px;border:1px solid #e3e6eb;background:#fff;cursor:pointer}
.dec button.y.on{background:#1a7f37;color:#fff}.dec button.n.on{background:#b42318;color:#fff}.dec button.u.on{background:#b54708;color:#fff}
.dec input{flex:1;font:inherit;padding:5px 8px;border-radius:7px;border:1px solid #e3e6eb}.hidden{display:none}kbd{font:12px ui-monospace,Menlo,monospace;background:#eef0f3;border:1px solid #d0d5dd;border-radius:4px;padding:0 5px}
</style></head><body>
<header><b>Hall verified rows · VALID review</b><span id="prog"></span>
<select id="filt"><option value="all">All</option><option value="todo">Unreviewed</option><option value="QUEUE">QUEUE only</option><option value="AUDIT">AUDIT only</option></select>
<button id="exp">Export CSV</button></header>
<main><div class="guide"><b>Question:</b> is this a correct citation of a real work? VerusCite marked every card <i>verified</i>.
<b>y</b> = real work; title, authors, year and venue are right (formatting, abbreviated venue, a one-letter typo are fine).
<b>n</b> = no such work, or a substantive error (wrong venue, wrong/missing authors, DOI of another paper, wrong year by more than one).
<b>u</b> = cannot tell (add a note).
<span class="tag QUEUE">QUEUE</span> autocheck failed (often: no DOI and Crossref misses arXiv/books).
<span class="tag AUDIT">AUDIT</span> uniform random sample of the final set, used for the error-rate bound.
Keys <kbd>y</kbd> <kbd>n</kbd> <kbd>u</kbd> <kbd>j</kbd>/<kbd>k</kbd> <kbd>c</kbd> note. Autosaves in this browser.</div><div id="list"></div></main>
<script>
const ITEMS=__DATA__,KEY='__KEY__';let S=JSON.parse(localStorage.getItem(KEY)||'{}'),cur=0;
const esc=s=>String(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function render(){document.getElementById('list').innerHTML=ITEMS.map((it,k)=>{const s=S[it.row_id]||{},q=encodeURIComponent(it.cited_title);
return`<section class="card" id="c${k}" data-k="${k}"><div class="top"><span>${it.tags.map(t=>`<span class="tag ${t}">${t}</span>`).join('')}<b>#${k+1}</b> · from: ${esc(it.paper_title)}</span>
<span><a href="${esc(it.original_url)}" target="_blank">original paper</a> · <a href="${esc(it.share_url)}" target="_blank">VerusCite report</a></span></div>
<div class="raw">${esc(it.raw)}</div>
<div class="why">Autocheck: <b>${esc(it.why)}</b></div>
<table><tr><th>Cited</th><td>${esc(it.cited_title)}<br>${esc(it.cited_authors)}<br>${esc(it.cited_venue)} ${esc(it.cited_ids)}</td></tr>
<tr><th>VerusCite match</th><td>${esc(it.match_title)}<br>${esc(it.match_authors)}<br>${esc(it.match_venue)} ${esc(it.match_doi)}</td></tr>
${it.doi?`<tr><th>Cited DOI (${esc(it.doi_status)})</th><td><a href="https://doi.org/${esc(it.doi)}" target="_blank">${esc(it.doi)}</a><br>${esc(it.doi_target)}</td></tr>`:''}
<tr><th>Crossref title search</th><td>${esc(it.search)||'-'}</td></tr>
<tr><th>VerusCite notes</th><td>${esc(it.notes)}</td></tr></table>
<div class="links"><a href="https://scholar.google.com/scholar?q=${q}" target="_blank">Google Scholar</a><a href="https://www.google.com/search?q=%22${q}%22" target="_blank">Google exact title</a><a href="https://arxiv.org/search/?searchtype=title&query=${q}" target="_blank">arXiv</a><a href="https://dblp.org/search?q=${q}" target="_blank">DBLP</a></div>
<div class="dec"><button class="y" data-v="y">y valid</button><button class="n" data-v="n">n not valid</button><button class="u" data-v="u">u unsure</button><input class="cm" placeholder="note" value="${esc(s.c)}"></div></section>`}).join('');
document.querySelectorAll('.card').forEach(el=>{const k=+el.dataset.k;el.querySelectorAll('.dec button').forEach(b=>b.onclick=e=>{e.stopPropagation();set(k,{v:b.dataset.v})});
el.querySelector('.cm').oninput=e=>set(k,{c:e.target.value},false);el.onclick=()=>{cur=k;refresh()}});refresh()}
function set(k,p,adv=true){const id=ITEMS[k].row_id;S[id]=Object.assign({},S[id]||{},p,{at:new Date().toISOString()});localStorage.setItem(KEY,JSON.stringify(S));refresh();if(adv&&p.v)go(k+1)}
function refresh(){const f=document.getElementById('filt').value,done=ITEMS.filter(i=>(S[i.row_id]||{}).v).length;document.getElementById('prog').textContent=`${done} / ${ITEMS.length} reviewed`;
document.querySelectorAll('.card').forEach(el=>{const it=ITEMS[+el.dataset.k],s=S[it.row_id]||{};el.className='card'+(s.v?' done-'+s.v:'')+(+el.dataset.k===cur?' cur':'');
el.querySelectorAll('.dec button').forEach(b=>b.classList.toggle('on',b.dataset.v===s.v));
el.classList.toggle('hidden',(f==='todo'&&!!s.v)||((f==='QUEUE'||f==='AUDIT')&&!it.tags.includes(f)))})}
function go(k){const vis=[...document.querySelectorAll('.card:not(.hidden)')].map(e=>+e.dataset.k);if(!vis.length)return;let n=vis.find(x=>x>=k);if(n===undefined)n=vis[vis.length-1];cur=n;refresh();document.getElementById('c'+n).scrollIntoView({behavior:'smooth',block:'start'})}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'){if(e.key==='Escape')e.target.blur();return}
if('ynu'.includes(e.key)&&e.key.length===1)set(cur,{v:e.key});else if(e.key==='j')go(cur+1);
else if(e.key==='k'){const v=[...document.querySelectorAll('.card:not(.hidden)')].map(e=>+e.dataset.k).filter(x=>x<cur);if(v.length){cur=v[v.length-1];refresh();document.getElementById('c'+cur).scrollIntoView({behavior:'smooth',block:'start'})}}
else if(e.key==='c'){e.preventDefault();document.querySelector('#c'+cur+' .cm').focus()}});
document.getElementById('filt').onchange=refresh;
document.getElementById('exp').onclick=()=>{const r=ITEMS.map(i=>{const s=S[i.row_id]||{};return{row_id:i.row_id,verdict:s.v||'',note:s.c||'',in_queue:i.tags.includes('QUEUE')?1:0,in_audit:i.tags.includes('AUDIT')?1:0,autocheck:i.why,reviewed_at:s.at||'',raw:i.raw}}),h=Object.keys(r[0]);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([[h.join(','),...r.map(o=>h.map(k=>'"'+String(o[k]).replace(/"/g,'""')+'"').join(','))].join('\n')],{type:'text/csv'}));a.download=KEY+'.csv';a.click()};
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
