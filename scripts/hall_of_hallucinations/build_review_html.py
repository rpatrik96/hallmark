"""Build a self-contained HTML page for manually reviewing Hall of Hallucinations rows.

Usage: python3 -I build_review_html.py <sample.csv> <out.html>
Decisions autosave in the browser (localStorage) and export as CSV/JSON.
"""

import csv
import json
import re
import sys
from pathlib import Path

TYPES = [
    ("plausible_fabrication", "No such work exists (whole citation invented)"),
    ("wrong_venue", "Real work, but cited in a different journal/conference"),
    ("swapped_authors", "author_mismatch: real work, wrong/replaced authors"),
    ("hybrid_fabrication", "Real DOI, but it points to a different work"),
    ("fabricated_doi", "DOI does not exist / does not resolve"),
    ("chimeric_title", "Title stitched from parts of real works"),
    ("merged_citation", "Fields of two or more real works merged"),
    ("near_miss_title", "Real work, title subtly altered"),
    ("preprint_as_published", "Preprint cited as a published venue paper"),
    ("partial_author_list", "Real work, author list truncated without et al."),
    ("nonexistent_venue", "Venue itself does not exist"),
    ("placeholder_authors", "Placeholder / generic author names"),
    ("future_date", "Year in the future"),
    ("arxiv_version_mismatch", "Wrong arXiv version details"),
]

RULES = [  # (type, regex on notes, human-readable reason) — first match wins
    (
        "fabricated_doi",
        r"doi[^.]{0,80}(does not resolve|not registered|invalid|404|no record|unregistered)",
        "notes: DOI does not resolve",
    ),
    (
        "hybrid_fabrication",
        r"doi[^.]{0,120}(entirely different|different (paper|work|article)|unrelated|belongs to|points to|another (paper|work))",
        "notes: DOI belongs to a different work",
    ),
    (
        "preprint_as_published",
        r"(preprint|arxiv)[^.]{0,120}(published|proceedings|journal)",
        "notes: preprint cited as published",
    ),
    (
        "swapped_authors",
        r"author[^.]{0,80}(conflict|do not match|does not match|differ|replaced|not (among|on) the)",
        "notes: author list conflicts with the record",
    ),
    (
        "wrong_venue",
        r"(venue chimera|different journal|wrong journal|whole-cloth wrong journal|not an abbreviation)",
        "notes: different journal/venue",
    ),
    (
        "chimeric_title",
        r"(blend|synthes|combin|stitch|hybrid of)[^.]{0,80}title|title[^.]{0,80}(blend|synthes|combin)",
        "notes: title blended from real works",
    ),
    ("merged_citation", r"(combin|mix|merg|blend)", "notes: elements of several works combined"),
    (
        "plausible_fabrication",
        r"(no results|fabricat|no (paper|article|record|evidence)|does not exist|could not (find|locate|verify)|not exist)",
        "notes: no matching work found",
    ),
]


def suggest(r: dict) -> tuple[str, str]:
    n = r["notes"].lower()
    for t, pat, why in RULES:
        if re.search(pat, n):
            return t, why
    if not r["match_title"]:
        return "plausible_fabrication", "no matched record"
    return "", "no rule matched; decide from the evidence"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    full = {}
    fj = src.parent / "hall_hallucinated_rows_flagged.jsonl"
    if fj.exists():
        for line in fj.open(encoding="utf-8"):
            x = json.loads(line)
            full[x["row_id"]] = x
    items = []
    for i, r in enumerate(rows):
        t, why = suggest(r)
        items.append(
            {
                "idx": i + 1,
                "row_id": r["row_id"],
                "paper_title": r["paper_title"],
                "paper_doi": r["paper_doi"],
                "original_url": r["original_url"],
                "share_url": r["share_url"],
                "raw": r["raw"],
                "cited_title": r["cited_title"],
                "cited_venue": r["cited_venue"],
                "match_title": r["match_title"],
                "match_venue": r["match_venue"],
                "match_doi": r["match_doi"],
                "cited_authors": full.get(r["row_id"], {}).get("cited_authors_year", ""),
                "match_authors": full.get(r["row_id"], {}).get("match_authors", ""),
                "notes": r["notes"],
                "sug_type": t,
                "sug_why": why,
            }
        )
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    types = json.dumps(TYPES).replace("</", "<\\/")
    page = (
        TEMPLATE.replace("__DATA__", data)
        .replace("__TYPES__", types)
        .replace("__N__", str(len(items)))
    )
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} with {len(items)} items")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Hall of Hallucinations — manual review (__N__)</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mut:#667085;--line:#e3e6eb;--y:#1a7f37;--n:#b42318;--u:#b54708;--hl:#fff3b0;--del:#ffd7d5;--add:#d3f5dc}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Helvetica,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:5;background:#1d2330;color:#fff;padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}.bar{flex:1;min-width:160px;height:8px;background:#3a4252;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:#4ade80}
header button,header select{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid #556;background:#2b3343;color:#fff;cursor:pointer}
.wrap{display:grid;grid-template-columns:360px minmax(0,1fr);gap:18px;max-width:1560px;margin:18px auto;padding:0 16px}main{min-width:0;padding-bottom:80px}aside.guide{position:sticky;top:64px;align-self:start;max-height:calc(100vh - 80px);overflow-y:auto;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;font-size:13.5px}aside.guide h2{font-size:15px;margin:0 0 8px}aside.guide ul{padding-left:18px;margin:6px 0}aside.guide p{margin:6px 0}@media(max-width:900px){.wrap{grid-template-columns:1fr}aside.guide{position:static;max-height:none}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:16px}
.card.done-y{border-left:6px solid var(--y)}.card.done-n{border-left:6px solid var(--n)}.card.done-u{border-left:6px solid var(--u)}
.card.cur{box-shadow:0 0 0 3px #93c5fd}
.top{display:flex;justify-content:space-between;gap:12px;color:var(--mut);font-size:13px}
.top a{color:#2563eb}
h3{margin:4px 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.raw{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;background:#f2f4f7;border-radius:6px;padding:10px 12px;white-space:pre-wrap}
table.cmp{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
table.cmp th,table.cmp td{border:1px solid var(--line);padding:6px 9px;vertical-align:top;text-align:left}
table.cmp th{background:#f8f9fb;width:110px;color:var(--mut);font-weight:600}
table.cmp td.same{color:var(--mut)}
del{background:var(--del);text-decoration:line-through;text-decoration-color:#b42318}ins{background:var(--add);text-decoration:none}
.flag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:9px;margin-left:6px;background:#fee4e2;color:var(--n)}
.flag.ok{background:#dcfae6;color:var(--y)}.ach{display:inline-block;padding:1px 7px;margin:2px 3px 2px 0;border-radius:5px;font-size:13.5px}.ach.ok{background:var(--add)}.ach.warn{background:#fef0c7}.ach.bad{background:var(--del)}
.notes{background:#fffaeb;border:1px solid #fedf89;border-radius:6px;padding:10px 12px;font-size:14px}
.notes a{word-break:break-all}
.links{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.links a{font-size:13px;padding:3px 9px;border:1px solid var(--line);border-radius:6px;text-decoration:none;color:#2563eb;background:#fff}
.sug{margin:12px 0 6px;padding:8px 12px;border-radius:6px;background:#eef4ff;border:1px solid #c7d7fe;font-size:14px}
.dec{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
.dec button{font:inherit;padding:7px 14px;border-radius:7px;border:1px solid var(--line);background:#fff;cursor:pointer}
.dec button.y.on{background:var(--y);color:#fff;border-color:var(--y)}.dec button.n.on{background:var(--n);color:#fff;border-color:var(--n)}.dec button.u.on{background:var(--u);color:#fff;border-color:var(--u)}
.dec select,.dec input{font:inherit;padding:6px 8px;border-radius:7px;border:1px solid var(--line)}
.dec input{flex:1;min-width:220px}
.help{font-size:13px;color:var(--mut)}kbd{font:12px ui-monospace,Menlo,monospace;background:#eef0f3;border:1px solid #d0d5dd;border-radius:4px;padding:0 5px}
.guide table{border-collapse:collapse;font-size:12.5px;margin-top:8px}.guide td{border-top:1px solid var(--line);padding:4px 10px 4px 0;vertical-align:top}
.hidden{display:none}
</style></head><body>
<header><b>Hall of Hallucinations · manual review</b>
<span id="prog">0 / __N__</span><div class="bar"><i id="pbar" style="width:0"></i></div>
<select id="filt"><option value="all">Show all</option><option value="todo">Only unreviewed</option><option value="u">Only unsure</option></select>
<button id="exp-csv">Export CSV</button><button id="exp-json">Export JSON</button><button id="reset">Reset</button></header>
<div class="wrap"><aside class="guide"><h2>How to judge</h2>
<p><b>Question for each card:</b> does the cited work exist <i>as cited</i>? VerusCite flagged every card as a hallucination; your call is final.</p>
<ul>
<li><b>Yes, hallucination</b> <kbd>y</kbd>: the work does not exist, or it is a real work cited with details that cannot be a typo (whole wrong journal, several wrong authors, a DOI that belongs to another paper, fields of two works merged).</li>
<li><b>No</b> <kbd>n</kbd>: real work and the citation is essentially right. Year off by one, page/volume typo, abbreviated venue, one misspelt name = <b>no</b> (minor error).</li>
<li><b>Unsure</b> <kbd>u</kbd>: evidence is unclear (paywalled, non-English, record not reachable). Add a note.</li>
</ul>
<p>Use the comparison table first (red = in the citation only, green = in the real record only), then the VerusCite notes, then open the links if still unclear.
Keys: <kbd>y</kbd> <kbd>n</kbd> <kbd>u</kbd> decide · <kbd>a</kbd> accept suggestion (yes + suggested type) · <kbd>j</kbd>/<kbd>k</kbd> next/previous · <kbd>t</kbd> change type · <kbd>c</kbd> comment. Everything autosaves in this browser.</p>
<h2 style="margin-top:12px">When several fields are wrong</h2>
<ul>
<li>Title + authors made up, no real paper behind them → <code>plausible_fabrication</code> (a real venue is what makes it plausible)</li>
<li>Same, but the DOI resolves to a different real paper → <code>hybrid_fabrication</code></li>
<li>Same, but the DOI does not resolve → <code>plausible_fabrication</code> (note "fake DOI")</li>
<li>Fields taken from two identifiable real papers → <code>merged_citation</code></li>
<li>Real paper (title + authors right), DOI <i>and</i> venue wrong, DOI resolves to another real paper → <code>merged_citation</code></li>
<li>Same, but the DOI does not resolve → <code>fabricated_doi</code>, note "also wrong venue"</li>
<li>Two defects in general: pick the main one, write the other in the note</li>
<li>Real paper, only one field wrong → that field's type (<code>wrong_venue</code>, <code>swapped_authors</code>, ...)</li>
</ul>
<h2 style="margin-top:12px">Types</h2><table id="tguide"></table></aside>
<main><div id="list"></div></main></div>
<script>
const ITEMS=__DATA__, TYPES=__TYPES__, KEY='hall_review_v1';
let S=JSON.parse(localStorage.getItem(KEY)||'{}'), cur=0;
const esc=s=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify=s=>esc(s).replace(/(https?:\/\/[^\s)]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>');
const norm=s=>(s||'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();
const STOP=new Set(['of','in','the','and','for','on','a','an','de','la','&']);
function isAbbrev(a,b){const A=norm(a).split(' ').filter(w=>w&&!STOP.has(w)),B=norm(b).split(' ').filter(w=>w&&!STOP.has(w));
 if(!A.length||A.length>B.length)return false;let j=0;for(const w of A){while(j<B.length&&!B[j].startsWith(w))j++;if(j===B.length)return false;j++}return true}
const isInit=t=>/^[A-Z]\.?([A-Z]\.?)*$/.test(t)||t.replace(/\./g,'').length<=1;
function cParts(n){return n.replace(/\(\d{4}\)/,'').replace(/et al\.?/i,'').split(/[\s,]+/).filter(Boolean)}
function authorCmp(c,m){const cited=(c||'').replace(/\s*\(\d{4}[a-z]?\)\s*$/,'').split(';').map(x=>x.trim()).filter(x=>x&&!/^et al\.?$/i.test(x));
 const real=(m||'').split(';').map(x=>x.trim()).filter(Boolean);
 if(!cited.length||!real.length)return null;
 const realS=real.map(r=>{if(r.includes(',')){const[sur,giv]=r.split(',',2).map(x=>x.trim());return{sur:norm(sur),all:[norm(sur),...giv.split(/[\s.]+/).map(norm)],ini:norm(giv)[0]||''}}const p=r.split(/\s+/);return{sur:norm(p[p.length-1]),all:p.map(norm),ini:norm(p[0])[0]||''}});
 const chips=cited.map(a=>{const ps=cParts(a),names=ps.filter(t=>!isInit(t)).map(norm),inits=ps.filter(isInit).map(t=>t.replace(/\./g,'').toLowerCase());
  const hit=realS.find(r=>names.some(n=>n===r.sur));
  if(hit){const ok=!inits.length||inits.some(i=>i[0]===hit.ini)||names.some(n=>hit.all.includes(n)&&n!==hit.sur);return`<span class="ach ${ok?'ok':'warn'}" title="${ok?'surname found in record':'surname found, initial differs'}">${esc(a)}</span>`}
  return`<span class="ach bad" title="not among the record's authors">${esc(a)}</span>`});
 const nb=chips.filter(x=>x.includes('ach bad')).length;
 return{cited:chips.join(' '),real:real.map(esc).join('; '),nbad:nb,n:cited.length}}
function diff(a,b){const A=(a||'').split(/\s+/).filter(Boolean),B=(b||'').split(/\s+/).filter(Boolean);
 const m=A.length,n=B.length,L=Array.from({length:m+1},()=>new Array(n+1).fill(0));
 for(let i=m-1;i>=0;i--)for(let j=n-1;j>=0;j--)L[i][j]=norm(A[i])===norm(B[j])?L[i+1][j+1]+1:Math.max(L[i+1][j],L[i][j+1]);
 let i=0,j=0,x=[],y=[];while(i<m&&j<n){if(norm(A[i])===norm(B[j])){x.push(esc(A[i]));y.push(esc(B[j]));i++;j++}else if(L[i+1][j]>=L[i][j+1]){x.push('<del>'+esc(A[i])+'</del>');i++}else{y.push('<ins>'+esc(B[j])+'</ins>');j++}}
 while(i<m)x.push('<del>'+esc(A[i++])+'</del>');while(j<n)y.push('<ins>'+esc(B[j++])+'</ins>');return [x.join(' '),y.join(' ')]}
function arow(it){const r=authorCmp(it.cited_authors,it.match_authors);if(!r)return it.cited_authors?`<tr><th>Authors</th><td>${esc(it.cited_authors)}</td><td class="same">${esc(it.match_authors)||'—'}</td></tr>`:'';
 const f=r.nbad?`<span class="flag">${r.nbad} of ${r.n} not in record</span>`:'<span class="flag ok">all found</span>';
 return`<tr><th>Authors${f}</th><td>${r.cited}</td><td>${r.real}</td></tr>`}
function vrow(a,b){if(a&&b&&norm(a)!==norm(b)&&isAbbrev(a,b))return`<tr><th>Venue<span class="flag ok">abbreviation</span></th><td class="same">${esc(a)}</td><td class="same">${esc(b)}</td></tr>`;return row('Venue',a,b)}
function row(lbl,a,b){if(!a&&!b)return'';if(!b)return`<tr><th>${lbl}</th><td>${esc(a)}</td><td class="same">—</td></tr>`;
 const same=norm(a)===norm(b),[da,db]=diff(a,b);
 return`<tr><th>${lbl}${same?'<span class="flag ok">same</span>':'<span class="flag">differs</span>'}</th><td class="${same?'same':''}">${same?esc(a):da}</td><td class="${same?'same':''}">${same?esc(b):db}</td></tr>`}
function typeOpts(sel){return '<option value="">— type —</option>'+TYPES.map(([k,d])=>`<option value="${k}" ${k===sel?'selected':''}>${k}</option>`).join('')}
function render(){
 document.getElementById('tguide').innerHTML=TYPES.map(([k,d])=>`<tr><td><code>${k}</code></td><td>${esc(d)}</td></tr>`).join('');
 document.getElementById('list').innerHTML=ITEMS.map((it,k)=>{const s=S[it.row_id]||{};const q=encodeURIComponent(it.cited_title||it.raw.slice(0,120));
 return`<section class="card" id="c${k}" data-k="${k}">
 <div class="top"><span><b>#${it.idx}</b> of ${ITEMS.length} · from: ${esc(it.paper_title)}</span><span><a href="${esc(it.original_url)}" target="_blank">original paper</a> · <a href="${esc(it.share_url)}" target="_blank">VerusCite report</a></span></div>
 <h3>Citation as printed in the paper</h3><div class="raw">${esc(it.raw)}</div>
 <h3 style="margin-top:14px">Citation vs. closest real record found by VerusCite</h3>
 ${it.match_title?`<table class="cmp"><tr><th></th><th>In the citation</th><th>Real record</th></tr>${row('Title',it.cited_title,it.match_title)}${arow(it)}${vrow(it.cited_venue,it.match_venue)}${it.match_doi?`<tr><th>DOI of record</th><td class="same">—</td><td><a href="https://doi.org/${esc(it.match_doi)}" target="_blank">${esc(it.match_doi)}</a></td></tr>`:''}</table>`:'<p class="flag">VerusCite found no matching record</p>'}
 <h3>VerusCite's reasoning</h3><div class="notes">${linkify(it.notes)}</div>
 <div class="links">${it.match_doi?`<a href="https://doi.org/${esc(it.match_doi)}" target="_blank">Open record DOI</a>`:''}<a href="https://scholar.google.com/scholar?q=${q}" target="_blank">Google Scholar: cited title</a><a href="https://search.crossref.org/search/works?q=${q}&from_ui=yes" target="_blank">Crossref: cited title</a><a href="https://www.google.com/search?q=%22${q}%22" target="_blank">Google: exact title</a></div>
 <div class="sug">Suggestion: <b>${it.sug_type?'Yes, '+it.sug_type:'decide manually'}</b> <span class="help">(${esc(it.sug_why)})</span></div>
 <div class="dec"><button class="y ${s.v==='y'?'on':''}" data-v="y">Yes, hallucination</button><button class="n ${s.v==='n'?'on':''}" data-v="n">No (real / minor error)</button><button class="u ${s.v==='u'?'on':''}" data-v="u">Unsure</button>
 <select class="ty">${typeOpts(s.t!==undefined?s.t:it.sug_type)}</select><input class="cm" placeholder="note (optional)" value="${esc(s.c||'')}"></div></section>`}).join('');
 document.querySelectorAll('.card').forEach(el=>{const k=+el.dataset.k,it=ITEMS[k];
  el.querySelectorAll('.dec button').forEach(b=>b.onclick=()=>{set(k,{v:b.dataset.v})});
  el.querySelector('.ty').onchange=e=>set(k,{t:e.target.value},false);
  el.querySelector('.cm').oninput=e=>set(k,{c:e.target.value},false);
  el.onclick=()=>{cur=k;mark()}});
 refresh();}
function set(k,patch,adv=true){const id=ITEMS[k].row_id;S[id]=Object.assign({t:ITEMS[k].sug_type},S[id]||{},patch,{at:new Date().toISOString()});
 if(patch.v==='n')S[id].t='';localStorage.setItem(KEY,JSON.stringify(S));refresh(k);if(adv&&patch.v)go(k+1)}
function refresh(k){const done=ITEMS.filter(i=>S[i.row_id]&&S[i.row_id].v).length;document.getElementById('prog').textContent=`${done} / ${ITEMS.length} reviewed`;
 document.getElementById('pbar').style.width=(100*done/ITEMS.length)+'%';
 const f=document.getElementById('filt').value;
 document.querySelectorAll('.card').forEach(el=>{const it=ITEMS[+el.dataset.k],s=S[it.row_id]||{};
  el.className='card'+(s.v?' done-'+s.v:'')+(+el.dataset.k===cur?' cur':'');
  el.querySelectorAll('.dec button').forEach(b=>b.classList.toggle('on',b.dataset.v===s.v));
  if(s.t!==undefined)el.querySelector('.ty').value=s.t;
  el.classList.toggle('hidden',(f==='todo'&&!!s.v)||(f==='u'&&s.v!=='u'))})}
function mark(){refresh()}
function go(k){const vis=[...document.querySelectorAll('.card:not(.hidden)')].map(e=>+e.dataset.k);if(!vis.length)return;
 let nk=vis.find(x=>x>=k);if(nk===undefined)nk=vis[vis.length-1];cur=nk;refresh();document.getElementById('c'+nk).scrollIntoView({behavior:'smooth',block:'start'})}
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)){if(e.key==='Escape')e.target.blur();return}
 const k=cur;if(e.key==='y'||e.key==='n'||e.key==='u')set(k,{v:e.key});else if(e.key==='a')set(k,{v:'y',t:ITEMS[k].sug_type});
 else if(e.key==='j'||e.key==='ArrowDown'&&e.altKey)go(k+1);else if(e.key==='k'){const vis=[...document.querySelectorAll('.card:not(.hidden)')].map(e=>+e.dataset.k).filter(x=>x<k);if(vis.length){cur=vis[vis.length-1];refresh();document.getElementById('c'+cur).scrollIntoView({behavior:'smooth',block:'start'})}}
 else if(e.key==='t'){e.preventDefault();document.querySelector('#c'+k+' .ty').focus()}else if(e.key==='c'){e.preventDefault();document.querySelector('#c'+k+' .cm').focus()}});
document.getElementById('filt').onchange=()=>refresh();
function dl(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click()}
function out(){return ITEMS.map(i=>{const s=S[i.row_id]||{};return{row_id:i.row_id,verdict:s.v||'',hallmark_type:s.v==='y'?(s.t!==undefined?s.t:i.sug_type):'',note:s.c||'',suggested_type:i.sug_type,reviewed_at:s.at||'',raw:i.raw,paper_doi:i.paper_doi,share_url:i.share_url}})}
document.getElementById('exp-json').onclick=()=>dl('hall_review_decisions.json',JSON.stringify(out(),null,1),'application/json');
document.getElementById('exp-csv').onclick=()=>{const r=out(),h=Object.keys(r[0]);dl('hall_review_decisions.csv',[h.join(','),...r.map(o=>h.map(k=>'"'+String(o[k]).replace(/"/g,'""')+'"').join(','))].join('\n'),'text/csv')};
document.getElementById('reset').onclick=()=>{if(confirm('Delete all decisions saved in this browser?')){S={};localStorage.removeItem(KEY);render()}};
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
