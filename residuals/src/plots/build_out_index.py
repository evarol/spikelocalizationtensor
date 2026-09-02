"""Single self-contained hub that routes to everything in residuals/out/.

Same idea as spiketensor/browser.py: one index.html with the run inventory
embedded as JSON, filter buttons, a sortable table, and click-through to the
real artifacts. Scope difference: this is a router over the whole out/ tree
(galleries, loose plot suites, standalone figures, docs), not a per-fit panel
browser, so the collector is a filesystem walk joined against the backing
residuals/runs/dataset1_p1/<tag>/summary.json + config.json when they exist.

Usage:
    singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
        /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
        /bin/bash -c "source /ext3/env.sh && python residuals/src/plots/build_out_index.py"
    open residuals/out/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "residuals" / "out"
RUNS = REPO / "residuals" / "runs"

VIEWABLE = {".png", ".pdf", ".html", ".mp4", ".jpg", ".jpeg", ".svg",
            ".pptx", ".npz", ".json", ".txt", ".md", ".npy"}
JUNK = {".aux", ".log", ".out", ".synctex.gz", ".fls", ".fdb_latexmk"}
THUMB_LIMIT = 8
LINK_CAP = 80


def family_of(name: str) -> str:
    m = re.match(r"(\d{4})_", name)
    return m.group(1) if m else "misc"


def load_run_meta(runs: Path, tag: str) -> dict:
    base = None
    for cand in (runs / "dataset1_p1" / tag, runs / tag):
        if cand.is_dir():
            base = cand
            break
    meta = {"n_events": None, "n_rejected": None, "stopping_reason": None,
            "threshold": None, "bar": None, "bars": None, "prototype_count": None,
            "run_href": None}
    if base is None:
        return meta
    meta["run_href"] = str(Path("..") / "runs" / base.relative_to(RUNS)) + "/"
    try:
        s = json.loads((base / "summary.json").read_text())
    except (OSError, ValueError):
        s = {}
    meta["n_events"] = s.get("n_events")
    meta["n_rejected"] = s.get("n_rejected")
    meta["stopping_reason"] = s.get("stopping_reason")
    passes = s.get("pass_summaries") or []
    bars = sorted({p.get("channel_fraction") for p in passes
                   if p.get("channel_fraction") is not None})
    if bars:
        meta["bar"] = bars[0]
        meta["bars"] = [round(b, 2) for b in bars]
    try:
        c = json.loads((base / "config.json").read_text())["config"]
    except (OSError, ValueError, KeyError):
        c = {}
    meta["threshold"] = c.get("threshold")
    meta["prototype_count"] = c.get("prototype_count")
    return meta


def fmt_n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else "—"


def gallery_row(d: Path, runs: Path) -> dict:
    pngs = sorted(d.glob("*.png"))
    row = load_run_meta(runs, d.name)
    title = d.name
    try:
        head = (d / "index.html").read_text(errors="ignore")[:2048]
        m = re.search(r"<title>(.*?)</title>", head)
        if m:
            row["gallery_title"] = m.group(1)
    except OSError:
        pass
    row.update({"tag": d.name, "family": family_of(d.name), "kind": "gallery",
                "gallery_href": f"{d.name}/index.html",
                "n_pngs": len(pngs),
                "updated": datetime.fromtimestamp(
                    (d / "index.html").stat().st_mtime, timezone.utc)
                    .strftime("%Y-%m-%d %H:%M")})
    return row


def collection_row(d: Path) -> dict:
    groups = []
    n_total = 0
    for p in sorted(d.rglob("*")):
        if p.is_dir() or p.suffix in JUNK or p.suffix not in VIEWABLE:
            continue
        n_total += 1
    top_pngs = sorted(d.glob("*.png"))
    thumb = top_pngs[0].name if top_pngs else None
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        files = [f for f in sorted(sub.iterdir())
                 if f.is_file() and f.suffix in VIEWABLE and f.suffix not in JUNK]
        if files:
            groups.append({"sub": sub.name,
                           "files": [{"name": f.name, "href": f"{d.name}/{sub.name}/{f.name}"}
                                     for f in files[:LINK_CAP]],
                           "n": len(files)})
    top_files = [f for f in sorted(d.iterdir())
                 if f.is_file() and f.suffix in VIEWABLE and f.suffix not in JUNK]
    if top_files:
        groups.insert(0, {"sub": None,
                          "files": [{"name": f.name, "href": f"{d.name}/{f.name}"}
                                    for f in top_files[:LINK_CAP]],
                          "n": len(top_files)})
    mtime = max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()),
                default=d.stat().st_mtime)
    return {"tag": d.name, "family": family_of(d.name), "kind": "collection",
            "n_files": n_total, "thumb": (f"{d.name}/{thumb}" if thumb else None),
            "groups": groups, "updated": datetime.fromtimestamp(
                mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")}


def collect(out: Path, runs: Path) -> dict:
    galleries, collections, figures = [], [], []
    for p in sorted(out.iterdir()):
        if p.name == "index.html":
            continue
        if p.is_dir():
            if (p / "index.html").is_file():
                galleries.append(gallery_row(p, runs))
            else:
                collections.append(collection_row(p))
        elif p.suffix in VIEWABLE and p.suffix not in JUNK:
            figures.append({"name": p.name, "href": p.name,
                            "family": family_of(p.name),
                            "updated": datetime.fromtimestamp(
                                p.stat().st_mtime, timezone.utc)
                                .strftime("%Y-%m-%d %H:%M")})
    return {"galleries": galleries, "collections": collections,
            "figures": figures, "generated": datetime.now(timezone.utc)
            .strftime("%Y-%m-%d %H:%M UTC")}


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>residual campaign browser</title>
<style>
:root{--bg:#0f1115;--fg:#e7e9ee;--dim:#9aa3b2;--card:#171a21;--line:#272c36;--acc:#4c8dff}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#14171c;--dim:#5b6472;
      --card:#f5f6f8;--line:#dde1e8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-sans-serif,-apple-system,'Segoe UI',sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:16px}
h2{font-size:13px;margin:18px 0 8px;color:var(--dim);font-weight:600}
.note{color:var(--dim);font-size:12px;max-width:1150px}
.controls{display:flex;flex-wrap:wrap;gap:14px;padding:10px 18px;
          border-bottom:1px solid var(--line)}
.grp{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.grp>b{color:var(--dim);font-weight:600;font-size:11px;margin-right:2px}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);
       border-radius:5px;padding:3px 9px;cursor:pointer;font:inherit;font-size:12px}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}
main{padding:14px 18px}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th{text-align:right;padding:5px 7px;border-bottom:1px solid var(--line);color:var(--dim);
   cursor:pointer;white-space:nowrap;user-select:none}
th:first-child,td:first-child{text-align:left}
td{padding:4px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
td a{color:var(--acc);text-decoration:none;margin-right:9px}
td a:hover{text-decoration:underline}
tr:hover td{background:rgba(127,127,127,.08)}
.count{color:var(--dim);font-size:12px;padding:6px 0}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.card{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:9px 11px}
.card h3{margin:0 0 6px;font-size:12.5px}
.card .meta{color:var(--dim);font-size:11px;margin-bottom:7px}
.files{margin:0;padding:0;list-style:none;column-gap:18px;font-size:11.5px}
.files li{break-inside:avoid}
.files a{color:var(--acc);text-decoration:none;word-break:break-all}
.files a:hover{text-decoration:underline}
.sub{color:var(--dim);font-weight:600;font-size:11px;margin:7px 0 2px}
.card img{max-width:100%;border:1px solid var(--line);border-radius:5px;margin:4px 0 8px;
          display:block;background:#fff}
.card details summary{cursor:pointer;color:var(--dim);font-size:11px}
figure{margin:0}
figure img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:11px;color:var(--dim);padding:3px 0}
</style>
<header>
  <h1>residual campaign browser</h1>
  <div class="note">One page that routes to everything under <b>residuals/out/</b>.
  <b>Galleries</b> are the per-run plot suites (click <i>open</i> for the gallery,
  <i>data</i> for the backing run directory in residuals/runs/).
  <b>Standalone figures</b> are top-level images; <b>collections</b> are directories
  from earlier sessions without a gallery. Generated __GENERATED__.</div>
</header>
<div class="controls"><div class="grp"><b>family</b><span id="fams"></span></div></div>
<main>
  <h2>galleries</h2>
  <table id="tbl"><thead><tr>
    <th data-k="tag">run</th><th data-k="family">fam</th><th data-k="threshold">thr</th>
    <th data-k="bar">bar</th><th data-k="n_events">events</th>
    <th data-k="n_rejected">rejected</th><th data-k="stopping_reason">stopping</th>
    <th data-k="n_pngs">pngs</th><th data-k="updated">updated</th><th></th>
  </tr></thead><tbody></tbody></table>
  <div class="count" id="gcount"></div>
  <h2>standalone figures</h2>
  <div class="grid" id="figs"></div>
  <h2>collections</h2>
  <div class="grid" id="cols"></div>
</main>
<script>
const D=__PAYLOAD__;
const fmt=n=>n==null?"—":n.toLocaleString("en-US");
let fam="all";
const fams=[...new Set([...D.galleries,...D.collections,...D.figures].map(r=>r.family))].sort();
const fs=document.getElementById("fams");
for(const f of ["all",...fams]){
  const b=document.createElement("button");b.textContent=f;b.onclick=()=>{fam=f;render();};
  fs.appendChild(b);
}
let sortK="tag",sortDir=1;
document.querySelectorAll("#tbl th").forEach(th=>{
  th.onclick=()=>{const k=th.dataset.k;if(!k)return;
    sortDir=(sortK===k)?-sortDir:1;sortK=k;render();};
});
function cmp(a,b){const x=a[sortK],y=b[sortK];
  if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;
  if(typeof x==="number"&&typeof y==="number")return (x-y)*sortDir;
  return String(x).localeCompare(String(y))*sortDir;}
function render(){
  document.querySelectorAll("#fams button").forEach(b=>
    b.classList.toggle("on",b.textContent===fam));
  const tb=document.querySelector("#tbl tbody");tb.innerHTML="";
  const rows=D.galleries.filter(r=>fam==="all"||r.family===fam).sort(cmp);
  for(const r of rows){
    const tr=document.createElement("tr");
    tr.innerHTML=`<td title="${r.gallery_title??""}">${r.tag}</td><td>${r.family}</td><td>${r.threshold??"—"}</td>`+
      `<td>${r.bars?r.bars.join("→"):"—"}</td><td>${fmt(r.n_events)}</td>`+
      `<td>${fmt(r.n_rejected)}</td><td>${r.stopping_reason??"—"}</td>`+
      `<td>${r.n_pngs}</td><td>${r.updated}</td>`+
      `<td><a href="${r.gallery_href}">open</a>`+
      (r.run_href?`<a href="${r.run_href}">data</a>`:"")+`</td>`;
    tb.appendChild(tr);
  }
  document.getElementById("gcount").textContent=`${rows.length} galleries`;
  const fg=document.getElementById("figs");fg.innerHTML="";
  for(const f of D.figures.filter(r=>fam==="all"||r.family===fam)){
    if(!/\.(png|jpg|svg)$/i.test(f.href)){
      addCard(fg,[{sub:null,files:[{name:f.name,href:f.href}]}],f.name,null);continue;}
    const fig=document.createElement("figure");
    fig.innerHTML=`<a href="${f.href}"><img loading="lazy" src="${f.href}"></a>`+
      `<figcaption><a href="${f.href}">${f.name}</a> · ${f.updated}</figcaption>`;
    fg.appendChild(fig);
  }
  const cg=document.getElementById("cols");cg.innerHTML="";
  for(const c of D.collections.filter(r=>fam==="all"||r.family===fam)){
    addCard(cg,c.groups,c.tag,c.thumb,c.n_files,c.updated);
  }
}
function addCard(parent,groups,title,thumb,nFiles,updated){
  const card=document.createElement("div");card.className="card";
  let inner=`<h3>${title}</h3>`;
  if(thumb)inner+=`<a href="${groups[0]?groups[0].files[0].href:thumb}">`+
    `<img loading="lazy" src="${thumb}"></a>`;
  if(nFiles!=null)inner+=`<div class="meta">${nFiles} files${updated?" · "+updated:""}</div>`;
  inner+=`<details><summary>files</summary>`;
  for(const g of groups){
    if(g.sub)inner+=`<div class="sub">${g.sub} (${g.n})</div>`;
    inner+=`<ul class="files">`+g.files.map(f=>
      `<li><a href="${f.href}">${f.name}</a></li>`).join("")+`</ul>`;
  }
  inner+=`</details>`;
  card.innerHTML=inner;parent.appendChild(card);
}
render();
</script>
"""


def build_payload(data: dict) -> str:
    txt = json.dumps(data, indent=None, default=str)
    return txt.replace("</", "<\\/")


def build() -> Path:
    data = collect(OUT, RUNS)
    page = HTML.replace("__GENERATED__", data["generated"])
    page = page.replace("__PAYLOAD__", build_payload(data))
    dest = OUT / "index.html"
    dest.write_text(page)
    n_g = len(data["galleries"])
    n_c = len(data["collections"])
    n_f = len(data["figures"])
    print(f"wrote {dest} ({n_g} galleries, {n_c} collections, {n_f} standalone "
          f"figures, {dest.stat().st_size // 1024} KiB)", flush=True)
    return dest


def main() -> None:
    build()


if __name__ == "__main__":
    main()
