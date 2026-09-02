"""Build a SpikeTensor-style offline browser for one residual-pursuit plot suite."""

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}
HTML_SUFFIXES = {".html", ".htm"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES | HTML_SUFFIXES

GROUPS = {
    "pursuit": "pursuit and stopping",
    "temporal": "temporal model",
    "localization": "localization",
    "reconstruction": "reconstruction",
    "depth_time": "depth × time",
    "other": "other",
}

PANEL_REGISTRY = {
    "peeling_overview.png": (
        "pursuit",
        "peeling overview",
        "Accepted events, residual passes, peeling rounds, and energy capture.",
    ),
    "stopping_diagnostics.png": (
        "pursuit",
        "stopping diagnostics",
        "Round occupancy, raw-energy drop, and pursuit stopping behavior.",
    ),
    "spiketensor/spiketensor_spikes.png": (
        "pursuit",
        "SpikeTensor-style spikes",
        "Full-recording event and fitted-amplitude summary.",
    ),
    "temporal_prototype_cones.png": (
        "temporal",
        "bipolar temporal prototype cones",
        "The positive/negative mother shapes, assigned atoms, and 35-degree cone boundary.",
    ),
    "temporal_codebook_usage.png": (
        "temporal",
        "temporal codebook usage",
        "Every Omega row with recording-wide and per-round assignment usage.",
    ),
    "localization_by_round_cohort.png": (
        "localization",
        "localization by round cohort",
        "Localization distributions stratified by peeling round.",
    ),
    "xyz_localization_by_round.png": (
        "localization",
        "xyz localization by round",
        "Three-dimensional source coordinates across peeling rounds.",
    ),
    "xyz_localization_density.png": (
        "localization",
        "xyz localization density",
        "Full-recording spatial density of fitted source locations.",
    ),
    "xyzsigma_localization_scatter.png": (
        "localization",
        "xyz-sigma localization scatter",
        "Fitted source coordinates and selected spatial scale.",
    ),
    "spiketensor/spiketensor_localization_density.png": (
        "localization",
        "SpikeTensor-style localization density",
        "Probe-plane localization density in the SpikeTensor visual convention.",
    ),
    "reconstruction_examples_by_round.png": (
        "reconstruction",
        "reconstruction examples by round",
        "Observed and predicted snippets sampled across peeling rounds.",
    ),
    "reconstruction_examples_score_boundary.png": (
        "reconstruction",
        "score-boundary reconstructions",
        "Examples near the fitted-projection acceptance boundary.",
    ),
    "reconstructions/reconstruction_diagnostics.png": (
        "reconstruction",
        "reconstruction diagnostics",
        "Captured fraction and channel-normalized reconstruction errors.",
    ),
    "reconstructions/reconstruction_examples.png": (
        "reconstruction",
        "raw residual reconstruction examples",
        "Observed, predicted, and residual waveforms from saved raw-pursuit shards.",
    ),
    "depth_time_omega_raster.png": (
        "depth_time",
        "depth × time Omega raster",
        "Recording-wide event raster colored by selected temporal atom.",
    ),
    "recording_replay_chunk0.png": (
        "reconstruction",
        "recording replay (chunk 0)",
        "Preprocessed input and residuals after each 0019 recording pass, replayed from saved chunk fits.",
    ),
    "recording_replay_chunk001580.png": (
        "reconstruction",
        "recording replay (chunk 1580, most subtractive)",
        "The highest captured-energy chunk: input versus residuals after each recording pass.",
    ),
    "recording_replay_full_recording.png": (
        "reconstruction",
        "full-recording replay",
        "Every chunk replayed: preprocessed input versus residuals after each recording pass, signed-block decimated.",
    ),
    "spiketensor/spiketensor_depth_time_basis.png": (
        "depth_time",
        "SpikeTensor-style depth × time basis",
        "Depth-time density separated by the fitted temporal basis index.",
    ),
}

UNAVAILABLE_STANDARD_PANELS = {
    "It and aggregate one-second movies": (
        "0018 did not render or save the per-second image stack used by the SpikeTensor movie panels."
    ),
    "optimization convergence": (
        "the alternating calibration objective history was not persisted; peeling diagnostics are available instead."
    ),
    "PCA / UMAP / t-SNE coefficient embeddings": (
        "0018 stores a one-hot temporal index and scalar gain, not SpikeTensor's dense shared coefficient vectors."
    ),
    "soft-versus-hard localization readout": (
        "0018 has one hard source fit per accepted event and no paired soft readout or retained reference-spike map."
    ),
    "multipole decomposition and support diagnostics": (
        "the residual model is single-source; it has no multi-source support, source weights, pair gain, LOO delta, or condition number."
    ),
    "rigid/nonrigid DREDge and corrected panel families": (
        "the required time/depth/amplitude inputs exist, but 0018 did not compute or save DREDge motion estimates or corrected localizations."
    ),
    "interactive source-cloud atom viewer": (
        "a single-source cloud can be derived, but the bounded waveform sample, full-probe pack, and DREDge pack expected by atom_viewer.py were not saved."
    ),
}


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def media_url(path, output):
    relative = Path(os.path.relpath(path, output.parent)).as_posix()
    return quote(relative, safe="/")


def panel_info(path, plot_root, output):
    relative = path.relative_to(plot_root).as_posix()
    group, label, description = PANEL_REGISTRY.get(
        relative,
        (
            "other",
            path.stem.replace("_", " ").replace("-", " "),
            f"Additional generated output: {relative}.",
        ),
    )
    suffix = path.suffix.lower()
    media_type = "image" if suffix in IMAGE_SUFFIXES else "video"
    if suffix in HTML_SUFFIXES:
        media_type = "html"
    return {
        "key": relative,
        "group": group,
        "group_label": GROUPS[group],
        "label": label,
        "description": description,
        "href": media_url(path, output),
        "type": media_type,
    }


def detail_values(summary, metadata):
    config = metadata.get("config", {})
    fs = metadata.get("fs")
    first_sample = metadata.get("first_sample")
    stop_sample = metadata.get("stop_sample")
    duration = None
    if fs and first_sample is not None and stop_sample is not None:
        duration = (stop_sample - first_sample) / fs
    return [
        ("events", f"{int(summary['n_events']):,}" if "n_events" in summary else "—"),
        ("chunks", f"{int(summary['n_chunks']):,}" if "n_chunks" in summary else "—"),
        ("duration", f"{duration / 60:.2f} min" if duration is not None else "—"),
        ("channels", metadata.get("n_channels", "—")),
        ("detector", metadata.get("discovery_peak_sign", "—")),
        ("threshold", f"{config.get('threshold', '—')} noise units"),
        ("temporal atoms", config.get("q", "—")),
        ("prototype pair", f"{config.get('prototype_count', '—')} (+/−)"),
        ("cone half-angle", f"{config.get('prototype_cone_deg', '—')}°"),
        ("peeling rounds", config.get("peeling_rounds", "—")),
        ("spatial kernel", config.get("kernel", "—")),
        ("waveforms", summary.get("waveforms", "—")),
    ]


HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0f1115;--fg:#e7e9ee;--dim:#9aa3b2;--card:#171a21;--line:#272c36;--acc:#4c8dff}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#14171c;--dim:#5b6472;
      --card:#f5f6f8;--line:#dde1e8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-sans-serif,-apple-system,'Segoe UI',sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:16px}.note{color:var(--dim);font-size:12px;max-width:1150px}
.controls{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:14px;
          padding:10px 18px;border-bottom:1px solid var(--line);background:var(--bg)}
.grp{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.grp>b{color:var(--dim);font-weight:600;font-size:11px;margin-right:2px}
button,input{background:var(--card);color:var(--fg);border:1px solid var(--line);
       border-radius:5px;padding:3px 9px;font:inherit;font-size:12px}
button{cursor:pointer}button.on{background:var(--acc);border-color:var(--acc);color:#fff}
input{width:min(330px,75vw)}main{padding:14px 18px;max-width:1500px}
.count{color:var(--dim);font-size:12px;padding:6px 18px}
.detail{margin:0 0 14px;padding:13px 14px;border:1px solid var(--line);border-radius:8px;
        background:var(--card);max-width:1180px}.detail h2{margin:0 0 7px;color:var(--fg);font-size:15px}
.detail p{margin:6px 0}.detail-grid{display:grid;
        grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:7px 15px}
.detail-grid div{color:var(--dim)}.detail-grid b{display:block;color:var(--fg);font-size:12px}
.badge{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:700;font-size:11px;
       border:1px solid var(--line);margin-right:6px}
.badge.ref{background:rgba(47,158,68,.18);color:#69db7c}
.badge.diag{background:rgba(132,94,247,.18);color:#b197fc}
.explore-note{margin-top:8px;padding:8px 10px;border-left:3px solid #845ef7;
              background:rgba(132,94,247,.09);max-width:1150px}
.unavailable{color:var(--dim);font-size:11px}.unavailable li{margin:5px 0}.unavailable b{color:var(--fg)}
.panels h2{font-size:13px;margin:16px 0 5px;color:var(--dim);font-weight:600}
.panels figure{margin:0 0 22px}.panels figcaption{margin:0 0 5px}
.panels figcaption b{display:block;color:var(--fg);font-size:13px}.panels figcaption span{color:var(--dim)}
.panels img,.panels video{max-width:100%;border:1px solid var(--line);border-radius:6px;
              background:#fff;display:block}.panels video{background:#0f1115}
.panels a{color:var(--acc);text-decoration:none}.open-html{display:grid;place-items:center;
              min-height:220px;border:1px dashed var(--line);border-radius:6px;background:var(--card)}
.sheet{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
.sheet figure{margin:0}.sheet figcaption{font-size:11px;color:var(--dim);padding:3px 0}
.missing{max-width:950px;padding:8px 10px;border-left:3px solid #c92a2a;background:rgba(201,42,42,.08)}
pre{max-height:420px;overflow:auto;font-size:11px}details summary{cursor:pointer}
@media(max-width:540px){.sheet{grid-template-columns:1fr}.controls,header,main{padding-left:10px;padding-right:10px}}
</style>
<header><h1>__TITLE__</h1><div class="note">
Y[s,c,t] ≈ α<sub>s</sub> g(Δ<sub>s,c</sub>; σ<sub>s</sub>) Ω<sub>q(s)</sub>[t],
fitted directly to raw recording snippets by threshold discovery and residual peeling.
<b>Ω has an explicit positive/negative prototype pair:</b> every temporal atom stays inside
the assigned 35° one-sided cone, and α ≥ 0 preserves polarity. This page deliberately mirrors
the SpikeTensor browser's typography, controls, detail card, full-panel stack, and contact sheet.
The scientific state is not identical: panels whose defining quantities were not saved by 0018
are disclosed below rather than silently approximated.
<div class="explore-note"><b>Browser inclusion is not scientific equivalence.</b>
The available panels are native 0018 residual-pursuit diagnostics. A similarly named SpikeTensor
panel is only exact when its required state has a direct 0018 counterpart.</div>
</div></header>
<div class="controls" id="controls"></div><div class="count" id="count"></div>
<main><section class="detail"><h2>__RUN_NAME__</h2>
<p><span class="badge ref">completed full recording</span><span class="badge diag">single-source residual pursuit</span></p>
<div class="detail-grid">__DETAILS__</div>
<p><b>Model:</b> __MODEL__</p><p><b>Detector:</b> __DETECTOR__</p>
<details class="unavailable"><summary>Unavailable standard SpikeTensor panels and why</summary>
<ul>__UNAVAILABLE__</ul></details>
<details><summary>saved run metadata</summary><pre>__METADATA__</pre></details>
</section><div class="panels" id="panels"></div></main>
<script>
const PANELS=__PANELS__, GROUPS=__GROUPS__;
const state={group:null,view:"stack",query:""};
const media=p=>p.type==="video"
  ? `<video src="${p.href}" controls loop muted playsinline preload="metadata"></video>`
  : p.type==="html" ? `<a class="open-html" href="${p.href}" target="_blank">open interactive HTML</a>`
  : `<a href="${p.href}" target="_blank"><img loading="lazy" src="${p.href}" alt="${p.label}"></a>`;
function visible(){return PANELS.filter(p=>(!state.group||p.group===state.group)&&
  (!state.query||`${p.label} ${p.description} ${p.key}`.toLowerCase().includes(state.query)));}
function ctrl(){const c=document.getElementById("controls");c.innerHTML="";
  const pg=document.createElement("div");pg.className="grp";pg.innerHTML="<b>panel group</b>";
  const mk=(value,label)=>{const b=document.createElement("button");b.textContent=label;
    b.className=state.group===value?"on":"";b.onclick=()=>{state.group=value;draw();};pg.append(b);};
  mk(null,"all");Object.entries(GROUPS).forEach(([key,label])=>{
    if(PANELS.some(p=>p.group===key))mk(key,label);});c.append(pg);
  const vg=document.createElement("div");vg.className="grp";vg.innerHTML="<b>view</b>";
  [["stack","full panels"],["sheet","contact sheet"]].forEach(([value,label])=>{
    const b=document.createElement("button");b.textContent=label;b.className=state.view===value?"on":"";
    b.onclick=()=>{state.view=value;draw();};vg.append(b);});c.append(vg);
  const sg=document.createElement("div");sg.className="grp";sg.innerHTML="<b>search</b>";
  const q=document.createElement("input");q.type="search";q.placeholder="panel name, path, or description";
  q.value=state.query;q.oninput=()=>{state.query=q.value.trim().toLowerCase();render();};sg.append(q);c.append(sg);}
function render(){const ps=visible();document.getElementById("count").textContent=
  `${ps.length} of ${PANELS.length} generated panels`;
  const p=document.getElementById("panels");if(!ps.length){p.innerHTML="<p class='missing'>No panels match.</p>";return;}
  if(state.view==="sheet")p.innerHTML=`<div class="sheet">${ps.map(x=>
    `<figure><figcaption>${x.label} · ${x.key}</figcaption>${media(x)}</figure>`).join("")}</div>`;
  else p.innerHTML=ps.map(x=>`<figure><h2>${x.group_label}</h2><figcaption><b>${x.label} —
    <a href="${x.href}" target="_blank">${x.key}</a></b><span>${x.description}</span></figcaption>${media(x)}</figure>`).join("");}
function draw(){ctrl();render();}draw();
</script></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plots", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()

    output = (args.out or args.plots / "index.html").resolve()
    plot_root = args.plots.resolve()
    run = args.run.resolve()
    files = [
        path
        for path in plot_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and path.resolve() != output
        and not any(part.startswith(".") for part in path.relative_to(plot_root).parts)
    ]
    panels = [panel_info(path, plot_root, output) for path in files]
    group_order = {name: index for index, name in enumerate(GROUPS)}
    panels.sort(key=lambda panel: (group_order[panel["group"]], panel["key"]))

    summary = load_json(run / "summary.json")
    metadata = load_json(run / "config.json") or load_json(run / "metadata.json")
    title = args.title or f"{run.name} residual pursuit browser"
    details = "".join(
        f"<div>{html.escape(str(label))}<b>{html.escape(str(value))}</b></div>"
        for label, value in detail_values(summary, metadata)
    )
    unavailable = "".join(
        f"<li><b>{html.escape(name)}</b>: {html.escape(reason)}</li>"
        for name, reason in UNAVAILABLE_STANDARD_PANELS.items()
    )
    metadata_text = html.escape(json.dumps(metadata, indent=2, sort_keys=True))
    document = (
        HTML.replace("__TITLE__", html.escape(title))
        .replace("__RUN_NAME__", html.escape(run.name))
        .replace("__DETAILS__", details)
        .replace("__MODEL__", html.escape(str(metadata.get("model", "—"))))
        .replace("__DETECTOR__", html.escape(str(metadata.get("detector", "—"))))
        .replace("__UNAVAILABLE__", unavailable)
        .replace("__METADATA__", metadata_text)
        .replace("__PANELS__", json.dumps(panels).replace("</", "<\\/"))
        .replace("__GROUPS__", json.dumps(GROUPS))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    registered = sum(panel["key"] in PANEL_REGISTRY for panel in panels)
    print(
        f"wrote {output} ({len(panels)} panels, {registered} registered)",
        flush=True,
    )
    try:
        import build_out_index

        build_out_index.build()
    except Exception as exc:
        print(f"hub index rebuild skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
