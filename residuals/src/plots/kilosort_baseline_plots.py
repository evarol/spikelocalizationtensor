"""Kilosort baseline plot suite for the iblsorter run on dataset1_p1.

Reads the ALF/pykilosort outputs of residuals/runs/dataset1_p1/kilosort_iblsorter/,
renders the full diagnostic panel stack at dpi=800, and writes a self-contained
SpikeTensor-style browser (index.html) with kilosort-accurate metadata so the
residuals/out hub lists it as a gallery. Includes an honest distribution-level
cross-comparison against the 0019 all-channel accepted events.

Usage:
    singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
        /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
        /bin/bash -c "source /ext3/env.sh && python residuals/src/plots/kilosort_baseline_plots.py"
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[3]
FS = 30000.0
DPI = 800

PANELS = {
    "raster_depth_time.png": (
        "sorting overview",
        "depth × time raster",
        "Every detected spike binned at 2 s × 20 µm (log count). "
        "The four zigzag column families of the NP1.4 probe appear as depth bands.",
    ),
    "raster_windows.png": (
        "sorting overview",
        "raster windows",
        "Four 20 s windows across the recording at spike resolution, colored by cluster identity.",
    ),
    "firing_rates.png": (
        "units",
        "firing rates",
        "Per-cluster mean firing rate: distribution and depth dependence, split by KSLabel.",
    ),
    "cluster_depths_labels.png": (
        "units",
        "cluster depths and waveform widths",
        "Cluster depth distribution and trough-to-peak waveform width by KSLabel.",
    ),
    "templates_gallery.png": (
        "units",
        "learned template shapes",
        "The 12 highest-count good units: kilosort's learned template on its four "
        "largest-amplitude footprint channels, time on x, peak-normalized (kilosort "
        "learns templates on destriped+whitened traces, so shape is comparable but the "
        "y-scale is not volts).",
    ),
    "template_variability.png": (
        "units",
        "raw waveforms vs mean template",
        "The 12 highest-count good units: up to 200 raw detected waveforms per unit, "
        "extracted from the raw AP bin on the template peak channel (thin lines), their "
        "empirical mean (thick, same color), and kilosort's learned template for that "
        "channel (dashed black, peak-normalized). Raw traces are gain-corrected to µV; "
        "the template shape comes from destriped+whitened data.",
    ),
    "drift_motion.png": (
        "stability",
        "estimated motion",
        "The 9 depth-block rigid drift estimates from iblsorter's datashift stage, in µm over time.",
    ),
    "amplitudes.png": (
        "stability",
        "amplitudes",
        "Spike amplitudes in µV: distribution, amplitude drift over time, and depth dependence.",
    ),
    "quality_summary.png": (
        "quality",
        "KSLabel and contamination",
        "Cluster counts per KSLabel and the contamination-percent distribution from kilosort's "
        "amplitude-cutoff QC.",
    ),
    "cross_comparison_0019.png": (
        "comparison",
        "kilosort vs 0019 accepted events",
        "Distribution-level comparison against the 0019 all-channel 20%-bar run: spike rate over "
        "time and depth density. Different objectives (template matching after destriping/whitening "
        "versus raw-residual peeling), so counts are not one-to-one.",
    ),
}

GROUP_ORDER = ["sorting overview", "units", "stability", "quality", "comparison"]

CSS = """
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
        background:var(--card);max-width:1180px}.detail h2{margin:0 0 7px;font-size:15px}
.detail p{margin:6px 0}.detail-grid{display:grid;
        grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:7px 15px}
.detail-grid div{color:var(--dim)}.detail-grid b{display:block;color:var(--fg);font-size:12px}
.badge{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:700;font-size:11px;
       border:1px solid var(--line);margin-right:6px}
.badge.ref{background:rgba(47,158,68,.18);color:#69db7c}
.badge.diag{background:rgba(132,94,247,.18);color:#b197fc}
.explore-note{margin-top:8px;padding:8px 10px;border-left:3px solid #845ef7;
              background:rgba(132,94,247,.09);max-width:1150px}
.panels h2{font-size:13px;margin:16px 0 5px;color:var(--dim);font-weight:600}
.panels figure{margin:0 0 22px}.panels figcaption{margin:0 0 5px}
.panels figcaption b{display:block;font-size:13px}.panels figcaption span{color:var(--dim)}
.panels img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff;display:block}
.panels a{color:var(--acc);text-decoration:none}
.sheet{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
.sheet figure{margin:0}.sheet figcaption{font-size:11px;color:var(--dim);padding:3px 0}
pre{max-height:420px;overflow:auto;font-size:11px}details summary{cursor:pointer;color:var(--dim);font-size:11px}
@media(max-width:540px){.sheet{grid-template-columns:1fr}.controls,header,main{padding-left:10px;padding-right:10px}}
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
<header><h1>__TITLE__</h1><div class="note">
A standard-sorter baseline for dataset1_p1: <b>pykilosort 2.5</b> as packaged by the IBL
(<code>ibl-sorter 1.13.0</code>), run with stock IBL parameters on the full 1957 s AP recording —
IBL destriping, local whitening, rigid-per-block drift registration, then Kilosort 2.5
clustering, merges, splits, and cutoff. This page deliberately mirrors the residual-pursuit
browsers' typography, controls, detail card, and full-panel stack.
<div class="explore-note"><b>Comparability caveat.</b> Kilosort template-matches the
destriped+whitened data and keeps everything above its projection thresholds, while the 0019
pursuit accepts an event only if its single-source residual fit clears a 20% channel-fraction
bar. The comparison panel therefore compares distributions (rate over time, depth density),
never one-to-one matches.</div>
</div></header>
<div class="controls" id="controls"></div><div class="count" id="count"></div>
<main><section class="detail"><h2>__RUN_NAME__</h2>
<p><span class="badge ref">completed full recording</span><span class="badge diag">standard sorter baseline</span></p>
<div class="detail-grid">__DETAILS__</div>
<p><b>Pipeline:</b> __PIPELINE__</p>
<details><summary>saved run metadata</summary><pre>__METADATA__</pre></details>
</section><div class="panels" id="panels"></div></main>
<script>
const PANELS=__PANELS__, GROUPS=__GROUPS__;
const state={group:null,view:"stack",query:""};
const media=p=>`<a href="${p.href}" target="_blank"><img loading="lazy" src="${p.href}" alt="${p.label}"></a>`;
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
    b.onclick=()=>{state.view=value;draw();};vg.append(b);});
  const sg=document.createElement("div");sg.className="grp";sg.innerHTML="<b>search</b>";
  const q=document.createElement("input");q.type="search";q.placeholder="panel name or description";
  q.value=state.query;q.oninput=()=>{state.query=q.value.trim().toLowerCase();render();};sg.append(q);c.append(sg);}
function render(){const ps=visible();document.getElementById("count").textContent=
  `${ps.length} of ${PANELS.length} generated panels`;
  const p=document.getElementById("panels");if(!ps.length){p.innerHTML="<p>No panels match.</p>";return;}
  if(state.view==="sheet")p.innerHTML=`<div class="sheet">${ps.map(x=>
    `<figure><figcaption>${x.label} · ${x.key}</figcaption>${media(x)}</figure>`).join("")}</div>`;
  else p.innerHTML=ps.map(x=>`<figure><h2>${x.group}</h2><figcaption><b>${x.label} —
    <a href="${x.href}" target="_blank">${x.key}</a></b><span>${x.description}</span></figcaption>${media(x)}</figure>`).join("");}
function draw(){ctrl();render();}draw();
</script></html>
"""


def load_tsv(path):
    if not path.exists():
        return {}
    with path.open() as fid:
        reader = csv.DictReader(fid, delimiter="\t")
        return {int(row["cluster_id"]): row for row in reader}


def style_ax(ax):
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def plot_raster_depth_time(alf, out, duration):
    times = np.load(alf / "spikes.times.npy", mmap_mode="r")
    depths = np.load(alf / "spikes.depths.npy", mmap_mode="r")
    t = times[:]
    d = depths[:]
    finite = np.isfinite(d)
    t_bins = np.arange(0, np.ceil(duration / 2) * 2 + 2, 2)
    d_bins = np.arange(0, 3841, 20)
    H, _, _ = np.histogram2d(t[finite], d[finite], bins=[t_bins, d_bins])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(
        H.T,
        aspect="auto",
        origin="lower",
        extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]],
        norm=LogNorm(vmin=1, vmax=H.max()),
        cmap="viridis",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="spikes / bin")
    ax.set(xlabel="time (s)", ylabel="depth (µm)", title="kilosort spikes: full recording")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_raster_windows(alf, out, duration):
    times = np.load(alf / "spikes.times.npy", mmap_mode="r")
    depths = np.load(alf / "spikes.depths.npy", mmap_mode="r")
    clusters = np.load(alf / "spikes.clusters.npy", mmap_mode="r")
    t = times[:]
    d = depths[:]
    c = clusters[:]
    finite = np.isfinite(d)
    starts = [0.0, duration * 0.35, duration * 0.6, duration * 0.85]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=False)
    for ax, start in zip(axes, starts):
        m = finite & (t >= start) & (t < start + 20)
        idx = np.flatnonzero(m)
        if idx.size > 120000:
            idx = np.random.default_rng(0).choice(idx, 120000, replace=False)
        sc = ax.scatter(t[idx], d[idx], c=c[idx], s=0.2, alpha=0.35, cmap="turbo", linewidths=0)
        ax.set(ylabel="depth (µm)")
        style_ax(ax)
    axes[0].set_title("20 s spike rasters at four recording epochs (≤120k points each)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def label_of(ks, cid):
    return ks.get(cid, {}).get("KSLabel", "mua")


def plot_firing_rates(alf, out, duration, ks):
    clusters = np.load(alf / "spikes.clusters.npy", mmap_mode="r")
    cdepths = np.load(alf / "clusters.depths.npy")
    counts = np.bincount(clusters[:], minlength=cdepths.size)
    rates = counts / duration
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, color in [("good", "#2f9e44"), ("mua", "#5b6472")]:
        idx = [i for i in range(cdepths.size) if label_of(ks, i) == label]
        axes[0].hist(rates[idx], bins=np.logspace(-3, 2, 40), color=color, alpha=0.65, label=label)
        axes[1].scatter(
            cdepths[idx],
            rates[idx],
            s=7,
            alpha=0.7,
            color=color,
            label=label,
            linewidths=0,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="mean firing rate (Hz)", ylabel="clusters")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="cluster depth (µm)", ylabel="mean firing rate (Hz)")
    for ax in axes:
        ax.legend(fontsize=8)
        style_ax(ax)
    fig.suptitle("per-cluster firing rate by KSLabel", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_cluster_depths_labels(alf, out, ks):
    cdepths = np.load(alf / "clusters.depths.npy")
    ptt_ms = np.abs(np.load(alf / "clusters.peakToTrough.npy")) * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins_d = np.arange(0, 3841, 40)
    for label, color in [("good", "#2f9e44"), ("mua", "#5b6472")]:
        idx = [i for i in range(cdepths.size) if label_of(ks, i) == label]
        axes[0].hist(
            cdepths[idx],
            bins=bins_d,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
        vals = np.abs(ptt_ms[idx])
        axes[1].hist(
            vals[np.isfinite(vals)],
            bins=np.linspace(0, 2, 40),
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
    axes[0].set(xlabel="cluster depth (µm)", ylabel="clusters")
    axes[1].set(xlabel="trough-to-peak width (ms)", ylabel="clusters")
    for ax in axes:
        ax.legend(fontsize=8)
        style_ax(ax)
    fig.suptitle("cluster depth and waveform width by KSLabel", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def pick_top_good(alf, ks, count=12):
    templates = np.load(alf / "templates.waveforms.npy")
    tchan = np.load(alf / "templates.waveformsChannels.npy")
    clusters = np.load(alf / "spikes.clusters.npy", mmap_mode="r")
    cdepths = np.load(alf / "clusters.depths.npy")
    counts = np.bincount(clusters[:], minlength=cdepths.size)
    good = [i for i in range(cdepths.size) if ks.get(i, {}).get("KSLabel") == "good"]
    top = sorted(good, key=lambda i: -counts[i])[:count]
    return templates, tchan, counts, top


def plot_templates_gallery(alf, out):
    templates, tchan, counts, top = pick_top_good(alf, load_tsv(alf / "cluster_KSLabel.tsv"))
    n_t = templates.shape[1]
    t_ms = (np.arange(n_t) - n_t / 2) / FS * 1e3
    fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    for ax, cid in zip(axes.ravel(), top):
        wf = templates[cid]
        peak = np.argmax(np.abs(wf).max(axis=0))
        trace = wf[:, peak]
        trace = trace / max(1e-9, np.abs(trace).max())
        ax.plot(t_ms, trace, color="#14171c", linewidth=1.4)
        ax.axhline(0, color="#9aa3b2", linewidth=0.5)
        ax.axvline(0, color="#9aa3b2", linewidth=0.5, linestyle=":")
        ax.set_title(f"unit {cid} · {counts[cid]:,} spikes", fontsize=9)
        style_ax(ax)
    fig.suptitle("learned templates: 12 highest-count good units (peak channel, peak-normalized)")
    fig.supxlabel("time from peak (ms)")
    fig.supylabel("normalized amplitude")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def extract_raw_waveforms(bin_file, samples, raw_chan, n_channels, max_wf=82, trough_offset=42):
    import spikeglx

    md = spikeglx.read_meta_data(bin_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)["ap"][0] * 1e6
    wf = np.zeros((len(samples), max_wf), dtype=np.float32)
    with open(bin_file, "rb") as fid:
        for i, s in enumerate(samples):
            start = int(s) - trough_offset
            fid.seek(start * n_channels * 2)
            chunk = np.frombuffer(fid.read(max_wf * n_channels * 2), dtype=np.int16)
            wf[i] = chunk.reshape(max_wf, n_channels)[:, raw_chan].astype(np.float32) * s2v
    return wf


def plot_template_variability(run, out, max_wf_per_unit=200, seed=0):
    alf = run / "alf"
    bin_file = Path("/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin")
    templates, tchan, counts, top = pick_top_good(alf, load_tsv(alf / "cluster_KSLabel.tsv"))
    spike_clusters = np.load(alf / "spikes.clusters.npy", mmap_mode="r")
    spike_samples = np.load(alf / "spikes.samples.npy", mmap_mode="r")
    raw_ind = np.load(alf / "channels.rawInd.npy")
    cdepths = np.load(alf / "clusters.depths.npy")
    import spikeglx

    ns = spikeglx.Reader(bin_file).ns
    n_channels = 385
    n_t = templates.shape[1]
    t_ms = (np.arange(n_t) - n_t / 2) / FS * 1e3
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(top)))
    fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    rng = np.random.default_rng(seed)
    for ax, cid, color in zip(axes.ravel(), top, colors):
        idx = np.flatnonzero(spike_clusters[:] == cid)
        if idx.size > max_wf_per_unit:
            idx = rng.choice(idx, max_wf_per_unit, replace=False)
        peak_slot = int(np.argmax(np.abs(templates[cid]).max(axis=0)))
        peak_chan = int(tchan[cid][peak_slot])
        raw_chan = int(raw_ind[peak_chan])
        samples = spike_samples[idx]
        keep = (samples >= n_t) & (samples < ns - n_t)
        samples = samples[keep]
        raw = extract_raw_waveforms(bin_file, samples, raw_chan, n_channels)
        med = np.median(raw) if raw.size else 0.0
        raw = raw - med
        for w in raw:
            ax.plot(t_ms, w, color=color, alpha=0.15, linewidth=0.4)
        if raw.size:
            ax.plot(t_ms, raw.mean(axis=0), color=color, linewidth=2.2, label="raw mean")
        tmpl = templates[cid][:, peak_slot]
        tmpl = tmpl / max(1e-9, np.abs(tmpl).max()) * np.abs(raw.mean()).max() if raw.size else tmpl
        ax.plot(t_ms, tmpl, color="#14171c", linewidth=1.4, linestyle="--", label="learned template")
        ax.axhline(0, color="#9aa3b2", linewidth=0.5)
        ax.set_title(f"unit {cid} · {counts[cid]:,} spikes · depth {cdepths[cid]:.0f} µm", fontsize=9)
        style_ax(ax)
    handles = [
        plt.Line2D([], [], color="#4c8dff", alpha=0.5, linewidth=1.0),
        plt.Line2D([], [], color="#4c8dff", linewidth=2.2),
        plt.Line2D([], [], color="#14171c", linewidth=1.4, linestyle="--"),
    ]
    fig.legend(handles, ["raw waveforms", "raw mean", "learned template"], loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("raw detected waveforms vs mean vs learned template (peak channel, µV)")
    fig.supxlabel("time from aligned sample (ms)")
    fig.supylabel("amplitude (µV)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_drift_motion(alf, out):
    drift_t = np.load(alf / "drift.times.npy")
    drift_um = np.load(alf / "drift.um.npy")
    drift_depths = np.load(alf / "drift_depths.um.npy").ravel()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k in range(drift_um.shape[1]):
        ax.plot(
            drift_t,
            drift_um[:, k],
            linewidth=0.9,
            label=f"block {k} (center {drift_depths[k]:.0f} µm)",
        )
    ax.set(xlabel="time (s)", ylabel="estimated motion (µm)")
    ax.legend(fontsize=7, ncol=3)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_amplitudes(alf, out):
    times = np.load(alf / "spikes.times.npy", mmap_mode="r")
    depths = np.load(alf / "spikes.depths.npy", mmap_mode="r")
    amps = np.load(alf / "spikes.amps.npy", mmap_mode="r") * 1e6
    t = times[:]
    d = depths[:]
    a = amps[:]
    finite = np.isfinite(d)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(np.log10(a), bins=60, color="#4c8dff")
    axes[0].set(xlabel="amplitude (µV, log10)", ylabel="spikes")
    t_bins = np.arange(0, 1958, 16)
    a_bins = np.logspace(np.log10(a.min()), np.log10(a.max()), 50)
    H, _, _ = np.histogram2d(t[finite], a[finite], bins=[t_bins, a_bins])
    im = axes[1].imshow(
        H.T,
        aspect="auto",
        origin="lower",
        extent=[t_bins[0], t_bins[-1], a_bins[0], a_bins[-1]],
        norm=LogNorm(vmin=1, vmax=H.max()),
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_yscale("log")
    fig.colorbar(im, ax=axes[1], label="spikes / bin")
    axes[1].set(xlabel="time (s)", ylabel="amplitude (µV)")
    d_bins = np.arange(0, 3841, 20)
    H, _, _ = np.histogram2d(d[finite], a[finite], bins=[d_bins, a_bins])
    im = axes[2].imshow(
        H.T,
        aspect="auto",
        origin="lower",
        extent=[d_bins[0], d_bins[-1], a_bins[0], a_bins[-1]],
        norm=LogNorm(vmin=1, vmax=H.max()),
        cmap="viridis",
        interpolation="nearest",
    )
    axes[2].set_yscale("log")
    fig.colorbar(im, ax=axes[2], label="spikes / bin")
    axes[2].set(xlabel="depth (µm)", ylabel="amplitude (µV)")
    for ax in axes:
        style_ax(ax)
    fig.suptitle("spike amplitudes (ALF conversion, µV)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_quality_summary(alf, ks_dir, out):
    ks = load_tsv(alf / "cluster_KSLabel.tsv")
    contam = load_tsv(ks_dir / "cluster_ContamPct.tsv")
    labels = [ks.get(i, {}).get("KSLabel", "mua") for i in sorted(ks)]
    names, counts = np.unique(labels, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, counts, color=["#2f9e44" if n == "good" else "#5b6472" for n in names])
    for i, n in enumerate(counts):
        axes[0].text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set(xlabel="KSLabel", ylabel="clusters")
    for label, color in [("good", "#2f9e44"), ("mua", "#5b6472")]:
        vals = []
        for cid, row in contam.items():
            if ks.get(cid, {}).get("KSLabel") != label:
                continue
            try:
                v = float(row["ContamPct"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        axes[1].hist(vals, bins=np.linspace(0, 100, 41), color=color, alpha=0.65, label=label)
    axes[1].set(xlabel="ContamPct (%)", ylabel="clusters")
    axes[1].legend(fontsize=8)
    for ax in axes:
        style_ax(ax)
    fig.suptitle("kilosort cluster quality", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_cross_comparison(run, comp_run, out):
    alf = run / "alf"
    fs = FS
    t = np.load(alf / "spikes.times.npy", mmap_mode="r")
    d = np.load(alf / "spikes.depths.npy", mmap_mode="r")
    kt = t[:]
    kd = d[:]
    kfinite = np.isfinite(kd)
    ct = np.load(comp_run / "spike_times.npy") / fs
    cy = np.load(comp_run / "centroids.npy")[:, 1]
    rt = np.load(comp_run / "rejected_spike_times.npy") / fs
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    bins_t = np.arange(0, 1960, 50)
    for arr, label, color in [
        (kt, "kilosort 2.5 spikes", "#4c8dff"),
        (ct, "0019 accepted (20% bar)", "#2f9e44"),
        (rt, "0019 rejected proposals", "#c92a2a"),
    ]:
        axes[0].hist(arr, bins=bins_t, histtype="step", linewidth=1.5, label=label, color=color)
    axes[0].set(xlabel="time (s)", ylabel="spikes / 50 s")
    axes[0].legend(fontsize=8)
    bins_d = np.arange(0, 3841, 20)
    for arr, label, color in [
        (kd[kfinite], "kilosort 2.5 spikes", "#4c8dff"),
        (cy, "0019 accepted centroids", "#2f9e44"),
    ]:
        axes[1].hist(
            arr,
            bins=bins_d,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=color,
        )
    axes[1].set(xlabel="depth (µm)", ylabel="normalized density")
    axes[1].legend(fontsize=8)
    for ax in axes:
        style_ax(ax)
    fig.suptitle(
        "kilosort baseline vs 0019 all-channel pursuit (20% bar): "
        f"{len(kt):,} kilosort spikes · {len(ct):,} accepted · "
        f"{len(rt):,} rejected proposals",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def build_index(out, run, summary, comp_summary, job_id, ks):
    files = sorted(p.name for p in out.glob("*.png"))
    panels = [
        {
            "key": name,
            "group": PANELS[name][0],
            "label": PANELS[name][1],
            "description": PANELS[name][2],
            "href": name,
        }
        for name in files
        if name in PANELS
    ]
    groups = {g: g for g in GROUP_ORDER}
    label_counts = {}
    for row in ks.values():
        label_counts[row.get("KSLabel", "mua")] = label_counts.get(row.get("KSLabel", "mua"), 0) + 1
    details = [
        ("sorter", "pykilosort 2.5 (ibl-sorter 1.13.0, stock IBL parameters)"),
        ("recording", "dataset1_p1 p1_g0_t0.imec0.ap.bin (1957 s, 384 AP + sync)"),
        ("spikes", f"{summary.get('n_spikes', '—'):,}"),
        ("clusters", f"{summary.get('n_clusters', '—'):,}"),
        (
            "good / mua",
            f"{label_counts.get('good', 0):,} / {label_counts.get('mua', 0):,} (KSLabel)",
        ),
        ("mean firing rate", f"{summary.get('mean_firing_rate_hz', 0):,.0f} Hz probe-wide"),
        ("iblsorter job", job_id or "—"),
        (
            "0019 comparison run",
            f"{comp_summary.get('n_events', '—'):,} accepted at the 20% bar",
        ),
    ]
    details_html = "".join(
        f"<div>{html.escape(str(k))}<b>{html.escape(str(v))}</b></div>" for k, v in details
    )
    metadata = {"kilosort_summary": summary, "0019_summary": comp_summary}
    doc = (
        INDEX_HTML.replace("__TITLE__", "kilosort baseline — pykilosort 2.5 on dataset1_p1")
        .replace("__RUN_NAME__", run.name)
        .replace("__DETAILS__", details_html)
        .replace(
            "__PIPELINE__",
            "IBL destriping → whitening → rigid 9-block drift registration → "
            "Kilosort 2.5 (learn, merge, split, cutoff) → phy + ALF conversion",
        )
        .replace("__METADATA__", html.escape(json.dumps(metadata, indent=2)))
        .replace("__PANELS__", json.dumps(panels).replace("</", "<\\/"))
        .replace("__GROUPS__", json.dumps(groups))
        .replace("__CSS__", CSS)
    )
    (out / "index.html").write_text(doc)
    print(f"wrote {out / 'index.html'} ({len(panels)} panels)", flush=True)
    try:
        import build_out_index

        build_out_index.build()
    except Exception as exc:
        print(f"hub index rebuild skipped: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=REPO / "residuals/runs/dataset1_p1/kilosort_iblsorter",
    )
    parser.add_argument(
        "--comparison-run",
        type=Path,
        default=REPO
        / "residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8",
    )
    parser.add_argument("--out", type=Path, default=REPO / "residuals/out/0022_kilosort_baseline")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    alf = args.run / "alf"
    ks_dir = args.run / "iblsorter"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "kilosort_summary.json").read_text())
    comp_summary = json.loads((args.comparison_run / "summary.json").read_text())
    duration = summary["duration_s"]
    ks = load_tsv(alf / "cluster_KSLabel.tsv")

    jobs = {
        "raster_depth_time.png": lambda out: plot_raster_depth_time(alf, out, duration),
        "raster_windows.png": lambda out: plot_raster_windows(alf, out, duration),
        "firing_rates.png": lambda out: plot_firing_rates(alf, out, duration, ks),
        "cluster_depths_labels.png": lambda out: plot_cluster_depths_labels(alf, out, ks),
        "templates_gallery.png": lambda out: plot_templates_gallery(alf, out),
        "template_variability.png": lambda out: plot_template_variability(args.run, out),
        "drift_motion.png": lambda out: plot_drift_motion(alf, out),
        "amplitudes.png": lambda out: plot_amplitudes(alf, out),
        "quality_summary.png": lambda out: plot_quality_summary(alf, ks_dir, out),
        "cross_comparison_0019.png": lambda out: plot_cross_comparison(
            args.run, args.comparison_run, out
        ),
    }
    for name, job in jobs.items():
        out = args.out / name
        if out.exists() and not args.overwrite:
            print(f"skip {name} (exists)", flush=True)
            continue
        job(out)
        print(f"wrote {out}", flush=True)

    build_index(args.out, args.run, summary, comp_summary, args.job_id, ks)


if __name__ == "__main__":
    main()
