"""Browsable index of every factorization fit.

Same idea as the ZNCC sweep browser: filter by hyperparameter, sort by metric, click a
row (or a dot) to open that fit's panels. Two differences that matter here.

The SCATTER defaults to the frontier that actually matters for this fork -- reconstruction
against localization slope -- rather than to the two loss terms, because reconstruction
alone does not distinguish a good codebook from a good basis. Every fit so far
reconstructs well; they differ almost entirely in whether the implied position moves.

SLOPE, not correlation, is the localization axis. A heavily shrunk position estimate stays
correlated with the monopole (r ~ +0.5) while barely moving, so correlation reads as
success where slope reads as failure. Both are in the table; the plot uses slope.

Metrics come from summary_*.json (fit-side) joined to frontier.json (localization,
computed once by frontier.py against the monopole on a common 60k spike sample).

Usage:
    python3 zncc/tensor/frontier.py          # writes frontier.json
    python3 zncc/tensor/browser.py
    open zncc/figures/tensor/index.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PANELS = [("It_zoom", "It_zoom.mp4",
           "movie: localizations binned at 1 s, zoomed to 400–900 µm"),
          ("It_full", "It_full.mp4", "movie: the same over the full probe"),
          ("convergence", "convergence.png",
           "loss convergence: reconstruction nMSE vs step and wall-clock"),
          ("spikes", "spikes.png", "example spikes: reconstruction, per-spike heat map, π"),
          ("aggregate_1s_zoom", "aggregate_1s_zoom.png", "1 s aggregate — 400–900 µm zoom"),
          ("aggregate_1s", "aggregate_1s.png", "1 s aggregate — full probe"),
          ("localize", "localize.png", "soft and hard readouts vs monopole"),
          ("components", "components.png", "the codebook: footprints and time courses"),
          ("basis", "basis.png",
           "the time basis per component, sorted by usage, with space coloured by which "
           "component each spike leans on"),
          ("centroid_basis_zoom", "centroid_basis_zoom.png",
           "all spike centroids, 400–900 µm zoom; jittered, coloured by RGB of the "
           "3-component PCA of v_s (shape-preserving colours)"),
          ("centroid_basis_full", "centroid_basis_full.png",
           "all spike centroids over the full probe; jittered, coloured by RGB of the "
           "3-component PCA of v_s"),
          ("centroid_basis_movie_zoom", "centroid_basis_movie_zoom.mp4",
           "one-second centroid scatter movie, 400–900 µm zoom; colour = RGB of the "
           "v_s PCA"),
          ("centroid_basis_movie_full", "centroid_basis_movie_full.mp4",
           "one-second centroid scatter movie over the full probe; colour = RGB of "
           "the v_s PCA"),
          ("depth_time_density_zoom", "depth_time_density_zoom.png",
           "depth × time raster, 400–900 µm; amplitude-weighted magma density"),
          ("depth_time_density_full", "depth_time_density_full.png",
           "depth × time raster over the full probe; amplitude-weighted magma density"),
          ("depth_time_basis_zoom", "depth_time_basis_zoom.png",
           "depth × time centroid scatter, 400–900 µm; jittered, coloured by RGB of "
           "the v_s PCA"),
          ("depth_time_basis_full", "depth_time_basis_full.png",
           "depth × time centroid scatter over the full probe; jittered, coloured by "
           "RGB of the v_s PCA"),
          ("depth_time_mse_zoom", "depth_time_mse_zoom.png",
           "depth × time raster, 400–900 µm; colour = mean per-spike reconstruction "
           "error (unexplained energy fraction) — shows where the model fits badly"),
          ("depth_time_mse_full", "depth_time_mse_full.png",
           "depth × time raster over the full probe; colour = mean per-spike "
           "reconstruction error"),
          ("basis_error", "basis_error.png",
           "spatial basis locations coloured by mean reconstruction error, plus error "
           "vs depth — is the error biased by WHERE the spike is localized?"),
          ("embed_umap", "embed_umap.png",
           "UMAP of the per-spike coefficients v_s, coloured by centroid x / y / z and "
           "by the 3-PC PCA RGB shape colour"),
          ("embed_pca", "embed_pca.png",
           "PCA of v_s — the linear, parameter-free baseline for the same view"),
          ("embed_tsne", "embed_tsne.png", "t-SNE of v_s, if it was computed"),
          ("dredge_real", "dredge_real.png",
           "canonical spikeinterface dredge_ap on this fit's localizations: rigid + "
           "nonrigid traces vs imposed motion, and corrected depth rasters"),
          ("usage", "usage.png", "π support and concentration")]

# Optional model-family panels.  Ordinary lattice rows simply leave these blank;
# the overall-extension registry below supplies them for multipole and noise-model
# examples without copying or rewriting either study's source artifacts.
PANELS += [
    ("spike_decomposition", "spike_decomposition.png",
     "multipole: observed waveform, each active source term, total, and residual"),
    ("source_cloud", "source_cloud.png",
     "multipole: every active source, colored by its fractional contribution q"),
    ("multipole_diagnostics", "multipole_diagnostics.png",
     "multipole: support, contribution, separation, conditioning, and source sensitivity"),
    ("readout_sensitivity", "readout_sensitivity.png",
     "multipole: all-source, dominant-source, and contribution-weighted centroid views"),
    ("spatial_temporal_shapes", "spatial_temporal_shapes.png",
     "multipole: observed/reconstructed spike, sub-source waveforms, and learned spatial footprint in x-y and z-y"),
]

# Every depth-dependent panel again, rendered after subtracting the fit's OWN DREDge
# motion. These are the test of the pipeline: an uncorrected raster shows the imposed
# sawtooth, a correctly corrected one is flat, and whatever is left is motion the model
# could not see. Soft uses the softmax-over-lags solve, hard the argmax one.
_CORRECTED = [("centroid_basis_zoom", "centroids, 400–900 µm zoom"),
              ("centroid_basis_full", "centroids, full probe"),
              ("depth_time_density_zoom", "depth × time density, zoom"),
              ("depth_time_density_full", "depth × time density, full probe"),
              ("depth_time_basis_zoom", "depth × time by basis, zoom"),
              ("depth_time_basis_full", "depth × time by basis, full probe"),
              ("depth_time_mse_zoom", "depth × time reconstruction error, zoom"),
              ("depth_time_mse_full", "depth × time reconstruction error, full probe"),
              ("aggregate_1s_zoom", "1 s aggregate, zoom"),
              ("aggregate_1s", "1 s aggregate, full probe")]
_CORRECTED_MOV = [("It_zoom", "I_t movie, zoom"), ("It_full", "I_t movie, full probe"),
                  ("centroid_basis_movie_zoom", "centroid movie, zoom"),
                  ("centroid_basis_movie_full", "centroid movie, full probe")]
# Simplified per user request: of all DREDge results only the CANONICAL (real) ones
# are shown. The internal soft/hard corrected panels and the internal dc figure remain
# on disk but are no longer registered.
for _w, _lab in (("drr", "REAL DREDge rigid"), ("drn", "REAL DREDge nonrigid")):
    PANELS += [(f"{k}_{_w}", f"{k}_{_w}.png", f"{_lab} DREDge-corrected — {d}")
               for k, d in _CORRECTED]
    PANELS += [(f"{k}_{_w}", f"{k}_{_w}.mp4", f"{_lab} DREDge-corrected — {d}")
               for k, d in _CORRECTED_MOV]

REF = {"free_rank1": 0.1029, "per_slot_mean": 0.3189, "mono_spread_y": 15.4}


def _real_cols(d: Path):
    """Canonical-DREDge scores for a row, if dredge_real has been run on it."""
    f = d / "dredge_real.npz"
    if not f.exists():
        return {"real_r": None, "real_gain": None, "real_r_nr": None}
    import numpy as _np
    z = _np.load(f)
    # Keep enough precision for deterministic ranking: the two leading Gaussian
    # fits both display as 0.935 but differ in the fourth decimal place.
    return {"real_r": round(float(z["gt_r_rigid"]), 6),
            "real_gain": round(float(z["gt_gain_rigid"]), 6),
            "real_r_nr": round(float(z["gt_r_nonrigid_best"]), 6)}


def collect(runs: Path, figs: Path):
    front = {}
    f = figs / "frontier.json"
    if f.exists():
        front = {r["tag"]: r for r in json.loads(f.read_text())}
    # mean pairwise ZNCC of each fit's own I_t (not computed by this package).
    # Held out: no fit here
    # ever optimised C, so it can be plotted against nMSE without the two being coupled
    # by construction the way they were in the 140 trained ZNCC runs.
    # from C_table.json, NOT dc_all.jsonl: that file carries the same exclusion policy the
    # markdown table and the figure use, so all three agree on which value is shown for the
    # fits whose sources fell off the grid.
    dc = {}
    fd = figs / "C_table.json"
    if fd.exists():
        dc = {r["tag"]: r for r in json.loads(fd.read_text())}
    ctl = {}
    fc = figs / "dc_controls.jsonl"
    if fc.exists():
        for line in fc.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ctl[r["tag"]] = r
    rows = []
    for sf in sorted(runs.glob("summary_*.json")):
        tag = sf.stem[len("summary_"):]
        s = json.loads(sf.read_text())
        ar = s.get("args", {})
        fr = front.get(tag, {})
        dr = dc.get(tag, {})
        d = figs / tag
        ml = re.search(r"_l1([0-9.]+)", tag)
        me = re.search(r"_ent([0-9.]+)", tag)
        mh = re.match(r"hard_(\w+?)_(\d+)x(\d+)x(\d+)_s([0-9.]+)_Q(\d+)", tag)
        mg = re.match(r"hg_(\w+?)_K(\d+)_(\d+)x(\d+)x(\d+)_Q(\d+)", tag)
        mp_ = re.search(r"_p([0-9.]+)(?:_|$)", tag)
        msp = re.search(r"_sp([0-9.]+)x([0-9.]+)", tag)
        msc = re.search(r"_sc([0-9.]+)-([0-9.]+)", tag)
        mlat = (re.match(r"(lat|lm)(\d+)_(\w+?)_Q(\d+)", tag)
                or re.match(r"(lrn)(\d+)_(\w+?)_M(\d+)", tag))
        if mlat:
            rows.append({
                "tag": tag,
                "model": ("basis-learned" if s.get("learned_basis")
                          else "lattice-learned" if s.get("learn_mu") else "lattice"),
                "kernel": ar.get("kernel", "monopole"),
                "learn_mu": bool(s.get("learn_mu")),
                "mu_shift": (round(s["mu_shift_med"], 2)
                             if s.get("mu_shift_med") else None),
                "n": int(mlat.group(2)), "K": s.get("K"), "S": s.get("S"),
                # learned-basis fits record the shape-basis size as top-level "Q" and as args["M"];
                # lattice fits put it in args["Q"]. Reading only args["Q"] left the column
                # empty for every lrn* row.
                "KS": s.get("KS"),
                "Q": ar.get("Q", s.get("Q", ar.get("M"))), "P": s.get("K"),
                "nz": None, "sigma": None,
                "sites_used": s.get("sites_used"), "cands_used": s.get("cands_used"),
                "penalty": "none (1-of-K)", "level": 0.0,
                "nmse": s.get("full_nmse"),
                "pos_used": 1.0, "pos_top1": 1.0,
                "slope_x": fr.get("slope_x"), "slope_y": fr.get("slope_y"),
                "r_x": fr.get("r_x"), "r_y": fr.get("r_y"),
                "spread_y": fr.get("spread_y"), "resid_y": fr.get("resid_y"),
                "resid_x": fr.get("resid_x"), "pexp": None,
                "span_x": 150.0, "span_y": 150.0,
                "C_soft": dr.get("C_soft_use"), "C_hard": dr.get("C_hard_use"),
                "gt_r_hard": dr.get("gt_r_use"), "gt_gain_hard": dr.get("gt_gain_hard"),
                "D_amp_hard": dr.get("D_amp_hard"),
                "lattice_frac": dr.get("lattice_frac"),
                "outside": (round(dr["outside_frac"] * 100, 2)
                            if dr.get("outside_frac") is not None else None),
                "limited": bool(dr.get("limited")),
                "scale_lo": 1.0, "scale_hi": 512.0,
                "panels": {k: (f"{tag}/{fn}" if (d / fn).exists() else None)
                           for k, fn, _ in PANELS},
                **_real_cols(d)})
            continue
        rows.append({
            "tag": tag,
            "model": ("hard-fixedvol" if mg else "hard" if mh else
                      ar.get("model", "grid" if tag.startswith("grid") else "cp")),
            "kernel": ar.get("kernel", "monopole"),
            "P": (s.get("K") if (mh or mg) else
                  (ar.get("P") if tag.startswith("grid") else None)),
            "Q": ar.get("Q") if (mh or mg or tag.startswith("grid")) else None,
            "K": s.get("K") if (mh or mg) else
                 (None if tag.startswith("grid") else ar.get("K")),
            "nz": int(mg.group(5)) if mg else (int(mh.group(4)) if mh else None),
            "sigma": None if mg else (float(mh.group(5)) if mh else None),
            "sites_used": s.get("sites_used"),
            "penalty": ("hard 1-of-K" if (mh or mg) else
                        ("entropy" if me else ("L1" if ml else "none"))),
            "level": (0.0 if (mh or mg) else
                      (float(me.group(1)) if me else
                       (float(ml.group(1)) if ml else 0.0))),
            "nmse": s.get("full_nmse"),
            "pos_used": s.get("pos_used_mean") or s.get("pi_used_mean"),
            "pos_top1": s.get("pos_mass_top1") or s.get("pi_mass_top1"),
            "slope_x": fr.get("slope_x"), "slope_y": fr.get("slope_y"),
            "r_x": fr.get("r_x"), "r_y": fr.get("r_y"),
            "spread_y": fr.get("spread_y"), "resid_y": fr.get("resid_y"),
            "resid_x": fr.get("resid_x"),
            "pexp": float(mp_.group(1)) if mp_ else None,
            "span_x": (90.0 if mg else
                       (float(msp.group(1)) if msp else (28.0 if mh else None))),
            "span_y": (60.0 if mg else
                       (float(msp.group(2)) if msp else (44.0 if mh else None))),
            "C_soft": dr.get("C_soft_use"), "C_hard": dr.get("C_hard_use"),
            "gt_r_hard": dr.get("gt_r_use"), "gt_gain_hard": dr.get("gt_gain_hard"),
            "D_amp_hard": dr.get("D_amp_hard"),
            "lattice_frac": dr.get("lattice_frac"),
            "outside": (round(dr["outside_frac"] * 100, 2)
                        if dr.get("outside_frac") is not None else None),
            "limited": bool(dr.get("limited")),
            "scale_lo": float(msc.group(1)) if msc else None,
            "scale_hi": float(msc.group(2)) if msc else None,
            "panels": {k: (f"{tag}/{fn}" if (d / fn).exists() else None)
                       for k, fn, _ in PANELS},
            **_real_cols(d),
        })
    for tag, c in ctl.items():
        d = figs / tag
        rows.append({
            "tag": tag, "model": "reference", "kernel": "—",
            "n": None, "K": None, "S": None, "KS": None, "Q": None, "P": None,
            "nz": None, "sigma": None, "sites_used": None, "cands_used": None,
            "penalty": "—", "level": None, "nmse": None,
            "pos_used": None, "pos_top1": None,
            "slope_x": None, "slope_y": None, "r_x": None, "r_y": None,
            "spread_y": None, "resid_y": None, "resid_x": None, "pexp": None,
            "span_x": None, "span_y": None,
            "C_soft": c.get("C_soft"), "C_hard": c.get("C_hard"),
            "gt_r_hard": c.get("gt_r_hard"), "gt_gain_hard": c.get("gt_gain_hard"),
            "D_amp_hard": c.get("D_amp_hard"),
            "lattice_frac": c.get("lattice_frac"),
            "outside": round(c.get("outside_frac", 0) * 100, 2), "limited": False,
            "scale_lo": None, "scale_hi": None,
            "panels": {k: (f"{tag}/{fn}" if (d / fn).exists() else None)
                       for k, fn, _ in PANELS},
            **_real_cols(d)})
    extension_file = figs / "overall_extensions.json"
    if extension_file.exists():
        payload = json.loads(extension_file.read_text())
        for extension in payload.get("rows", []):
            row = dict(extension)
            root = figs / row.pop("panel_root")
            explicit = row.pop("panel_files", {})
            row.setdefault("panels", {})
            row["panels"] = {
                key: (str((root / explicit.get(key, filename)).relative_to(figs))
                      if (root / explicit.get(key, filename)).exists() else None)
                for key, filename, _ in PANELS
            }
            # Keep the browser schema stable for sorting/filtering even when a
            # family does not define a lattice-specific quantity.
            defaults = {
                "kernel": "monopole", "n": None, "K": None, "S": None,
                "KS": None, "Q": None, "P": None, "nz": None,
                "sigma": None, "sites_used": None, "cands_used": None,
                "penalty": "—", "level": None, "nmse": None,
                "pos_used": None, "pos_top1": None, "slope_x": None,
                "slope_y": None, "r_x": None, "r_y": None,
                "spread_y": None, "resid_y": None, "resid_x": None,
                "pexp": None, "span_x": None, "span_y": None,
                "C_soft": None, "C_hard": None, "gt_r_hard": None,
                "gt_gain_hard": None, "D_amp_hard": None,
                "lattice_frac": None, "outside": None, "limited": False,
                "scale_lo": None, "scale_hi": None, "learn_mu": True,
                "mu_shift": None, "real_r": None, "real_gain": None,
                "real_r_nr": None,
                "study": None, "study_label": None, "experiment_id": None,
                "seed": None, "eval_nmse": None, "loo_nmse": None,
                "support_mean": None, "multi_source_fraction": None,
                "condition_p95": None, "exact_support_agreement": None,
                "location_r": None, "barycenter_r": None,
                "displacement_p95": None, "boundary_fraction": None,
                "held_profile_sensitivity": None, "baseline_loo": None,
                "scientific_status": None, "promotion_status": None,
                "promotion_passed": None, "failure_reason": None,
                "failed_gates": [], "completion_status": None,
                "experiment_role": None, "source_cap": None,
                "spatial_dictionary": None, "spatial_profile": None,
                "temporal_sharing": None, "metric_scope": None,
                "misfit": None, "example_flags": [],
                "representative_event_ids": [], "panel_scope": None,
                "unavailable_panels": {}, "state_href": None,
                "bounded_state_href": None, "manifest_href": None,
                "full_recording_status": None,
                "model_href": None, "metrics_href": None,
                "gates_href": None, "report_href": None,
                "provenance_href": None, "audit_href": None,
            }
            rows.append({**defaults, **row})
    return rows


HTML = """<meta charset="utf-8"><title>tensor factorization browser</title>
<style>
:root{--bg:#0f1115;--fg:#e7e9ee;--dim:#9aa3b2;--card:#171a21;--line:#272c36;--acc:#4c8dff}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#14171c;--dim:#5b6472;
      --card:#f5f6f8;--line:#dde1e8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-sans-serif,-apple-system,'Segoe UI',sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:16px}
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
   cursor:pointer;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:4px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
tr.sel td{background:rgba(76,141,255,.16)}
tr:hover td{background:rgba(127,127,127,.10);cursor:pointer}
.g{color:#2f9e44}.b{color:#c92a2a}
#scatter svg{background:var(--card);border:1px solid var(--line);border-radius:8px}
#scatter circle{cursor:pointer}
#tip{position:fixed;pointer-events:none;z-index:50;background:var(--card);
     border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:11px;
     box-shadow:0 4px 18px rgba(0,0,0,.45);display:none;white-space:pre;
     font-variant-numeric:tabular-nums}
.panels h2{font-size:13px;margin:16px 0 5px;color:var(--dim);font-weight:600}
.panels video{max-width:100%;border:1px solid var(--line);border-radius:6px;
              background:#0f1115;display:block}
.panels img{max-width:100%;border:1px solid var(--line);border-radius:6px;
            background:#fff;display:block}
.sheet{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
.sheet figure{margin:0}
.sheet figcaption{font-size:11px;color:var(--dim);padding:3px 0}
.count{color:var(--dim);font-size:12px;padding:6px 18px}
.explore-note{margin-top:8px;padding:8px 10px;border-left:3px solid #845ef7;
              background:rgba(132,94,247,.09);max-width:1150px}
.detail{margin:14px 0;padding:13px 14px;border:1px solid var(--line);border-radius:8px;
        background:var(--card);max-width:1180px}
.detail h2{margin:0 0 7px;color:var(--fg);font-size:15px}.detail p{margin:6px 0}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:7px 15px}
.detail-grid div{color:var(--dim)}.detail-grid b{display:block;color:var(--fg);font-size:12px}
.badge{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:700;font-size:11px;
       border:1px solid var(--line);margin-right:6px}
.badge.ref{background:rgba(47,158,68,.18);color:#69db7c}.badge.no{background:rgba(201,42,42,.16);color:#ff8787}
.badge.diag{background:rgba(132,94,247,.18);color:#b197fc}
.links a{color:var(--acc);margin-right:12px}.unavailable{color:var(--dim);font-size:11px}
</style>
<header>
  <h1>tensor factorization browser</h1>
  <div class="note">
    Y[s,c,t] ≈ Σ<sub>p,q</sub> π[s,p,q] g<sub>p</sub>(Δ[s,c]) a<sub>q</sub>[t], fitted on
    DC-removed waveforms. <b>nMSE</b> references: free rank-1 oracle <b>0.1029</b>,
    per-slot mean 0.3189. <b>slope</b> is the implied position against the monopole —
    1.0 would be full dynamic range; correlation stays high even when the estimate
    barely moves, so slope is the honest axis. <b>pos_top1</b> is the π mass in the
    single best position. Click a row or a dot to open its panels.<br>
    <b>C_soft / C_hard</b> are the mean pairwise ZNCC of each fit's own images
    I<sub>t</sub> — 245 images, each <b>one second</b> wide, sampled every 8 s, on the
    project's standard grid with a 4 µm blur and a ±80 µm y search. These fits minimised
    waveform nMSE <i>only</i>, so <b>C is held out here</b>, not a training score.<br>
    <b style="color:#c92a2a">Do not read C as a localization score.</b> The anchor-only
    collapse control — every spike placed at its peak channel, waveform discarded — scores
    C_hard <b>0.795</b>, higher than all fitted methods, while recovering the imposed drift far
    worse than the monopole (GT r +0.465 vs +0.865). C rises as localizations collapse
    onto the channel grid. Monopole baseline: C_soft 0.266, C_hard 0.504.<br>
    <b>outside %</b> is the fraction of sources landing off the grid (which ends at
    z=160 µm; the third coordinate is source depth for monopole kernels but kernel WIDTH
    for gauss, so they are not one axis). The 8 fits above 5% are re-measured with z
    clipped into the grid — colour by <i>measurement-limited</i> to see them.
    <br><b>Primary ranking:</b> <b>real GT r</b>, the rigid canonical
    SpikeInterface <code>dredge_ap</code> trace against imposed motion. The table and
    scatter open on this metric; internal GT r remains visible for comparison.
    <br><b>New centroid / basis panels:</b> each learned dot is the spike anchor plus its
    chosen localization-codebook site, coloured by argmax<sub>q</sub>|v<sub>s,q</sub>|.
    A deterministic display-only ±1.5 µm jitter separates spikes that share a site; the
    depth × time views additionally use ±0.18 s jitter. Reference methods have no learned
    time basis and are therefore shown in one labelled neutral colour.
    <br><b>Additional model examples:</b> registered multipole rows display every active
    source.  A source contributes fractional count <i>q</i> (and amplitude × <i>q</i>) so
    the source weights from each parent spike sum to one.  The heteroscedastic H2 row is
    an explicitly labelled diagnostic example, not a promotion decision.
    <div class="explore-note"><b>Exploratory spatial models stay visible even when they fail.</b>
    Generalized-g and co-located scale-mixture rows are registered only after frozen
    full-recording inference and the same standard C/D, DREDge, raster, reconstruction,
    embedding, and correction audit used for the multipole rows. Bounded <b>eval nMSE</b>,
    <b>LOO nMSE</b>, and saved promotion gates remain visible separately. Browser inclusion
    is comparative evaluation, not promotion.</div>
  </div>
</header>
<div class="controls" id="controls"></div>
<div class="count" id="count"></div>
<main>
  <div id="scatter"></div>
  <div id="tablewrap"></div>
  <div class="panels" id="panels"></div>
</main>
<div id="tip"></div>
<script>
const ROWS = __DATA__, PANELS = __PANELS__, REF = __REF__;
const FILTERS = [["study","study"],["scientific_status","status"],["misfit","failure mode"],
                 ["model","model"],["learn_mu","learned µ"],["kernel","kernel"],
                 ["n","lattice n"],["P","K/P"],["Q","Q"],
                 ["nz","n scales"],["sigma","σ"],["span_x","span x"],
                 ["span_y","span y"],["scale_hi","scale hi"],["pexp","p"],
                 ["penalty","penalty"],["level","level"]];
const COLS = [["tag","fit"],["study","study"],["scientific_status","status"],
  ["model","model"],["kernel","kernel"],["n","n"],
  ["K","K=n³"],["cands_used","cands used"],["sites_used","sites used"],
  ["mu_shift","µ shift"],["P","K/P"],
  ["Q","Q"],["nz","n sc"],["sigma","σ"],["span_x","span x"],["span_y","span y"],
  ["scale_hi","sc hi"],["penalty","penalty"],["level","level"],["nmse","full nMSE"],
  ["eval_nmse","eval nMSE"],["loo_nmse","LOO nMSE"],["location_r","loc r"],
  ["displacement_p95","disp P95"],["multi_source_fraction","multi"],
  ["condition_p95","cond P95"],["exact_support_agreement","exact"],["pos_used","pos used"],
  ["pos_top1","pos top1"],["slope_x","slope x"],["slope_y","slope y"],
  ["r_y","r y"],["spread_y","spread y"],["resid_y","resid y"],
  ["C_soft","C soft"],["C_hard","C hard"],["gt_r_hard","GT r"],
  ["real_r","real GT r"],["real_gain","real gain"],["real_r_nr","real GT r nr"],
  ["D_amp_hard","D amp"],["lattice_frac","lattice"],["outside","outside %"]];
const AX = [["nmse","full-recording reconstruction nMSE"],
            ["eval_nmse","bounded evaluation nMSE"],["loo_nmse","strict channel-LOO nMSE"],
            ["displacement_p95","dominant-source displacement P95 (µm)"],
            ["condition_p95","support Gram condition P95"],["n","lattice side n"],
            ["K","lattice sites K = n³"],["Q","time-basis size Q"],["C_hard","mean C_hard (held out)"],
            ["C_soft","mean C_soft (held out)"],["sigma","kernel width σ (µm)"],
            ["Q","number of shared shapes Q"],["span_x","lattice span in x (µm)"],
            ["outside","% of sources outside the grid"],
            ["mu_shift","median µ displacement (µm)"],
            ["pos_top1","π mass in top position"],
            ["pos_used","positions used"],["level","penalty level"]];
const AY = [["location_r","dominant localization correlation with retained reference"],
            ["barycenter_r","barycentric localization correlation with retained reference"],
            ["exact_support_agreement","screened/exact support agreement"],
            ["C_hard","mean C_hard — pairwise ZNCC of I_t, held out"],
            ["C_soft","mean C_soft — pairwise ZNCC of I_t, held out"],
            ["gt_r_hard","DREDge motion vs the imposed drift"],
            ["real_r","canonical dredge_ap rigid vs imposed drift"],
            ["slope_y","slope of implied y vs monopole"],
            ["slope_x","slope of implied x vs monopole"],
            ["spread_y","std of implied y (µm)"],
            ["resid_y","residual scatter about the monopole (µm)"],
            ["nmse","reconstruction nMSE"]];
const state={f:{},sort:"real_r",desc:true,sel:null,sheet:null,preset:null,
             sx:"nmse",sy:"real_r",cby:"model"};
const CBY=[["study","study"],["scientific_status","status"],["kernel","kernel"],
           ["n","lattice n"],["Q","Q"],["model","model"],
           ["P","K/P"],["nz","nz"],
           ["sigma","σ"],["penalty","penalty"],["level","level"],
           ["limited","measurement-limited"],[null,"none"]];
const VIR=[[68,1,84],[62,74,137],[38,130,142],[53,183,121],[180,222,44],[253,231,37]];
function ramp(t){t=Math.max(0,Math.min(1,t));const x=t*(VIR.length-1),i=Math.floor(x),
  f=x-i,a=VIR[i],b=VIR[Math.min(i+1,VIR.length-1)];
  return `rgb(${a.map((v,k)=>Math.round(v+f*(b[k]-v))).join(",")})`;}
const CAT={true:"#f59f00",false:"#4c8dff",generalized_spatial:"#15aabf",
           colocated_scale_mixture:"#845ef7","retained reference":"#2f9e44",
           "exploratory—not promoted":"#c92a2a","negative control":"#e8590c",
           "identifiability diagnostic":"#845ef7","matched control—not promoted":"#f59f00",
           monopole:"#4c8dff",gauss:"#e8590c",entropy:"#2f9e44",L1:"#845ef7",
           none:"#888",hard:"#c92a2a","hard-fixedvol":"#f59f00",
           grid:"#2f9e44",cp:"#845ef7",
           "hard 1-of-K":"#c92a2a"};
const num=v=>{if(v===null||v===undefined)return "—";const x=+v;
  if(Number.isInteger(x)&&Math.abs(x)<1e4)return String(x);
  if(Math.abs(x)>=1e4||(Math.abs(x)>0&&Math.abs(x)<1e-3))return x.toExponential(2);
  return x.toFixed(3);};
const PRESETS=[["exploratory","all exploratory"],["reconstruction","largest recon gain"],
  ["localization","largest localization change"],["stable","stable + gain"],
  ["source_count","changed source count"],["pair_replaced","pair → broad composite"],
  ["search","support disagreement"],["conditioning","high conditioning"],
  ["boundary","boundary saturation"],["sensitivity","held sensitivity"],
  ["c5","C5 held-channel failure"]];
// panels are a mix of stills and 1 s/frame movies; pick the right element per extension
const media=src=>src.endsWith(".mp4")
  ? `<video src="${src}" controls loop muted playsinline preload="metadata"></video>`
  : `<img loading="lazy" src="${src}">`;
function colourOf(r){
  const k=state.cby; if(!k) return "#8aa";
  const v=r[k]; if(v===null||v===undefined) return "#666";
  if(CAT[v]) return CAT[v];
  const vs=[...new Set(ROWS.map(x=>x[k]).filter(x=>x!==null&&x!==undefined))]
            .sort((a,b)=>a-b);
  return ramp(vs.length>1?vs.indexOf(v)/(vs.length-1):0.5);}
const uniq=k=>[...new Set(ROWS.map(r=>r[k]).filter(v=>v!==null&&v!==undefined))]
              .sort((a,b)=>typeof a==="number"?a-b:String(a).localeCompare(b));
function ctrl(){
  const c=document.getElementById("controls");c.innerHTML="";
  const pg=document.createElement("div");pg.className="grp";pg.innerHTML="<b>misfit presets</b>";
  const pmk=(v,t)=>{const b=document.createElement("button");b.textContent=t;
    b.className=state.preset===v?"on":"";b.onclick=()=>{state.preset=state.preset===v?null:v;draw();};pg.append(b);};
  pmk(null,"off");PRESETS.forEach(([v,t])=>pmk(v,t));c.append(pg);
  for(const [k,lab] of FILTERS){
    const g=document.createElement("div");g.className="grp";g.innerHTML=`<b>${lab}</b>`;
    const mk=(v,t)=>{const b=document.createElement("button");b.textContent=t;
      b.className=(state.f[k]===v||(v===null&&state.f[k]===undefined))?"on":"";
      b.onclick=()=>{v===null?delete state.f[k]:state.f[k]=v;draw();};g.append(b);};
    mk(null,"all");uniq(k).forEach(v=>mk(v,String(v)));c.append(g);}
  for(const [nm,lab,arr,key] of [["scatter x","x",AX,"sx"],["scatter y","y",AY,"sy"]]){
    const g=document.createElement("div");g.className="grp";g.innerHTML=`<b>${nm}</b>`;
    arr.forEach(([v,t])=>{const b=document.createElement("button");b.textContent=v;
      b.title=t;b.className=state[key]===v?"on":"";
      b.onclick=()=>{state[key]=v;draw();};g.append(b);});c.append(g);}
  const cb=document.createElement("div");cb.className="grp";cb.innerHTML="<b>colour</b>";
  CBY.forEach(([v,t])=>{const b=document.createElement("button");b.textContent=t;
    b.className=state.cby===v?"on":"";b.onclick=()=>{state.cby=v;draw();};cb.append(b);});
  c.append(cb);
  const sh=document.createElement("div");sh.className="grp";
  sh.innerHTML="<b>contact sheet</b>";
  const mk2=(v,t)=>{const b=document.createElement("button");b.textContent=t;
    b.className=state.sheet===v?"on":"";b.onclick=()=>{state.sheet=v;draw();};sh.append(b);};
  mk2(null,"off");PANELS.forEach(p=>mk2(p[0],p[0]));c.append(sh);
}
function presetMatch(r){const f=r.example_flags||[];switch(state.preset){
  case "exploratory":return !!r.study;case "reconstruction":return f.includes("largest reconstruction improvement");
  case "localization":return f.includes("largest localization displacement");
  case "stable":return f.includes("stable localization with improved reconstruction");
  case "source_count":return f.includes("changed source count");
  case "pair_replaced":return f.includes("pair replaced by one broad composite");
  case "search":return f.includes("exact/screened support disagreement");
  case "conditioning":return f.includes("high conditioning");
  case "boundary":return f.includes("profile-boundary saturation");
  case "sensitivity":return f.includes("high held-channel sensitivity");
  case "c5":return f.includes("catastrophic C5 held-channel failure");default:return true;}}
function match(r){return presetMatch(r)&&FILTERS.every(([k])=>state.f[k]===undefined||r[k]===state.f[k]);}
function detail(r){if(!r.study)return "";const cls=r.scientific_status==="retained reference"?"ref":
  (r.scientific_status||"").includes("exploratory")|| (r.scientific_status||"").includes("not promoted")?"no":"diag";
  const link=(href,label)=>href?`<a href="${href}" target="_blank">${label}</a>`:"";
  const unavailable=Object.entries(r.unavailable_panels||{}).filter(([,v])=>v)
    .map(([k,v])=>`<li><b>${k.replaceAll("_"," ")}</b>: ${v}</li>`).join("");
  return `<section class="detail"><h2>${r.tag}</h2>
    <p><span class="badge ${cls}">${r.scientific_status}</span><span class="badge">${r.completion_status}</span></p>
    <div class="detail-grid"><div>experiment<b>${r.experiment_id} · seed ${r.seed}</b></div>
    <div>role<b>${r.experiment_role}</b></div><div>dictionary<b>${r.spatial_dictionary}</b></div>
    <div>profile<b>${r.spatial_profile}</b></div><div>source cap<b>R ≤ ${r.source_cap}</b></div>
    <div>temporal model<b>${r.temporal_sharing}</b></div><div>metric scope<b>${r.metric_scope}</b></div>
    <div>primary misfit<b>${r.misfit}</b></div></div>
    <p><b>Failed gates:</b> ${r.failure_reason}</p><p><b>Full-recording status:</b> ${r.full_recording_status}</p>
    <p><b>Panel scope:</b> ${r.panel_scope}</p>
    <p class="links">${link(r.state_href,"saved state")}${link(r.model_href,"model")}
    ${link(r.bounded_state_href,"bounded state")}${link(r.manifest_href,"full manifest")}
    ${link(r.metrics_href,"metrics")}${link(r.gates_href,"gates")}${link(r.report_href,"report")}
    ${link(r.provenance_href,"provenance")}${link(r.audit_href,"audit")}</p>
    ${unavailable?`<details class="unavailable"><summary>Unavailable standard panels and why</summary><ul>${unavailable}</ul></details>`:""}</section>`;}
function scatter(keep){
  const el=document.getElementById("scatter");
  const rs=ROWS.filter(r=>r[state.sx]!==null&&r[state.sy]!==null);
  if(!rs.length){el.innerHTML="";return;}
  const W=980,H=400,L=64,R=18,T=14,B=44;
  const xs=rs.map(r=>r[state.sx]),ys=rs.map(r=>r[state.sy]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const px=v=>L+(v-x0)/((x1-x0)||1)*(W-L-R), py=v=>T+(1-(v-y0)/((y1-y0)||1))*(H-T-B);
  const tick=(lo,hi)=>{const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/4)||0));
    const s2=[1,2,5,10].map(m=>m*st).find(m=>(hi-lo)/m<=6)||st;const o=[];
    for(let v=Math.ceil(lo/s2)*s2;v<=hi;v+=s2)o.push(+v.toFixed(6));return o;};
  let g="";
  for(const v of tick(x0,x1)) g+=`<line x1="${px(v)}" y1="${T}" x2="${px(v)}" y2="${H-B}"
     stroke="var(--line)"/><text x="${px(v)}" y="${H-B+15}" fill="var(--dim)"
     font-size="10" text-anchor="middle">${v}</text>`;
  for(const v of tick(y0,y1)) g+=`<line x1="${L}" y1="${py(v)}" x2="${W-R}" y2="${py(v)}"
     stroke="var(--line)"/><text x="${L-7}" y="${py(v)+3}" fill="var(--dim)"
     font-size="10" text-anchor="end">${v}</text>`;
  if(state.sx==="nmse") g+=`<line x1="${px(REF.free_rank1)}" y1="${T}"
     x2="${px(REF.free_rank1)}" y2="${H-B}" stroke="#2f9e44" stroke-dasharray="5 4"/>
     <text x="${px(REF.free_rank1)+4}" y="${T+11}" fill="#2f9e44" font-size="10">
     free rank-1 oracle ${REF.free_rank1}</text>`;
  if(state.sy==="spread_y") g+=`<line x1="${L}" y1="${py(REF.mono_spread_y)}"
     x2="${W-R}" y2="${py(REF.mono_spread_y)}" stroke="#c92a2a" stroke-dasharray="5 4"/>
     <text x="${W-R-4}" y="${py(REF.mono_spread_y)-4}" fill="#c92a2a" font-size="10"
     text-anchor="end">monopole ${REF.mono_spread_y} µm</text>`;
  for(const r of rs){const on=keep.has(r.tag),s=state.sel===r.tag;
    g+=`<circle cx="${px(r[state.sx])}" cy="${py(r[state.sy])}" r="${s?8:5.5}"
        fill="${colourOf(r)}" opacity="${on?0.95:0.12}"
        stroke="${s?"#fff":"rgba(0,0,0,.5)"}" stroke-width="${s?2.2:0.7}"
        data-tag="${r.tag}"/>`;}
  g+=`<text x="${(L+W-R)/2}" y="${H-6}" fill="var(--dim)" font-size="11"
      text-anchor="middle">${AX.find(a=>a[0]===state.sx)[1]}</text>
      <text x="14" y="${(T+H-B)/2}" fill="var(--dim)" font-size="11" text-anchor="middle"
      transform="rotate(-90 14 ${(T+H-B)/2})">${AY.find(a=>a[0]===state.sy)[1]}</text>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`;
  const tip=document.getElementById("tip");
  el.querySelectorAll("circle").forEach(c=>{
    const r=ROWS.find(x=>x.tag===c.dataset.tag);
    c.onmousemove=e=>{tip.style.display="block";
      tip.style.left=Math.min(e.clientX+14,innerWidth-320)+"px";
      tip.style.top=(e.clientY+14)+"px";
      tip.textContent=`${r.tag}\\nkernel ${r.kernel}  P ${r.P}  Q ${r.Q}`+
        `\\n${r.penalty} ${r.level}`+
        `\\nnMSE ${num(r.nmse)}   pos used ${num(r.pos_used)}  top1 ${num(r.pos_top1)}`+
        `\\nslope ${num(r.slope_x)} / ${num(r.slope_y)}   spread_y ${num(r.spread_y)}`;};
    c.onmouseleave=()=>{tip.style.display="none";};
    c.onclick=()=>{state.sel=state.sel===r.tag?null:r.tag;state.sheet=null;draw();
      document.getElementById("panels").scrollIntoView({behavior:"smooth"});};});
}
function draw(){
  ctrl();
  let rs=ROWS.filter(match);
  rs.sort((a,b)=>{const x=a[state.sort],y=b[state.sort];
    if(x===null||x===undefined)return 1; if(y===null||y===undefined)return -1;
    const c=typeof x==="number"?x-y:String(x).localeCompare(String(y));
    return state.desc?-c:c;});
  document.getElementById("count").textContent=`${rs.length} of ${ROWS.length} fits`;
  let h="<table><thead><tr>"+COLS.map(([k,l])=>
    `<th data-k="${k}">${l}${state.sort===k?(state.desc?" ▾":" ▴"):""}</th>`).join("")+
    "</tr></thead><tbody>";
  for(const r of rs){
    h+=`<tr class="${state.sel===r.tag?"sel":""}" data-tag="${r.tag}">`+
      COLS.map(([k])=>{let v=r[k],s=typeof v==="number"?num(v):(v??"—"),c="";
        if(k==="slope_y"&&v!==null) c=(v>0.6&&v<1.3)?"g":(v<0.12?"b":"");
        if(k==="resid_y"&&v!==null) c=v<14?"g":(v>25?"b":"");
        if(k==="nmse"&&v!==null) c=v<REF.free_rank1?"g":(v>0.25?"b":"");
        if(k==="pos_top1"&&v!==null) c=v>0.5?"g":(v<0.15?"b":"");
        return `<td class="${c}">${s}</td>`;}).join("")+"</tr>";}
  h+="</tbody></table>";
  document.getElementById("tablewrap").innerHTML=h;
  document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if(state.sort===k)state.desc=!state.desc;else{state.sort=k;state.desc=true;}draw();});
  document.querySelectorAll("tr[data-tag]").forEach(tr=>tr.onclick=()=>{
    state.sel=state.sel===tr.dataset.tag?null:tr.dataset.tag;state.sheet=null;draw();
    document.getElementById("panels").scrollIntoView({behavior:"smooth"});});
  scatter(new Set(rs.map(r=>r.tag)));
  const p=document.getElementById("panels");
  if(state.sheet){
    const d=PANELS.find(x=>x[0]===state.sheet);
    p.innerHTML=`<h2>${d[2]} — ${rs.length} shown</h2><div class="sheet">`+
      rs.map(r=>r.panels[state.sheet]?
        `<figure><figcaption>${r.tag}</figcaption>
         <a href="${r.panels[state.sheet]}" target="_blank">
         ${media(r.panels[state.sheet])}</a></figure>`:"").join("")+
      "</div>";
  } else if(state.sel){
    const r=ROWS.find(x=>x.tag===state.sel);
    p.innerHTML=detail(r)+`<h2>${r.tag}</h2>`+PANELS.map(([k,fn,lab])=>r.panels[k]?
      `<h2>${lab} — <a href="${r.panels[k]}" target="_blank">${fn}</a></h2>
       ${media(r.panels[k])}`:"").join("");
  } else p.innerHTML="<h2>click a row or a dot to open its panels</h2>";
}
const params=new URLSearchParams(location.search);
for(const k of ["study","scientific_status","misfit"])if(params.get(k))state.f[k]=params.get(k);
if(params.get("preset"))state.preset=params.get("preset");
if(params.get("sx")&&AX.some(x=>x[0]===params.get("sx")))state.sx=params.get("sx");
if(params.get("sy")&&AY.some(x=>x[0]===params.get("sy")))state.sy=params.get("sy");
if(params.get("select"))state.sel=params.get("select");
draw();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/tensor")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/tensor")
    ap.add_argument("--index-name", default="index.html",
                    help="write a copy under this filename without replacing index.html")
    a = ap.parse_args()
    rows = collect(a.runs, a.figs)
    index_path = a.figs / a.index_name
    index_path.write_text(
        HTML.replace("__DATA__", json.dumps(rows))
            .replace("__PANELS__", json.dumps(PANELS))
            .replace("__REF__", json.dumps(REF)))
    npan = sum(1 for r in rows if any(r["panels"].values()))
    print(f"wrote {index_path}  ({len(rows)} fits, {npan} with figures)")


if __name__ == "__main__":
    main()
