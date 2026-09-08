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
    if relative.startswith("recording_replay_chunk") and relative not in PANEL_REGISTRY:
        stem = relative[len("recording_replay_chunk"):].removesuffix(".png")
        group, label, description = (
            "reconstruction",
            f"recording replay (chunk {int(stem)}, most subtractive)",
            "The highest captured-energy chunk: input versus residuals after each recording pass.",
        )
    else:
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


CHIP_EXPLANATIONS = {
    "events": "accepted spike events across all recording passes and peeling rounds",
    "chunks": "one-second chunks the recording was streamed through, one GPU visit each",
    "duration": "recording span covered by the run",
    "channels": "recording channels processed",
    "detector": "how candidates are proposed: signed local extrema of the noise-standardized residual, then a local winner-take-all over neighbors",
    "threshold": "detection trigger level in per-channel robust-noise units (MAD/0.6745); the same on every pass, only the acceptance bar escalates",
    "temporal atoms": "rows in the shared temporal codebook Omega; every event selects exactly one atom (one-hot)",
    "prototype pair": "number of temporal prototypes: every atom is constrained to sit inside the cone of one of them, one cone per extremum polarity",
    "cone half-angle": "largest allowed angle between an atom and its prototype, re-enforced after every update",
    "peeling rounds": "detect-fit-subtract repetitions per chunk visit",
    "spatial kernel": "family of spatial footprints the fitted source uses to spread across channels",
    "waveforms": "where the raw snippets backing the reconstruction panels live",
}


def detail_values(summary, metadata):
    config = metadata.get("config", {})
    fs = metadata.get("fs")
    first_sample = metadata.get("first_sample")
    stop_sample = metadata.get("stop_sample")
    duration = None
    if fs and first_sample is not None and stop_sample is not None:
        duration = (stop_sample - first_sample) / fs
    values = [
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
    return [
        (label, value, CHIP_EXPLANATIONS.get(label, "")) for label, value in values
    ]


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def math_sections(metadata, omega_source):
    config = metadata.get("config", {})
    if not config:
        return []
    fs = metadata.get("fs")
    samples = round(
        (config.get("ms_before", 0) + config.get("ms_after", 0))
        * (fs or 0) / 1000
    )
    window = (
        f"±{_fmt(config.get('ms_before'))} ms "
        f"({samples} samples at {_fmt(fs)} kHz)" if fs else "±— ms"
    )
    rule = config.get("all_channel_rule")
    spatial = config.get("spatial_score")
    kernel = config.get("kernel", "—")
    cone = _fmt(config.get("prototype_cone_deg"))
    thr = _fmt(config.get("threshold"))
    f0 = _fmt(config.get("all_channel_min_fraction"))
    step = _fmt(config.get("pass_fraction_step"))
    share = _fmt(config.get("all_channel_required_share"))
    pmin = _fmt(config.get("min_fitted_projection"))
    rmax = _fmt(config.get("max_channel_normalized_rmse"))
    fmin = _fmt(config.get("min_captured_fraction"))
    passes = _fmt(config.get("recording_passes"))
    peel = _fmt(config.get("peeling_rounds"))
    sweep = _fmt(config.get("exclude_sweep_ms"))
    dupcorr = _fmt(config.get("duplicate_temporal_correlation"))
    q = _fmt(config.get("q"))

    sections = []

    body = r"""<p>Each event explains the recording as a single scaled spatial footprint
    times a single temporal atom:</p>
    $$\hat{Y}[c,t] \;=\; \alpha\; g_c(x,y,z,\sigma)\; \Omega_q[t],
      \qquad \alpha \ge 0,$$
    <p>where $q$ is a one-hot choice among the $Q=__Q__$ codebook rows and the
    __KERNELTEXT__ spatial footprint is</p>
    $$g_c \;=\; \frac{\sigma}{\sqrt{\lVert \mathbf{x}_c-\mathbf{x}\rVert_2^2
      \;+\; z^2 \;+\; \sigma^2}},$$
    <p>with $\lVert\mathbf{x}_c-\mathbf{x}\rVert$ the lateral distance from the
    source to channel $c$ and $z$ the fitted depth. $\Omega$ rows are unit-norm
    and their sign encodes polarity, so $\alpha \ge 0$ means an event can only
    ever copy its atom, never flip it.</p>"""
    kerneltext = (
        "monopole" if kernel == "monopole"
        else f"__KERNEL__ (the monopole form below is shown for reference only)"
    )
    sections.append((
        "signal model",
        body.replace("__Q__", q).replace("__KERNELTEXT__", kerneltext)
            .replace("__KERNEL__", kernel),
    ))

    body = r"""<p>Channel noise is estimated robustly from the preprocessed chunk:</p>
    $$n_c \;=\; \frac{\operatorname{median}_t\,\lvert Y[c,t]-m_c\rvert}{0.6745},
      \qquad m_c = \operatorname{median}_t Y[c,t].$$
    <p>Detection runs on the current residual $R$ (the raw chunk minus every
    accepted atom so far), standardized by noise. A sample is a peak when it is
    a signed local extremum in time that crosses the threshold
    $\tau = __THR__$ noise units,</p>
    $$\left\lvert \frac{R[c,t]}{n_c}\right\rvert \;\ge\; \tau,$$
    <p>and proposals then compete inside a $\pm$__SWEEP__ ms window across
    neighboring channels: a peak survives only if no nearby channel sample has a
    strictly larger $\lvert R\rvert/n_c$ (ties go to the earlier sample).
    Survivors are capped at the strongest events per pass. The discovery score
    is signed, $s = R[c,t]/n_c$, so event polarity is known from detection.</p>"""
    sections.append((
        "noise and detection",
        body.replace("__THR__", thr).replace("__SWEEP__", sweep),
    ))

    spatial_case = (
        r"\operatorname{mean}_c\;\text{nRMSE}_c \quad\text{(mean-channel-rmse)}"
        if spatial == "mean-channel-rmse"
        else r"\max_c\;\text{nRMSE}_c \quad\text{(max-channel-rmse)}"
    )
    body = r"""<p>Each proposal is fit over a coarse site lattice and then refined by
    coordinate descent over position, scale $\sigma$, and atom $q$. For a fixed
    candidate, the amplitude has a closed form with noise weights
    $w_c = g_c/n_c$:</p>
    $$\alpha^\* \;=\;
      \frac{\sum_c w_c\,\langle Y_c,\; g_c\,\Omega_q\rangle_t}
           {\sum_c w_c^{\,2}} \;\ge\; 0,$$
    <p>and the search minimizes the noise-normalized reconstruction error</p>
    $$\text{nRMSE}_c \;=\; \frac{1}{n_c}\sqrt{\tfrac{1}{T}\textstyle\sum_t
      \bigl(Y_c-\hat{Y}_c\bigr)^2},
      \qquad S \;=\; __SPATIALCASE__.$$
    <p>This run minimizes __SPATIAL__: the __SPATIALTEXT__ choice. The
    mean-channel objective keeps a narrow template that touches only the peak
    channel from winning the position search.</p>"""
    sections.append((
        "fit and spatial score",
        body.replace("__SPATIALCASE__", spatial_case)
            .replace("__SPATIAL__", spatial or "—")
            .replace(
                "__SPATIALTEXT__",
                "average error over all valid channels"
                if spatial == "mean-channel-rmse"
                else "worst-channel error",
            ),
    ))

    if rule in ("min-channel", "mean-channel", "k-of-n"):
        rule_case = {
            "min-channel": r"f_c \;\ge\; \beta_p \quad \text{for every valid channel } c",
            "mean-channel": r"\frac{1}{\lvert I\rvert}\sum_{c\in I} f_c \;\ge\; \beta_p",
            "k-of-n": r"\#\bigl\{c:\; f_c \ge \beta_p\bigr\} \;\ge\; \lceil s\,n\rceil",
        }[rule]
        rule_text = {
            "min-channel": "every valid channel must clear the bar on its own",
            "mean-channel": "the captured fractions averaged over informative channels must clear the bar",
            "k-of-n": "at least a share $s = __SHARE__$ of the valid channels must clear the bar",
        }[rule].replace("__SHARE__", share)
    else:
        rule_case = r"\text{(rule not recorded)}"
        rule_text = "the per-channel aggregation rule was not saved for this run."
    body = r"""<p>All energies are noise-normalized,
    $\lVert Y\rVert_n^2=\sum_c \lVert Y_c/n_c\rVert^2$. A proposal becomes an
    accepted event only if every gate below holds. The projection score is the
    noise-normalized energy the fitted atom actually captures,</p>
    $$s_{\text{proj}} \;=\; \sqrt{\;\lVert Y\rVert_n^2
      \;-\; \lVert R_{\text{new}}\rVert_n^2\;} \;\ge\; __PMIN__,$$
    <p>the worst channel must not be left with too much structured leftover,</p>
    $$\max_c\;\text{nRMSE}_c \;\le\; __RMAX__,
      \qquad f \;\ge\; __FMIN__,$$
    <p>and the per-channel captured fractions must clear the pass bar
    $\beta_p$. Per channel,</p>
    $$f_c \;=\; \frac{\lVert Y_c\rVert_n^2-\lVert R^{\text{new}}_c\rVert_n^2}
      {\lVert Y_c\rVert_n^2},$$
    <p>with channels of near-zero input energy excluded from the mean and count
    rules, and the gate for this run (__RULE__) is</p>
    $$__RULECASE__$$
    <p>with the bar schedule rising by one step per recording pass and capped at
    0.9,</p>
    $$\beta_p \;=\; \min\bigl(f_0 + \Delta\, p,\; 0.9\bigr),
      \qquad f_0=__F0__,\;\; \Delta=__STEP__,\;\; P=__PASSES__.$$
    <p>__RULETEXT__</p>"""
    sections.append((
        "energy capture and acceptance",
        body.replace("__PMIN__", pmin).replace("__RMAX__", rmax)
            .replace("__FMIN__", fmin).replace("__RULE__", rule or "—")
            .replace("__RULECASE__", rule_case).replace("__F0__", f0)
            .replace("__STEP__", step).replace("__PASSES__", passes)
            .replace("__RULETEXT__", rule_text),
    ))

    body = r"""<p>On acceptance the prediction is subtracted,
    $R \leftarrow R - \hat{Y}$, and the same chunk can be visited for up to
    __PEEL__ peeling rounds. The recording is walked __PASSES__ times:
    pass 2+ rebuilds each chunk's starting residual on the GPU by replaying
    every saved earlier-pass event, so pursuit continues exactly where the last
    pass stopped while the bar keeps escalating. Detection never changes between
    passes. A chunk visit that accepts nothing marks the chunk exhausted, and
    later passes skip it. Proposals that duplicate an earlier-pass event — same
    time within the sweep window, adjacent channel, and temporal-atom cosine at
    least __DUPCORR__ — are dropped as duplicates.</p>"""
    sections.append((
        "peeling and recording passes",
        body.replace("__PEEL__", peel).replace("__PASSES__", passes)
            .replace("__DUPCORR__", dupcorr),
    ))

    kind = (omega_source or {}).get("kind")
    if kind == "external_prior" or config.get("omega_prior"):
        prior = config.get("omega_prior") or (omega_source or {}).get("path", "?")
        body = r"""<p>The temporal codebook was not learned here: this run loaded a
        saved $\Omega$ from <code>__PRIOR__</code> (shape $Q\times T$), re-oriented each
        row so its largest-magnitude sample is negative, and froze it for the whole
        pursuit.</p>"""
        sections.append((
            "temporal codebook provenance",
            body.replace("__PRIOR__", str(prior)),
        ))
    else:
        chunks = _fmt(config.get("calibration_chunks"))
        max_ev = _fmt(config.get("calibration_max_events"))
        ev_chunk = _fmt(config.get("calibration_events_per_chunk"))
        isol = _fmt(config.get("calibration_isolation_ms"))
        iters = _fmt(config.get("alternating_iterations"))
        km = _fmt(config.get("prototype_kmeans_iterations"))
        cone = config.get("prototype_cone_deg")
        cone = _fmt(cone)
        proto = _fmt(config.get("prototype_count"))
        seed = _fmt(config.get("seed"))
        body = r"""<p>The codebook was learned on this recording before pursuit and then
        frozen — no $\Omega$ updates happen while peeling. Calibration draws
        __CHUNKS__ random one-second chunks (seed __SEED__), detects at the same
        threshold $\tau$, keeps isolated events only (at least __ISOL__ ms from any
        other detected event), and takes up to __EVCHUNK__ events per chunk,
        __MAXEV__ total. Waveforms are __WINDOW__ around each peak, peak-aligned
        on their strongest channel and $L_2$-normalized to unit vectors $u_b$.</p>
        <p>The pool splits by extremum polarity into __PROTO__ groups; each
        prototype $p_j$ is the polarity-fixed normalized group mean, and the
        $Q=__Q__$ atoms are dealt out round-robin, $q \bmod __PROTO__$, to the cones.
        Initialization runs spherical k-means (__KM__ cosine iterations) inside each
        group and projects every center into its cone; then __ITERS__ alternating
        iterations re-fit every event and update each row in closed form before
        projecting it back into the cone:</p>
        $$\omega_q \;\leftarrow\; \Pi_{\mathcal{C}_j}\!\left(
          \frac{\sum_{b:\,q(b)=q} w_b\,Y_b}
               {\sum_{b:\,q(b)=q} w_b^{\,2}}\right),
        \qquad w_b = \alpha_b\, g_b,$$
        <p>and each prototype is refit as the leading right singular vector of its
        cone's weighted atom matrix. Updates are backtracked (step $1,\tfrac12,\tfrac14$)
        and only kept when the fixed-assignment objective does not increase; the loop
        stops when the largest row change drops below the tolerance. With cone
        half-angle __CONE__$^\circ$, $c_0=\cos$__CONE__$^\circ$, the projection is</p>
        $$\Pi_{\mathcal{C}}(u) \;=\;
        \begin{cases}
          u, & u^{\!\top} p \,\ge\, c_0,\\[4pt]
          c_0\,p \;+\; \sqrt{1-c_0^2}\;
            \dfrac{u-(u^{\!\top}p)\,p}{\lVert u-(u^{\!\top}p)\,p\rVert},
            & \text{otherwise.}
        \end{cases}$$"""
        sections.append((
            "temporal codebook provenance",
            body.replace("__CHUNKS__", chunks).replace("__SEED__", seed)
                .replace("__ISOL__", isol).replace("__EVCHUNK__", ev_chunk)
                .replace("__MAXEV__", max_ev).replace("__WINDOW__", window)
                .replace("__PROTO__", proto).replace("__Q__", q)
                .replace("__KM__", km).replace("__ITERS__", iters)
                .replace("__CONE__", cone),
        ))
    return sections


HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script>MathJax={tex:{inlineMath:[["$","$"]],displayMath:[["$$","$$"]]},
        svg:{fontCache:"global"}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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
.detail .why{display:block;color:var(--dim);font-size:11px;font-weight:400;margin-top:1px}
.math{margin:12px 0 0;padding:11px 14px;border:1px solid var(--line);border-radius:8px;
      max-width:1180px;background:var(--card)}
.math h3{margin:12px 0 4px;font-size:13px;color:var(--acc)}
.math h3:first-child{margin-top:0}
.math p{margin:5px 0}
.math code{background:rgba(127,127,127,.15);border-radius:4px;padding:1px 4px}
math{color:var(--fg)}
mjx-container{overflow-x:auto;overflow-y:hidden;max-width:100%}
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
<div class="math">__MATH__</div>
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
    omega_source = load_json(run / "omega_source.json")
    title = args.title or f"{run.name} residual pursuit browser"
    details = "".join(
        f"<div>{html.escape(str(label))}<b>{html.escape(str(value))}</b>"
        f"<span class='why'>{html.escape(str(why))}</span></div>"
        if why else
        f"<div>{html.escape(str(label))}<b>{html.escape(str(value))}</b></div>"
        for label, value, why in detail_values(summary, metadata)
    )
    math_html = "".join(
        f"<h3>{html.escape(name)}</h3>{body}"
        for name, body in math_sections(metadata, omega_source)
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
        .replace("__MATH__", math_html)
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
