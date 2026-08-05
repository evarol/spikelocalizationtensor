"""Tabulate every tensor fit by mean pairwise ZNCC, and plot it against waveform MSE.

These fits were optimised for reconstruction ONLY, so mean C is a held-out measurement
here rather than a training score. That is what makes the MSE-vs-C plot worth having.

What the plot needs in order not to mislead is a COLLAPSE CONTROL, and it has one. The
anchor-only localizer -- every spike placed at its peak channel, waveform discarded
entirely -- scores C_hard 0.795, above all 244 fits, while recovering the imposed drift
far worse than the monopole (GT r +0.465 vs +0.865). So C is monotone in positional
collapse and is not, on its own, a localization score. The control line is drawn on every
panel; read distance below it, not height.

Two measurement caveats are handled explicitly rather than buried:

  * BIN WIDTH. Images are ONE SECOND wide (~1240 spikes each), sampled every 8 s. The
    stride subsamples which images are compared, it does not widen them.
  * THE THIRD COORDINATE is not one physical quantity across families -- source depth for
    monopole kernels, spatial width for gauss, the sigma lattice for hard fits. The shared
    grid ends at z=160 um, so 8 fits push sources past it and 4 lose every spike. Those are
    re-measured with z clipped into the grid (dc_clamped.jsonl) and plotted hollow at the
    rescued value, never at the meaningless 0.000.

Reads dc_all.jsonl, dc_controls.jsonl and dc_clamped.jsonl (all from dc_batch.py) joined
to summary_*.json. Writes C_TABLE.md, C_table.json and mse_vs_C.png.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REF_NMSE = {"free rank-1 oracle": 0.1029, "per-slot mean": 0.3189,
            "raw-waveform lookup": 0.4883}
BAD = 0.05                      # outside-grid fraction above which C is not comparable


def family(tag):
    if tag.startswith("lat"):
        return "lattice " + ("gauss" if "_gauss_" in tag else "monopole")
    if tag.startswith("hg_"):
        return "hard, fixed volume"
    if tag.startswith("hard_"):
        return "hard 1-of-K"
    if tag.startswith("grid_"):
        return "grid (position x shape)"
    return "CP"


def read_jsonl(p: Path):
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["tag"]] = r            # later lines win on a redo
    return out


def load_rows(runs: Path, figs: Path):
    dc = read_jsonl(figs / "dc_all.jsonl")
    clamped = read_jsonl(figs / "dc_clamped.jsonl")
    rows = []
    for tag, r in dc.items():
        sf = runs / f"summary_{tag}.json"
        if not sf.exists():
            continue
        s = json.loads(sf.read_text())
        ar = s.get("args", {})
        mh = re.match(r"hard_(\w+?)_(\d+)x(\d+)x(\d+)_s([0-9.]+)_Q(\d+)", tag)
        me = re.search(r"_ent([0-9.]+)", tag)
        ml = re.search(r"_l1([0-9.]+)", tag)
        cl = clamped.get(tag)
        rows.append({**r,
                     "family": family(tag),
                     "kernel": ar.get("kernel", "monopole"),
                     "K": s.get("K") or ar.get("P") or ar.get("K"),
                     "Q": ar.get("Q"),
                     "sigma": float(mh.group(5)) if mh else ar.get("sigma"),
                     "nz": int(mh.group(4)) if mh else ar.get("nz"),
                     "penalty": ("hard 1-of-K" if tag.startswith(("hard_", "hg_"))
                                 else "entropy" if me else "L1" if ml else "none"),
                     "level": float(me.group(1)) if me else
                              (float(ml.group(1)) if ml else 0.0),
                     "nmse": s.get("full_nmse"),
                     "sites_used": s.get("sites_used"),
                     "limited": r["outside_frac"] > BAD,
                     # the value to PLOT: rescued for the fits that fell off the grid
                     "C_soft_use": (cl or r)["C_soft"],
                     "C_hard_use": (cl or r)["C_hard"],
                     "gt_r_use": (cl or r)["gt_r_hard"],
                     "C_soft_clamped": cl["C_soft"] if cl else None,
                     "C_hard_clamped": cl["C_hard"] if cl else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--figs", type=Path, default=REPO / "figures")
    a = ap.parse_args()

    rows = load_rows(a.runs, a.figs)
    ctl = read_jsonl(a.figs / "dc_controls.jsonl")
    rows.sort(key=lambda r: -(r["C_hard_use"] or 0))
    (a.figs / "C_table.json").write_text(json.dumps(rows, indent=1))
    lim = [r for r in rows if r["limited"]]

    # ---------------------------------------------------------------- markdown
    hdr = ["fit", "family", "kernel", "K", "Q", "σ", "penalty", "lvl", "nMSE",
           "C_soft", "C_hard", "GT r", "D amp", "lat", "outside"]
    L = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        star = "\\*" if r["limited"] else ""
        L.append("| " + " | ".join([
            f"`{r['tag']}`", r["family"], r["kernel"],
            str(r["K"] or "—"), str(r["Q"] or "—"),
            f"{r['sigma']:g}" if r.get("sigma") else "—",
            r["penalty"], f"{r['level']:g}",
            f"{r['nmse']:.4f}" if r["nmse"] is not None else "—",
            f"{r['C_soft_use']:.3f}{star}", f"{r['C_hard_use']:.3f}{star}",
            f"{r['gt_r_use']:+.3f}{star}", f"{r['D_amp_hard']:.1f}",
            f"{r['lattice_frac']:.3f}",
            (f"**{r['outside_frac']*100:.1f}%**" if r["limited"]
             else f"{r['outside_frac']*100:.1f}%")]) + " |")
    for tag, c in ctl.items():
        L.append(f"| `{tag}` | **reference** | — | — | — | — | — | — | — | "
                 f"{c['C_soft']:.3f} | {c['C_hard']:.3f} | {c['gt_r_hard']:+.3f} | "
                 f"{c['D_amp_hard']:.1f} | {c['lattice_frac']:.3f} | "
                 f"{c['outside_frac']*100:.1f}% |")

    ac = ctl.get("CONTROL_anchor_only_ptp", {})
    mb = ctl.get("BASELINE_monopole", {})
    head = [
        "# Mean pairwise ZNCC of every tensor fit", "",
        "C(t,t') is the shift-max ZNCC of the per-bin images I_t built from each fit's own "
        "localizations, through the project's standard GridSpec with a 4 µm blur and a "
        "±80 µm y search. `C_soft` uses softmax over lags (τ=0.2), `C_hard` the argmax lag. "
        "**None of these fits optimised C** — they minimised waveform nMSE alone, so C is "
        "held out.", "",
        "**Read this before ranking anything by C.** The collapse control "
        "`CONTROL_anchor_only_ptp` discards the waveform completely and places every spike "
        f"at its peak channel. It scores C_hard **{ac.get('C_hard', 0):.3f}** — higher than "
        f"every one of the {len(rows)} fits — while recovering the imposed drift much worse "
        f"than the monopole (GT r {ac.get('gt_r_hard', 0):+.3f} vs "
        f"{mb.get('gt_r_hard', 0):+.3f}). C rises as localizations collapse toward the "
        "channel grid, so a high C is not evidence of a good localizer. Distance below the "
        "control is the informative quantity, not absolute height.", "",
        "**Bins.** 245 images, each **one second** wide (~1240 spikes), sampled every 8 s; "
        "12.3% of the recording's spikes enter any image. The stride subsamples which "
        "images are compared — it does not widen them. Individual C(t,t') values are "
        "unaffected by the stride (verified to 6 decimals against the stride-2 matrices), "
        "so fits are mutually comparable; only the absolute level of C reflects 1 s images.",
        "",
        "**The third coordinate is not one physical quantity.** It is source depth for "
        "monopole kernels, spatial width for gauss, and the σ lattice for hard fits. The "
        "shared grid ends at z=160 µm, so some fits push sources past it. `outside` is the "
        "fraction of spikes landing outside the grid; rows above 5% are **bold** and their "
        "C/GT r are marked \\* — those are re-measured with z clipped into the grid, "
        "because their raw values (0.000 for the four σ=192/256 fits, whose images are "
        "entirely empty) measure grid coverage rather than localization.", ""]
    resc = [r for r in lim if r["C_hard_clamped"] is not None]
    unre = [r for r in lim if r["C_hard_clamped"] is None]
    if resc:
        head += ["**Re-measured with positions clipped into the grid:** " + ", ".join(
            f"`{r['tag']}` ({r['outside_frac']*100:.0f}% outside → C_hard "
            f"{r['C_hard_clamped']:.3f})" for r in resc), ""]
    if unre:
        head += ["**Flagged but NOT re-measured** (raw value shown, treat as a floor): "
                 + ", ".join(f"`{r['tag']}` ({r['outside_frac']*100:.0f}% outside)"
                             for r in unre), ""]
    (a.figs / "C_TABLE.md").write_text("\n".join(head + L) + "\n")

    # ---------------------------------------------------------------- figure
    FAM = [("lattice monopole", "#4c8dff", "o"), ("lattice gauss", "#e8590c", "^"),
           ("hard, fixed volume", "#f59f00", "*"), ("hard 1-of-K", "#c92a2a", "D"),
           ("grid (position x shape)", "#2f9e44", "s"), ("CP", "#845ef7", "v")]
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.9), constrained_layout=True)
    for A, key in zip(ax[:2], ("C_soft_use", "C_hard_use")):
        raw = key.replace("_use", "")
        for nm, col, mk in FAM:
            sub = [r for r in rows if r["family"] == nm and r["nmse"] is not None]
            ok = [r for r in sub if not r["limited"]]
            bad = [r for r in sub if r["limited"]]
            if ok:
                A.scatter([r["nmse"] for r in ok], [r[key] for r in ok], s=52, c=col,
                          marker=mk, alpha=.82, edgecolor="k", linewidth=.4, label=nm,
                          zorder=3)
            if bad:
                A.scatter([r["nmse"] for r in bad], [r[key] for r in bad], s=70,
                          facecolor="none", marker=mk, edgecolor=col, linewidth=1.5,
                          zorder=4)
        if ctl.get("CONTROL_anchor_only_ptp"):
            v = ctl["CONTROL_anchor_only_ptp"][raw]
            A.axhline(v, color="#c92a2a", ls="-", lw=1.8, alpha=.85)
            A.annotate(f"anchor-only collapse control {v:.3f}  (no waveform used)",
                       xy=(0.015, v), xycoords=("axes fraction", "data"), ha="left",
                       va="bottom", fontsize=8.5, color="#c92a2a", weight="bold")
        if ctl.get("BASELINE_monopole"):
            v = ctl["BASELINE_monopole"][raw]
            A.axhline(v, color="#2f6df6", ls="--", lw=1.4)
            A.annotate(f"monopole baseline {v:.3f}", xy=(0.99, v),
                       xycoords=("axes fraction", "data"), ha="right", va="bottom",
                       fontsize=8, color="#2f6df6")
        for lab, v in REF_NMSE.items():
            A.axvline(v, color="#888", ls=":", lw=1.1)
            A.annotate(lab, xy=(v, 0.01), xycoords=("data", "axes fraction"),
                       rotation=90, fontsize=7, color="#888", ha="right", va="bottom")
        A.set_xlabel("waveform reconstruction nMSE  (lower = better fit) →", fontsize=9)
        A.set_ylabel(f"mean {raw}   —   pairwise ZNCC of I_t, held out", fontsize=9)
        A.set_title(f"{raw}: what the fits get for free", fontsize=10)
        A.grid(alpha=.3); A.legend(fontsize=7.5, loc="lower right")

    A = ax[2]
    for nm, col, mk in FAM:
        sub = [r for r in rows if r["family"] == nm]
        if sub:
            A.scatter([r["C_hard_use"] for r in sub], [r["gt_r_use"] for r in sub], s=52,
                      c=col, marker=mk, alpha=.75, edgecolor="k", linewidth=.4, label=nm,
                      zorder=3)
    for tag, col, lab in (("CONTROL_anchor_only_ptp", "#c92a2a", "anchor-only control"),
                          ("BASELINE_monopole", "#2f6df6", "monopole baseline")):
        if ctl.get(tag):
            c = ctl[tag]
            A.scatter([c["C_hard"]], [c["gt_r_hard"]], s=430, marker="*", c=col,
                      edgecolor="k", linewidth=.9, zorder=6)
            A.annotate(lab, xy=(c["C_hard"], c["gt_r_hard"]), textcoords="offset points",
                       xytext=(-8, 12), fontsize=9, color=col, weight="bold", ha="right")
    rr = np.corrcoef([r["C_hard_use"] for r in rows], [r["gt_r_use"] for r in rows])[0, 1]
    A.set_title(f"C does not measure drift recovery\nacross fits r = {rr:+.3f} "
                f"(n={len(rows)}), but the control has the HIGHEST C and among the "
                f"worst recovery", fontsize=9.5)
    A.set_xlabel("mean C_hard →", fontsize=9)
    A.set_ylabel("correlation of DREDge motion with the imposed drift", fontsize=9)
    A.grid(alpha=.3); A.legend(fontsize=7.5, loc="lower left")

    nb = len(lim)
    fig.suptitle(
        "tensor fits: reconstruction vs. temporal consistency of I_t   —   "
        "every point minimised waveform MSE only, C was never in any objective\n"
        "245 one-second images sampled every 8 s"
        + (f"   ·   {nb} hollow markers fell >5% outside the grid and are shown "
           f"re-measured with z clipped" if nb else ""), fontsize=11)
    fig.savefig(a.figs / "mse_vs_C.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- console
    print(f"{len(rows)} fits tabulated -> {a.figs}/C_TABLE.md, mse_vs_C.png")
    print(f"  {nb} measurement-limited (re-measured with clipped z)")
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    print(f"\n  {'family':26s} {'n':>4s} {'C_soft':>8s} {'C_hard':>8s} {'nMSE':>8s}")
    for k, v in sorted(fam.items(), key=lambda kv: -np.mean([x["C_hard_use"]
                                                            for x in kv[1]])):
        print(f"  {k:26s} {len(v):4d} {np.mean([x['C_soft_use'] for x in v]):8.3f} "
              f"{np.mean([x['C_hard_use'] for x in v]):8.3f} "
              f"{np.mean([x['nmse'] for x in v]):8.4f}")
    for tag, c in ctl.items():
        print(f"  {tag:26s} {'—':>4s} {c['C_soft']:8.3f} {c['C_hard']:8.3f} {'—':>8s}"
              f"   GT r {c['gt_r_hard']:+.3f}")
    n = [r["nmse"] for r in rows]
    print(f"\n  corr(nMSE, C_soft) {np.corrcoef(n, [r['C_soft_use'] for r in rows])[0,1]:+.3f}"
          f"   corr(nMSE, C_hard) "
          f"{np.corrcoef(n, [r['C_hard_use'] for r in rows])[0,1]:+.3f}"
          f"   corr(C_hard, GT r) "
          f"{np.corrcoef([r['C_hard_use'] for r in rows], [r['gt_r_use'] for r in rows])[0,1]:+.3f}")
    above = sum(r["C_hard_use"] > ctl.get("CONTROL_anchor_only_ptp", {}).get("C_hard", 9)
                for r in rows)
    print(f"  fits above the anchor-only collapse control: {above}/{len(rows)}")


if __name__ == "__main__":
    main()
