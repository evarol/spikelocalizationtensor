"""Per-run loss-convergence panels, plus a family overlay.

Every fit already logged its trajectory (summary_*.json -> "history"), so nothing needs
refitting. Two things about what that curve IS, both of which matter for reading it:

WHAT IS PLOTTED IS RECONSTRUCTION ONLY, NOT RECONSTRUCTION + PENALTY. In the soft models
the regularizer never enters the outer objective: pi is solved to optimality under the
L1 / entropy penalty in a no-gradient inner block minimization (solve_pi, solve_pi_grid,
solve_pi_grid_ent), and the codebook then descends the plain reconstruction residual
evaluated AT that penalized pi (fit.py:127-140). So the penalty shows up as a level shift
of the whole curve -- more penalty, worse reconstruction -- never as a term in it. The
hard fits have no penalty at all: 1-of-K is a constraint, and hard_assign picks the site
in closed form (fit_hard.py:118). Their curve is the pure reconstruction objective.

RESOLUTION IS COARSE FOR MOST RUNS. History is logged every 50 steps, and 197 of the 244
runs were fit for 250-300 steps, so they have only 6-7 points. The 47 longer runs (1200
or 2500 steps) have 25-51. The curves show the trajectory and where it flattened; they do
not resolve fine convergence behaviour, and getting that would require refitting.

The last logged value is a minibatch estimate; the final full-data nMSE is separate and is
marked on each panel (median gap 0.0021, max 0.0127).

Usage:
    python3 zncc/tensor/convergence.py            # per-run + overlay
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

REF = {"free rank-1 oracle": 0.1029, "per-slot mean": 0.3189}
FAM = [("lattice monopole", "#4c8dff"), ("lattice gauss", "#e8590c"),
       ("hard, fixed volume", "#f59f00"), ("hard 1-of-K", "#c92a2a"),
       ("grid (position x shape)", "#2f9e44"), ("CP", "#845ef7")]


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


def penalty_of(tag):
    me, ml = re.search(r"_ent([0-9.]+)", tag), re.search(r"_l1([0-9.]+)", tag)
    if tag.startswith("lat"):
        return "none (1-of-K constraint)", 0.0
    if tag.startswith(("hard_", "hg_")):
        return "none (1-of-K constraint)", 0.0
    if me:
        return "entropy", float(me.group(1))
    if ml:
        return "L1", float(ml.group(1))
    return "none", 0.0


def panel(out: Path, tag, s):
    h = s["history"]
    # schema tolerance: some fitters log (iteration, objective_after,
    # wall_seconds) instead of (step, nmse, wall_s). An objective is not an nMSE, so
    # those runs get an objective trajectory without the nMSE reference lines.
    is_obj = "nmse" not in h[0]
    st = np.array([r.get("step", r.get("iteration", i + 1))
                   for i, r in enumerate(h)], float)
    ms = np.array([r.get("nmse", r.get("objective_after", np.nan)) for r in h], float)
    wl = np.array([r.get("wall_s", r.get("wall_seconds", np.nan)) for r in h], float)
    if np.isnan(wl).any():
        wl = np.nancumsum(np.where(np.isnan(wl), 0.0, wl))
    used = ([r["used"] for r in h] if "used" in h[0] else None)
    pen, lvl = penalty_of(tag)
    full = s.get("full_nmse")

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    for A, x, xl in ((ax[0], st, "optimizer step"), (ax[1], wl, "wall-clock (s)")):
        lbl = "EM objective (higher = better)" if is_obj else "minibatch nMSE"
        A.plot(x, ms, "-o", color="#4c8dff", ms=4, lw=1.6, label=lbl)
        if full is not None and not is_obj:
            A.axhline(full, color="#c92a2a", ls="--", lw=1.3,
                      label=f"final full-data nMSE {full:.4f}")
        if not is_obj:
            for lab, v in REF.items():
                A.axhline(v, color="#888", ls=":", lw=1.0)
                A.annotate(lab, xy=(0.99, v), xycoords=("axes fraction", "data"),
                           ha="right", va="bottom", fontsize=7, color="#888")
        A.set_xlabel(xl, fontsize=9)
        A.set_ylabel("EM objective" if is_obj
                     else "reconstruction nMSE (DC-removed target)", fontsize=9)
        A.grid(alpha=.3)
        if not is_obj:
            A.set_ylim(0, max(0.36, float(ms.max()) * 1.06))
    if used is not None:
        B = ax[0].twinx()
        B.plot(st, used, "-s", color="#2f9e44", ms=3, lw=1.1, alpha=.75)
        B.set_ylabel("codebook positions used per spike", color="#2f9e44", fontsize=8.5)
        B.tick_params(axis="y", labelcolor="#2f9e44", labelsize=7)
    ax[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        f"loss convergence — {tag}\n"
        f"{family(tag)} · penalty: {pen}"
        + (f" {lvl:g}" if lvl else "")
        + f" · {len(h)} logged points over {int(st[-1])} steps"
        + "\nplotted curve is RECONSTRUCTION ONLY — the penalty is solved exactly in the "
          "inner π step and shifts the curve's level, it is not a term in it",
        fontsize=9.5)
    fig.savefig(out / "convergence.png", dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/tensor")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/tensor")
    a = ap.parse_args()

    rows = []
    for f in sorted(a.runs.glob("summary_*.json")):
        tag = f.stem[len("summary_"):]
        s = json.loads(f.read_text())
        if not s.get("history"):
            continue
        d = a.figs / tag; d.mkdir(parents=True, exist_ok=True)
        try:
            panel(d, tag, s)
        except Exception as e:
            print(f"  skip {tag}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append((tag, s))
    print(f"wrote {len(rows)} per-run convergence.png")

    # ---- overlay: with 6-7 points per run the family shape is more legible than any
    # single curve, and the penalty-level shift is only visible across runs anyway
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for tag, s in rows:
        h = s["history"]
        if "nmse" not in h[0]:
            continue
        st = [r["step"] for r in h]; ms = [r["nmse"] for r in h]
        col = dict(FAM)[family(tag)]
        ax[0].plot(st, ms, "-", color=col, lw=.8, alpha=.42)
        ax[1].plot(np.asarray(st) / st[-1], ms, "-", color=col, lw=.8, alpha=.42)
    for nm, col in FAM:
        ax[0].plot([], [], color=col, lw=2, label=nm)
    for A, xl in ((ax[0], "optimizer step"), (ax[1], "fraction of that run's steps")):
        for lab, v in REF.items():
            A.axhline(v, color="#888", ls=":", lw=1.0)
            A.annotate(lab, xy=(0.99, v), xycoords=("axes fraction", "data"),
                       ha="right", va="bottom", fontsize=7, color="#888")
        A.set_xlabel(xl, fontsize=9); A.set_ylabel("reconstruction nMSE", fontsize=9)
        A.grid(alpha=.3); A.set_ylim(0, .40)
    ax[0].legend(fontsize=8); ax[0].set_title("all 244 trajectories", fontsize=10)
    ax[1].set_title("time-normalised — do they flatten?", fontsize=10)

    # penalty level vs where it converged: this is where the regularizer actually shows up
    A = ax[2]
    for nm, col in FAM:
        sub = [(penalty_of(t)[1], s["full_nmse"]) for t, s in rows
               if family(t) == nm and s.get("full_nmse") is not None]
        if sub:
            A.scatter([x for x, _ in sub], [y for _, y in sub], s=42, c=col, alpha=.75,
                      edgecolor="k", linewidth=.4, label=nm)
    A.set_xscale("symlog", linthresh=1e-3)
    A.set_xlabel("penalty level (L1 ρ or entropy β; 0 = unregularized)", fontsize=9)
    A.set_ylabel("final full-data nMSE", fontsize=9)
    A.set_title("the penalty is a level shift, not a curve term", fontsize=10)
    A.grid(alpha=.3); A.legend(fontsize=8)
    fig.suptitle("loss convergence across all fits — reconstruction nMSE only; π is solved "
                 "to optimality under the penalty in an inner no-gradient step", fontsize=11)
    fig.savefig(a.figs / "convergence_all.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.figs}/convergence_all.png")


if __name__ == "__main__":
    main()
