"""The README headline schematic, drawn end to end from a real fit.

Five panels follow one spike through the factorization: the measurement, the spatial
superposition the model selected for it, the codebook shape each source chose, those
shapes after their learned lags, and the resulting reconstruction.  Nothing is
illustrative -- every curve is fitted output.

    python3 docs/make_schematic.py --runs <runs> --tag prior2_shift_M64_R4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from matplotlib.colors import PowerNorm               # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                     # noqa: E402
from spiketensor.fit import load_batch                # noqa: E402
from spiketensor.source_core import selected_footprints                    # noqa: E402
from spiketensor.source_figures import (              # noqa: E402
    _run_footprints, _source_waves, load_run)

HEADLINE_TAG = "prior2_shift_M64_R4"
HEADLINE_SPIKE = 1278403
COL = ["#e8590c", "#2f9e44", "#4c8dff", "#845ef7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--tag", default=HEADLINE_TAG)
    ap.add_argument("--spike", type=int, default=HEADLINE_SPIKE)
    ap.add_argument("--out", type=Path, default=REPO / "docs/panels/schematic.png")
    a_ = ap.parse_args()

    rec = D.load("np1")
    state, summary, cb = load_run(a_.runs, a_.tag)
    row = int(np.searchsorted(state["spike_index"], a_.spike))
    dic = _run_footprints(summary, cb, rec)
    mu = np.asarray(dic.candidate_pos, np.float32)
    sig = np.asarray([float(p[1][0]) for p in cb["dictionary_metadata"]["profiles"]],
                     np.float32)
    shift_a = np.asarray(state["anchor_shift"], np.float32)
    k = int(rec.spike_channels[a_.spike])
    off = rec.channel_offsets().astype(np.float32) - shift_a[:, None]
    Y, boff = load_batch(rec, np.array([a_.spike]), off, "cpu")
    Yn, offs = Y[0].numpy(), boff[0].numpy()

    rows_arr = np.array([row])
    idx = torch.as_tensor(state["source_index"][rows_arr], dtype=torch.long)
    conf = torch.as_tensor(dic.cfg_id_by_channel[[k]], dtype=torch.long)
    coef = torch.as_tensor(state["source_coeff"][rows_arr], dtype=torch.float32)
    H = selected_footprints(torch.as_tensor(dic.footprints), conf, idx)
    waves = _source_waves(state, rows_arr, coef)          # amplitude and shift applied
    fit = (H.transpose(1, 2).unsqueeze(-1) * waves.unsqueeze(2)).sum(1)[0].numpy()

    act = state["source_index"][row] >= 0
    srcs = state["source_index"][row][act]
    atoms = state["source_temporal_atom"][row][act]
    taus = state["source_shift"][row][act]
    amps = state["source_amp"][row][act]
    order = np.argsort(-amps)
    srcs, atoms, taus, amps = srcs[order], atoms[order], taus[order], amps[order]
    R = len(srcs)
    OM = np.asarray(state["omega"])
    ASG = np.asarray(state["atom_prototype"])
    T = OM.shape[1]
    tt = np.arange(T) / 30.0

    fig = plt.figure(figsize=(17.6, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.05, .95, .95, 1.0])

    def draw_spike(ax, sig_, colour, title, ref=None):
        amp = 11 / max(np.abs(Yn).max(), 1e-9)
        t = np.arange(sig_.shape[1]) * 0.5
        for c in range(sig_.shape[0]):
            if ref is not None:
                ax.plot(offs[c, 0] + t, offs[c, 1] + ref[c] * amp, color="#e03131",
                        lw=.7, alpha=.45)
            ax.plot(offs[c, 0] + t, offs[c, 1] + sig_[c] * amp, color=colour, lw=1.0)
            ax.plot(offs[c, 0], offs[c, 1], "s", ms=3, mfc="none", mec="#999", mew=.6)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # A -- the measurement, contacts in their true geometry
    draw_spike(fig.add_subplot(gs[0]), Yn, "#e03131",
               "observed spike $Y_s$\n10 contacts $\\times$ 90 samples")

    # B -- the selected kernels, superposed, over the contact layout
    ax = fig.add_subplot(gs[1])
    sx, sy = mu[srcs.astype(int), 0], mu[srcs.astype(int), 1]
    lim_x = (min(offs[:, 0].min(), sx.min()) - 40, max(offs[:, 0].max(), sx.max()) + 40)
    lim_y = (min(offs[:, 1].min(), sy.min()) - 35, max(offs[:, 1].max(), sy.max()) + 35)
    GX, GY = np.meshgrid(np.linspace(*lim_x, 260), np.linspace(*lim_y, 260))
    field = np.zeros_like(GX)
    for r in range(R):
        n = int(srcs[r])
        d2 = (GX - mu[n, 0]) ** 2 + (GY - mu[n, 1]) ** 2 + mu[n, 2] ** 2
        field += float(amps[r]) * sig[n] / np.sqrt(d2 + sig[n] ** 2)
    ax.imshow(field, origin="lower", extent=[*lim_x, *lim_y], cmap="magma",
              aspect="equal", interpolation="bilinear",
              norm=PowerNorm(0.7, vmin=0, vmax=float(np.percentile(field, 99.0))))
    ax.scatter(offs[:, 0], offs[:, 1], s=26, marker="s", facecolors="none",
               edgecolors="w", linewidths=.9)
    for r in range(R):
        n = int(srcs[r])
        ax.plot(mu[n, 0], mu[n, 1], "o", ms=9, mfc="none", mec=COL[r % 4], mew=2.0)
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("spatial footprint $\\sum_r a_r\\,g_{n_r}$\n"
                 "superposed kernels; squares = contacts, rings = sources", fontsize=10)

    # C -- each source's chosen atom, against the cone of the codebook it came from
    ax = fig.add_subplot(gs[2])
    for r in range(R):
        fam = np.flatnonzero(ASG == ASG[int(atoms[r])])
        for q in fam[:22]:
            ax.plot(tt, OM[q] - r * 0.55, color=COL[r % 4], lw=.35, alpha=.16, zorder=1)
        ax.plot(tt, OM[int(atoms[r])] - r * 0.55, color=COL[r % 4], lw=1.9, zorder=3)
        ax.text(3.02, -r * 0.55, f"$q_{{{r + 1}}}$={int(atoms[r])}", fontsize=8,
                color=COL[r % 4], va="center")
    ax.set_yticks([])
    ax.set_xlabel("ms", fontsize=8.5)
    ax.set_xlim(0, 3.5)
    ax.set_title("chosen shape per source\nfrom the shared codebook $\\{\\psi_q\\}^M$",
                 fontsize=10)

    # D -- the same atoms after their learned lags
    ax = fig.add_subplot(gs[3])
    for r in range(R):
        psi, tau = OM[int(atoms[r])], int(taus[r])
        v = np.zeros(T)
        if tau >= 0:
            v[tau:] = psi[:T - tau] if tau else psi
        else:
            v[:T + tau] = psi[-tau:]
        v = v / max(np.linalg.norm(v), 1e-9)
        ax.plot(tt, psi - r * 0.55, color=COL[r % 4], lw=.7, alpha=.35, ls="--")
        ax.plot(tt, v - r * 0.55, color=COL[r % 4], lw=1.9)
        ax.text(3.02, -r * 0.55, f"$\\tau_{{{r + 1}}}$={tau:+d}", fontsize=8,
                color=COL[r % 4], va="center")
    ax.set_yticks([])
    ax.set_xlabel("ms", fontsize=8.5)
    ax.set_xlim(0, 3.5)
    ax.set_title("shifted by its learned lag\n"
                 "$S_{\\tau_r}\\psi_{q_r}$ (dashed = unshifted)", fontsize=10)

    # E -- the reconstruction, over the measurement it is fitting
    rel = float(np.square(Yn - fit).sum() / max(np.square(Yn).sum(), 1e-9))
    draw_spike(fig.add_subplot(gs[4]), fit, "#2f9e44",
               "reconstruction $\\hat Y_s=\\sum_r a_r g_{n_r}"
               "(S_{\\tau_r}\\psi_{q_r})^{\\!\\top}$\n"
               f"measured in red · unexplained {rel:.2f}", ref=Yn)

    a_.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a_.out, dpi=170, bbox_inches="tight")
    print(f"  wrote {a_.out} | R = {R} | atoms {atoms.tolist()} "
          f"| lags {taus.tolist()} | rel.err {rel:.3f}", flush=True)


if __name__ == "__main__":
    main()
