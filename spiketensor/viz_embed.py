"""Low-dimensional embeddings of the per-spike coefficient vector v_s.

v_s is the model's WAVEFORM-TYPE readout: Q numbers per spike, the weights on a time basis
shared by every spike in the recording. Embedding that into 2-D asks whether waveform shape
is organised -- into discrete types, a continuum, or nothing.

DIRECTION, NOT MAGNITUDE, BY DEFAULT. ||v_s|| tracks spike amplitude and spans ~240x on
this recording, so a raw embedding is dominated by loudness and shows an amplitude gradient
rather than shape structure. Each v_s is L2-normalised first, so the embedding is over the
unit sphere in coefficient space -- which is what "type" means. `--raw` disables that.

Points are coloured by the CENTROID coordinates from the same fit (x, y, z in um) plus the
dominant basis component, so a reader can see directly whether shape structure lines up
with anatomy: bands in the y panel mean waveform type varies with depth, an unstructured y
panel means it does not.

Methods are separate files, so the browser's panel selector is the toggle between them:
    embed_pca.png    linear, deterministic, no free parameters -- the honest baseline
    embed_umap.png   neighbourhood-preserving, shows clusters PCA cannot separate
    embed_tsne.png   optional, slower; included for comparison

Embedding coordinates are cached to embed_<method>.npz so recolouring never refits.

Usage:
    python3 zncc/tensor/viz_embed.py --runs zncc/runs/lattice --figs zncc/figures/lattice
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D          # noqa: E402


def load_fit(runs: Path, tag: str, rec=None):
    """(v, centroid xyz, dominant q). Handles both npz layouts."""
    ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu", weights_only=False)
    z = np.load(runs / f"pi_{tag}.npz")
    k = z["k"].astype(np.int64)
    if "pos" in z.files:
        # centroid-anchored: mu here is already the ABSOLUTE position minus nothing --
        # hand back pos-with-anchor-removed so the caller's anchor+mu reconstruction
        # still yields the true position
        rel = z["pos"].astype(np.float32).copy()
        rel[:, :2] -= rec.anchors[rec.spike_channels][:, :2]
        return ck, z["v"], rel
    if "mu_site" in z.files:
        S = int(z["S"]); mu = z["mu_site"].astype(np.float32)[k // S]
    else:
        S = int(ck["S"]); mu = z["mu"][k].astype(np.float32)
    return ck, z["v"], mu


def embed(Vn, method, seed=0):
    t0 = time.perf_counter()
    if method == "pca":
        X = Vn - Vn.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        E = X @ vt[:2].T
    elif method == "umap":
        import umap
        E = umap.UMAP(n_neighbors=30, min_dist=0.10, n_components=2,
                      random_state=seed, verbose=False).fit_transform(Vn)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        E = TSNE(n_components=2, perplexity=40, init="pca", random_state=seed,
                 max_iter=750).fit_transform(Vn)
    else:
        raise ValueError(method)
    return np.asarray(E, np.float32), time.perf_counter() - t0


def figure(E, pos, dom, qcol, out: Path, tag, method, n, norm, secs, Q):
    """One embedding, four colourings: the three centroid axes plus shape colour.

    The fourth panel colours points by RGB(PC1..3) of the SAME coefficient space
    (local depth-block PCA, rank-equalised -- see pca_rgb_local)
    (qcol carries per-point RGB), so shape similarity maps to colour similarity."""
    lo, hi = np.percentile(E, [0.4, 99.6], axis=0)
    pad = 0.04 * (hi - lo)
    fig, ax = plt.subplots(1, 4, figsize=(19.2, 5.0), constrained_layout=True)
    panels = [("centroid x (µm)", pos[:, 0], "viridis"),
              ("centroid depth y (µm)", pos[:, 1], "viridis"),
              ("centroid z (µm, log)", np.log10(np.maximum(pos[:, 2], 1e-3)), "viridis"),
              ("shape colour: RGB of local v_s PCA 1..3", None, None)]
    for A, (lab, c, cm) in zip(ax, panels):
        if c is None:
            cc = qcol[dom] if qcol.ndim == 2 and len(qcol) <= 64 else qcol
            A.scatter(E[:, 0], E[:, 1], s=1.6, c=cc, alpha=.55, linewidths=0,
                      rasterized=True)
        else:
            v0, v1 = np.percentile(c, [1, 99])
            sc = A.scatter(E[:, 0], E[:, 1], s=1.6, c=c, cmap=cm, vmin=v0, vmax=v1,
                           alpha=.55, linewidths=0, rasterized=True)
            cb = fig.colorbar(sc, ax=A, fraction=.045, pad=.02)
            cb.set_label(lab + ("  [log10]" if "log" in lab else ""), fontsize=8)
            cb.ax.tick_params(labelsize=7)
        A.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        A.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        A.set_xticks([]); A.set_yticks([])
        A.set_title(lab, fontsize=10)
        A.set_xlabel(f"{method.upper()} 1", fontsize=8.5)
    ax[0].set_ylabel(f"{method.upper()} 2", fontsize=8.5)
    kind = "unit-normalised, shape only" if norm else "raw, includes amplitude"
    fig.suptitle(
        f"waveform-type space — {tag}\n"
        f"{method.upper()} of $v_s$ ({kind}) · Q={Q} · {n:,} spikes · {secs:.0f}s\n"
        f"colour is the CENTROID from the same fit — structure here lining up with depth "
        f"would mean waveform type varies with anatomy", fontsize=11.5)
    fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/lattice")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/lattice")
    ap.add_argument("--methods", default="pca,umap")
    ap.add_argument("--n", type=int, default=60000,
                    help="spikes embedded; UMAP on all 2.5 M is not worth the wall clock "
                         "and the structure is visible well below that")
    ap.add_argument("--raw", action="store_true",
                    help="embed v_s as-is; by default it is L2-normalised so the "
                         "embedding reflects shape rather than amplitude")
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--refig", action="store_true",
                    help="regenerate figures from cached embeddings (recolour)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rec = D.load("np1")
    anc = rec.anchors[rec.spike_channels][:, :2]
    meths = [m for m in a.methods.split(",") if m]
    tags = [f.stem[len("summary_"):] for f in sorted(a.runs.glob("summary_*.json"))
            if not a.only or a.only in f.stem]
    print(f"{len(tags)} fits × {len(meths)} methods · {a.n:,} spikes · "
          f"{'raw' if a.raw else 'unit-normalised'} v", flush=True)

    for n_, tag in enumerate(tags, 1):
        ck, V, mu = load_fit(a.runs, tag, rec)
        Q = V.shape[1]
        rng = np.random.default_rng(a.seed)
        idx = np.sort(rng.choice(len(V), min(a.n, len(V)), replace=False))
        Vs = V[idx].astype(np.float32)
        if not a.raw:
            Vs = Vs / np.maximum(np.linalg.norm(Vs, axis=1, keepdims=True), 1e-9)
        pos = np.empty((len(idx), 3), np.float32)
        pos[:, :2] = anc[idx] + mu[idx][:, :2]
        pos[:, 2] = mu[idx][:, 2]
        dom = np.arange(len(idx))
        from spiketensor.viz_centroid_basis import pca_rgb_local
        qcol, _ = pca_rgb_local(V[idx], pos[:, 1])

        d = a.figs / tag; d.mkdir(parents=True, exist_ok=True)
        for m in meths:
            cache, png = d / f"embed_{m}.npz", d / f"embed_{m}.png"
            if cache.exists() and png.exists() and not a.redo and not a.refig:
                print(f"  [{n_}/{len(tags)}] {tag} {m}: cached", flush=True)
                continue
            if cache.exists() and a.refig and not a.redo:
                # recolour from the cached embedding without refitting
                zc = np.load(cache)
                figure(zc["E"], pos, dom, qcol, png, tag, m, len(idx),
                       not a.raw, 0.0, Q)
                print(f"  [{n_}/{len(tags)}] {tag} {m}: recoloured", flush=True)
                continue
            try:
                E, secs = embed(Vs, m, a.seed)
            except Exception as e:
                print(f"  [{n_}/{len(tags)}] {tag} {m}: FAILED {type(e).__name__}: {e}",
                      flush=True)
                continue
            np.savez_compressed(cache, E=E, idx=idx.astype(np.int64), pos=pos,
                                dom=dom.astype(np.int16))
            figure(E, pos, dom, qcol, png, tag, m, len(idx), not a.raw, secs, Q)
            print(f"  [{n_}/{len(tags)}] {tag} {m}: {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
