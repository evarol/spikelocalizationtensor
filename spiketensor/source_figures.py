"""Standard and multipole-specific static panels for one isolated run."""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                       # noqa: E402
from spiketensor.fit import load_batch                                # noqa: E402
from spiketensor.drift import correction                              # noqa: E402
from spiketensor.example_spikes import (                              # noqa: E402
    N_EXAMPLE_SPIKES, example_spike_ids)
from spiketensor.dictionary import (                        # noqa: E402
    _learned_footprints,
    _unique_configs,
    load_dictionary,
)
from spiketensor.source_core import selected_footprints             # noqa: E402
from spiketensor.state_io import load_multipole_state             # noqa: E402
from spiketensor.viz_centroid_basis import (                           # noqa: E402
    FULL_Y,
    ZOOM_Y,
    jittered,
    save_basis_raster,
    save_centroid,
    save_density_raster,
    shuffled_palette,
)

XY_LIM = (-70.0, 130.0)
Q_CMAP = "viridis"


ASPECT_TOL = 0.02


def _match_standard_canvas(path: Path) -> None:
    """Normalise TEXT-driven canvas jitter to the authoritative panel size.

    The point of this is that tight-layout trims to the text, so two runs of the same
    figure can differ by a few pixels purely because one tag is longer. Those are the
    same picture and should share a canvas.

    It must NOT be applied when the layout genuinely differs. `basis.png` has
    ceil(M/4) rows of subplots and `spike_decomposition.png` has R+3, so a model with
    more shape terms or more sources produces a legitimately TALLER figure. Forcing that
    into the reference row's aspect ratio with ImageOps.fit (which crops to fill) silently
    cut the top row of atoms and the title off every such panel -- the bug this guard
    fixes. When the aspect ratios differ by more than ASPECT_TOL the image is left alone;
    the browser scales panels by width, so an honest extra row of subplots costs nothing.

    Within tolerance the image is PADDED (letterboxed), never cropped, so no content can
    be lost by this function again.
    """
    reference = REPO / "zncc/figures/multipole/learned_m2_r2" / path.name
    if not reference.exists():
        return
    from PIL import Image, ImageOps
    with Image.open(reference) as image:
        target = image.size
    with Image.open(path) as image:
        if image.size == target:
            return
        ar, ar_ref = image.width / image.height, target[0] / target[1]
        if abs(ar - ar_ref) / ar_ref > ASPECT_TOL:
            return                      # genuinely different layout: keep it intact
        standardized = ImageOps.pad(image.convert("RGB"), target,
                                    method=Image.Resampling.LANCZOS,
                                    color="white", centering=(.5, .5))
        standardized.save(path)


def _state_path(runs: Path, tag: str) -> Path:
    return runs / f"multipole_{tag}.npz"


def load_run(runs: Path, tag: str):
    try:
        state = load_multipole_state(_state_path(runs, tag))
    except ValueError:
        # C5 is deliberately nonseparable: it exposes atom_coeff and an
        # explicit shape_feature instead of fabricating an ordinary M2 row.
        with np.load(_state_path(runs, tag), allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        if "atom_coeff" not in state or "shape_feature" not in state:
            raise
    summary = json.loads((runs / f"summary_{tag}.json").read_text())
    codebook = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu",
                          weights_only=False)
    return state, summary, codebook


def flattened_sources(state: dict, rec, max_points: int = 0,
                      seed: int = 17) -> dict[str, np.ndarray]:
    """Flatten active source slots while conserving each spike's model amplitude."""
    active = state["source_index"] >= 0
    parent, slot = np.nonzero(active)
    if max_points and len(parent) > max_points:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(parent), max_points, replace=False))
        parent, slot = parent[take], slot[take]
    spike = state["spike_index"][parent]
    if "atom_coeff" in state:
        # C5 is nonseparable across scale atoms.  Categorical basis colour uses
        # the dominant shared temporal coordinate after pooling atom magnitudes;
        # PCA panels use the full explicit shape_feature instead.
        coeff = np.linalg.vector_norm(state["atom_coeff"][parent, slot], axis=1)
    else:
        coeff = state["source_coeff"][parent, slot]
    return {
        "parent": parent,
        "slot": slot,
        "spike": spike,
        "pos": state["source_pos"][parent, slot],
        "amp": state["source_amp"][parent, slot],
        "q": state["source_weight"][parent, slot],
        "dom": np.argmax(np.abs(coeff), axis=1).astype(np.int16),
        "time": rec.spike_times[spike].astype(np.float64) / rec.fs,
    }


def _sample_rows(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def _title(summary: dict) -> str:
    return f"{summary['dictionary']['kind']} / {summary['model']} / {summary['tag']}"


def fig_components(state: dict, summary: dict, out: Path) -> None:
    omega = state["omega"]
    cand = state["candidate_pos"]
    active = state["source_index"] >= 0
    ids = state["source_index"][active]
    mass = np.bincount(ids, weights=state["source_weight"][active],
                       minlength=len(cand))
    fig, ax = plt.subplots(1, 4, figsize=(17.0, 4.2), constrained_layout=True)
    tt = np.arange(omega.shape[1]) / 30.0
    for q, row in enumerate(omega):
        ax[0].plot(tt, row, lw=1.1, label=f"q{q}")
    ax[0].set(xlabel="time (ms)", ylabel="basis amplitude",
              title=f"orthonormal temporal basis (M={len(omega)})")
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
    sc = ax[1].scatter(cand[:, 0], cand[:, 1], c=np.log1p(mass), s=8,
                       cmap="magma", rasterized=True)
    fig.colorbar(sc, ax=ax[1]).set_label("log(1 + contribution mass)")
    ax[1].set(xlabel="candidate x (µm)", ylabel="candidate y (µm)",
              title="spatial dictionary usage")
    ax[2].scatter(cand[:, 2], mass, s=7, alpha=.5, rasterized=True)
    ax[2].set_xscale("symlog", linthresh=.5); ax[2].set_yscale("symlog", linthresh=1)
    ax[2].set(xlabel="candidate z / scale (µm)", ylabel="contribution mass",
              title="depth/scale usage")
    used = np.sort(mass[mass > 0])[::-1]
    if len(used):
        ax[3].plot(np.arange(1, len(used) + 1), np.cumsum(used) / used.sum())
    ax[3].set_xscale("log")
    ax[3].set(xlabel="used candidates, ranked", ylabel="cumulative contribution",
              title=f"{len(used):,}/{len(cand):,} candidates active")
    ax[3].grid(alpha=.3)
    fig.suptitle(f"components — {_title(summary)}")
    fig.savefig(out / "components.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def fig_basis(state: dict, summary: dict, flat: dict, out: Path) -> None:
    omega = state["omega"]
    m = len(omega)
    cols = min(4, m); rows = int(math.ceil(m / cols))
    fig = plt.figure(figsize=(3.2 * cols, 2.2 * rows + 4.2), constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [1.6])
    palette = shuffled_palette(m)
    usage = np.bincount(flat["dom"], weights=flat["amp"], minlength=m)
    for q in range(m):
        a = fig.add_subplot(gs[q // cols, q % cols])
        a.plot(np.arange(omega.shape[1]) / 30.0, omega[q], color=palette[q], lw=1.4)
        a.axhline(0, color=".6", lw=.5)
        a.set_title(f"q{q} · {100 * usage[q] / max(usage.sum(), 1e-12):.1f}% amp", fontsize=8)
        a.tick_params(labelsize=7)
    a = fig.add_subplot(gs[rows, :])
    take = _sample_rows(len(flat["pos"]), 80000, 19)
    for q in range(m):
        use = take[flat["dom"][take] == q]
        if len(use):
            a.scatter(flat["pos"][use, 0], flat["pos"][use, 1], s=1.2,
                      color=palette[q], alpha=.35, label=f"q{q}", rasterized=True)
    a.set_xlim(*XY_LIM); a.set_ylim(*FULL_Y)
    a.set(xlabel="lateral x (µm)", ylabel="probe depth y (µm)",
          title="active-source cloud colored by dominant temporal coefficient")
    a.legend(fontsize=6, ncol=min(8, m), markerscale=4)
    fig.suptitle(f"temporal basis — {_title(summary)}")
    fig.savefig(out / "basis.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def fig_usage(state: dict, summary: dict, out: Path) -> None:
    support = state["support_size"].astype(int)
    active = state["source_index"] >= 0
    ids = state["source_index"][active]
    count = np.bincount(ids, minlength=len(state["candidate_pos"]))
    ranked = np.sort(count[count > 0])[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(16.0, 4.0), constrained_layout=True)
    bins = np.arange(support.max() + 2) - .5
    ax[0].hist(support, bins=bins, rwidth=.85)
    ax[0].set_xticks(np.arange(1, support.max() + 1))
    ax[0].set(xlabel="effective support", ylabel="spikes", title="source count")
    if len(ranked):
        ax[1].plot(np.arange(1, len(ranked) + 1), np.cumsum(ranked) / ranked.sum())
    ax[1].set_xscale("log"); ax[1].grid(alpha=.3)
    ax[1].set(xlabel="candidate rank", ylabel="cumulative active slots",
              title="candidate concentration")
    amp = state["source_amp"][active]
    ax[2].hist(np.log10(np.maximum(amp, 1e-8)), bins=70)
    ax[2].set(xlabel="log10 source coefficient norm", ylabel="sources",
              title="source amplitudes")
    q = state["source_weight"][active]
    ax[3].hist(q, bins=np.linspace(0, 1, 51))
    ax[3].set(xlabel="normalized contribution q", ylabel="sources",
              title="contribution distribution")
    fig.suptitle(f"usage — {_title(summary)}")
    fig.savefig(out / "usage.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def _run_footprints(summary: dict, codebook: dict, rec, device: str = "cpu"):
    if codebook.get("adapter") and codebook.get("footprints") is not None:
        return SimpleNamespace(
            kind="learned",
            candidate_pos=np.asarray(codebook["candidate_pos"], np.float32),
            footprints=np.asarray(codebook["footprints"], np.float32),
            cfg_id_by_channel=np.asarray(codebook["cfg_id_by_channel"], np.int32),
            anchor_shift=np.asarray(codebook["anchor_shift"], np.float32),
        )
    base = load_dictionary(REPO / "zncc/runs/lattice", summary["dictionary"]["tag"],
                           device=device, cache=REPO / "zncc/runs/multipole/cache")
    if summary["dictionary"]["kind"] != "learned":
        return base
    pos = np.asarray(codebook["candidate_pos"], np.float32)
    profiles = codebook["dictionary_metadata"]["profiles"]
    sigma = np.asarray([float(p[1][0]) for p in profiles], np.float32)
    off = rec.channel_offsets().astype(np.float32) - base.anchor_shift[:, None]
    cfg, cfg_id = _unique_configs(off)
    base.candidate_pos = pos
    base.footprints = _learned_footprints(
        pos, sigma, cfg, str(codebook["dictionary_metadata"]["kernel"]), device)
    base.cfg_id_by_channel = cfg_id
    return base


@functools.lru_cache(maxsize=64)
def _load_adapter_module(model_path: str, study: str, model_kind: str):
    """Load one frozen exploratory spatial module for faithful panel rendering."""
    if study == "generalized_spatial":
        from spiketensor.generalized_spatial.families import GeneralizedSpatialFamily
        from spiketensor.generalized_spatial.pilot import load_warm_start
        warm = load_warm_start()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        module = GeneralizedSpatialFamily(checkpoint["family"], warm.mu, warm.sigma)
        module.load_state_dict(checkpoint["state_dict"])
    else:
        from spiketensor.colocated_scale_mixture.pilot import load_model
        module = load_model(Path(model_path), "cpu")
        if model_kind == "continuous_scale":
            from spiketensor.colocated_scale_mixture.model import ScaleMixtureDictionary
            from spiketensor.generalized_spatial.pilot import load_warm_start
            warm = load_warm_start()
            base = module
            module = ScaleMixtureDictionary(
                warm.mu, warm.sigma, hierarchy="c4", bank="gaussian",
                n_scales=4, scale_mode="relative", learn_scales=False)
            with torch.no_grad():
                module.profile_logits.copy_(base.profile_logits[:1])
                module.raw_scale_edges.copy_(base.raw_scale_edges)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _adapter_spatial_field(codebook: dict, state: dict, row: int, slot: int,
                           first: np.ndarray, depth: np.ndarray,
                           plane: str) -> np.ndarray:
    """Evaluate the fitted spatial family on an x-y or z-y display plane."""
    adapter = codebook.get("adapter")
    if not adapter:
        from spiketensor.viz_lattice import kern_img
        candidate = int(state["source_index"][row, slot])
        kernel, pars = codebook["dictionary_metadata"]["profiles"][candidate]
        return kern_img(kernel, first, depth,
                        np.asarray(codebook["candidate_pos"])[candidate], tuple(pars))
    module = _load_adapter_module(
        adapter["model_path"], adapter["study"], adapter["model_kind"])
    candidate = int(state["source_index"][row, slot])
    if plane == "xy":
        points = np.stack([first.ravel(), depth.ravel(),
                           np.zeros(first.size)], axis=1)
    elif plane == "zy":
        points = np.stack([np.zeros(first.size), depth.ravel(),
                           first.ravel()], axis=1)
    else:
        raise ValueError("plane must be xy or zy")
    x = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        if adapter["study"] == "generalized_spatial":
            site = candidate
            z = module.mu[site].expand(len(x), -1)
            scale = module.site_scale()[site].expand(len(x))
            values = module._raw_from_points(x, z, scale)
        else:
            if adapter["model_kind"] == "finite_scale":
                cand = torch.tensor([candidate], dtype=torch.long)
                site, profile = module.decode_candidates(cand)
                site = int(site[0]); profile = int(profile[0])
                alpha = module._alpha_for(torch.tensor([site]),
                                          torch.tensor([profile]))[0]
            else:
                site = candidate
                if adapter["model_kind"] == "continuous_scale":
                    alpha = torch.as_tensor(state["source_alpha"][row, slot])
                else:
                    coefficient = torch.as_tensor(state["atom_coeff"][row, slot])
                    alpha = torch.linalg.vector_norm(coefficient, dim=-1)
                    alpha = alpha / alpha.sum().clamp_min(1e-8)
            radius = torch.linalg.vector_norm(x - module.mu[site], dim=-1)
            sites = torch.full((len(x),), site, dtype=torch.long)
            values = (module.radial_atoms(radius, sites) * alpha).sum(-1)
    return values.reshape(first.shape).cpu().numpy()


def _source_waves(state: dict, rows: np.ndarray, coef: torch.Tensor) -> torch.Tensor:
    """Per-source temporal waveform, WITH the learned time shift applied.

    Shift-invariant models (M4/M5) store the shape as a one-hot `source_coeff` over the M
    atoms plus a separate integer `source_shift`. Reconstructing as `coef @ omega` draws
    every source at ZERO lag, which made the reconstruction panels look like a bad fit
    even though the fit was good -- the panel's residual disagreed with the stored
    per-spike sse by up to 4x.

    The shifted atom is rebuilt exactly as `onehot_shift_fit.shift_bank` builds it (zero
    padding, then renormalisation to unit norm) and scaled by the stored amplitude,
    because that is the atom the amplitude was fitted against.
    """
    omega = torch.as_tensor(state["omega"])
    if "source_shift" not in state:
        return coef @ omega
    T = omega.shape[1]
    shift = np.asarray(state["source_shift"])[rows]
    amp = coef.abs().amax(2)                       # one-hot: the amplitude
    which = coef.abs().argmax(2)                   # ... and which atom
    out = torch.zeros(coef.shape[0], coef.shape[1], T)
    for b in range(out.shape[0]):
        for r in range(out.shape[1]):
            if float(amp[b, r]) == 0.0:
                continue
            tau, q = int(shift[b, r]), int(which[b, r])
            v = torch.zeros(T)
            if tau >= 0:
                v[tau:] = omega[q, :T - tau] if tau else omega[q]
            else:
                v[:T + tau] = omega[q, -tau:]
            out[b, r] = amp[b, r] * v / v.norm().clamp_min(1e-8)
    return out


def reconstruction_examples(state: dict, summary: dict, codebook: dict, rec,
                            n: int = N_EXAMPLE_SPIKES, seed: int = 23):
    # The SAME spikes as every other model in the browser (see example_spikes.py).
    # This deliberately replaces the old multi-source-preferring draw: selecting on
    # support_size made each model show a different, self-flattering set of spikes,
    # so the reconstruction panels could not be compared across models.
    ids = example_spike_ids(rec.n_spikes, n)
    order = np.argsort(state["spike_index"])
    rows = order[np.searchsorted(state["spike_index"], ids, sorter=order)]
    spike = state["spike_index"][rows]
    if not np.array_equal(spike, ids):
        raise ValueError("state does not cover the canonical example spikes")
    adapter = codebook.get("adapter")
    anchor_shift = np.asarray(codebook.get("anchor_shift", np.zeros((384, 2))),
                              np.float32)
    off = rec.channel_offsets().astype(np.float32) - anchor_shift[:, None]
    Y, batch_off = load_batch(rec, spike, off, "cpu")
    idx = torch.as_tensor(state["source_index"][rows], dtype=torch.long)
    if adapter and adapter["model_kind"] in {"continuous_scale", "atom_specific"}:
        module = _load_adapter_module(
            adapter["model_path"], adapter["study"], adapter["model_kind"])
        offsets = batch_off
        if adapter["model_kind"] == "continuous_scale":
            alpha = torch.as_tensor(state["source_alpha"][rows], dtype=torch.float32)
            H, _ = module.selected_from_alpha(offsets, idx, alpha)
            coefficients = torch.as_tensor(state["source_coeff"][rows],
                                             dtype=torch.float32)
            waves = coefficients @ torch.as_tensor(state["omega"])
            source = H.transpose(1, 2).unsqueeze(-1) * waves.unsqueeze(2)
        else:
            b, r = idx.shape
            repeated = offsets[:, None].expand(b, r, offsets.shape[1], 2).reshape(
                b * r, offsets.shape[1], 2)
            atoms = module.atoms_selected(repeated, idx.clamp_min(0).reshape(-1)).reshape(
                b, r, offsets.shape[1], module.n_atoms)
            atoms = atoms / torch.linalg.vector_norm(atoms, dim=2,
                                                      keepdim=True).clamp_min(1e-8)
            atoms = atoms * (idx >= 0)[:, :, None, None]
            coefficients = torch.as_tensor(state["atom_coeff"][rows],
                                             dtype=torch.float32)
            atom_waves = coefficients @ torch.as_tensor(state["omega"])
            source = torch.einsum("brca,brat->brct", atoms, atom_waves)
    else:
        dictionary = _run_footprints(summary, codebook, rec)
        conf = torch.as_tensor(
            dictionary.cfg_id_by_channel[rec.spike_channels[spike]], dtype=torch.long)
        coef = torch.as_tensor(state["source_coeff"][rows], dtype=torch.float32)
        H = selected_footprints(torch.as_tensor(dictionary.footprints), conf, idx)
        waves = _source_waves(state, rows, coef)
        source = H.transpose(1, 2).unsqueeze(-1) * waves.unsqueeze(2)
    total = source.sum(1)
    return rows, Y.numpy(), total.numpy(), source.numpy(), batch_off.numpy()


def fig_spikes(state: dict, summary: dict, codebook: dict, rec, out: Path) -> None:
    """Render multipole examples in the same four-row idiom as lattice models."""
    rows, Y, total, source, offs = reconstruction_examples(
        state, summary, codebook, rec)
    n = len(rows)
    fig, ax = plt.subplots(
        4, n, figsize=(3.0 * n, 12.0), constrained_layout=True, squeeze=False,
        gridspec_kw={"height_ratios": [1.25, .62, 1.2, .55]})
    amp = 16.0 / max(1e-6, float(np.abs(Y).max()))
    lim = 160.0
    gx = np.linspace(-lim, lim, 141)
    XX, YY = np.meshgrid(gx, gx, indexing="ij")
    candidate_pos = np.asarray(codebook["candidate_pos"])
    colours = ("#4c8dff", "#ff922b", "#12b886", "#ae3ec9")
    shared_waveform = str(summary["model"]).startswith("m1")

    for j, row in enumerate(rows):
        spike_id = int(state["spike_index"][row])
        t = np.arange(Y.shape[2]) * .32
        A = ax[0, j]
        for c in range(Y.shape[1]):
            A.plot(offs[j, c, 0] + t, offs[j, c, 1] + Y[j, c] * amp,
                   "#e03131", lw=.9)
            A.plot(offs[j, c, 0] + t, offs[j, c, 1] + total[j, c] * amp,
                   "#2f9e44", lw=1.05, ls="--")
            A.plot(offs[j, c, 0], offs[j, c, 1], "s", ms=2.6, c="#999")
        err = ((Y[j] - total[j]) ** 2).mean() / max(1e-12, (Y[j] ** 2).mean())
        A.set_title(f"spike {spike_id}  rel.err {err:.3f}", fontsize=8)
        A.tick_params(labelsize=6)

        B = ax[1, j]
        width = .38
        xc = np.arange(Y.shape[1])
        measured, fitted = np.ptp(Y[j], axis=1), np.ptp(total[j], axis=1)
        B.bar(xc - width / 2, measured, width, color="#e03131",
              label="measured" if j == 0 else None)
        B.bar(xc + width / 2, fitted, width, color="#2f9e44",
              label="model" if j == 0 else None)
        ptp_err = np.abs(fitted - measured).sum() / max(1e-9, measured.sum())
        B.set_title(f"ptp · rel.err {ptp_err:.3f}", fontsize=7.5)
        B.tick_params(labelsize=6); B.set_xlabel("channel", fontsize=7)
        if j == 0:
            B.legend(fontsize=6)

        C = ax[2, j]
        active = np.flatnonzero(state["source_index"][row] >= 0)
        spatial = np.zeros_like(XX, dtype=np.float32)
        for slot in active:
            candidate = int(state["source_index"][row, slot])
            pos = candidate_pos[candidate]
            weight = float(state["source_weight"][row, slot])
            spatial += weight * _adapter_spatial_field(
                codebook, state, row, slot, XX, YY, "xy")
        if spatial.max() > 0:
            spatial /= spatial.max()
        C.imshow(spatial.T, origin="lower", extent=[-lim, lim, -lim, lim],
                 cmap="magma", aspect="equal", vmin=0, vmax=1)
        C.scatter(offs[j, :, 0], offs[j, :, 1], s=12, marker="s", c="none",
                  edgecolors="w", linewidths=.6)
        C.plot(0, 0, "c+", ms=11, mew=1.7)
        labels = []
        for k, slot in enumerate(active):
            candidate = int(state["source_index"][row, slot])
            pos = candidate_pos[candidate]
            q = float(state["source_weight"][row, slot])
            color = colours[k % len(colours)]
            C.plot(pos[0], pos[1], "o", mfc="none", mec=color,
                   ms=10 + 5 * q, mew=2.0)
            labels.append(f"s{k + 1}:q={q:.2f},z={pos[2]:.0f}")
        C.set_title("  ".join(labels), fontsize=7.2)
        C.tick_params(labelsize=6)

        E = ax[3, j]
        feature_key = "shape_feature" if "shape_feature" in state else "source_coeff"
        coeff = state[feature_key][row, active]
        m = coeff.shape[1]
        if shared_waveform:
            # M1 serializes each source contribution as q_r * v. Because the
            # active q values sum to one, summing the rows recovers the one
            # shared waveform vector v exactly.
            E.bar(np.arange(m), coeff.sum(axis=0), .8, color="#4c8dff",
                  label="shared v")
        else:
            bar_width = .8 / max(1, len(active))
            for k, values in enumerate(coeff):
                offset = (k - (len(active) - 1) / 2) * bar_width
                E.bar(np.arange(m) + offset, values, bar_width,
                      color=colours[k % len(colours)], label=f"source {k + 1}")
        E.axhline(0, color="0.5", lw=.5)
        label = "atom-shape feature" if feature_key == "shape_feature" else "shape q"
        E.set_xlabel(label, fontsize=7); E.tick_params(labelsize=6)
        if j == 0:
            E.legend(fontsize=6)

    ax[0, 0].set_ylabel("measured (red) vs model (green)", fontsize=8)
    ax[3, 0].set_ylabel("shared v" if shared_waveform else "v by source", fontsize=8)
    fig.suptitle(
        f"example spikes — {_title(summary)}   ·   active sources shown as coloured ○; "
        f"cyan + anchor, white ▫ contacts\n"
        f"canonical MULTI-SOURCE spike ids "
        f"{[int(state['spike_index'][r]) for r in rows]} — identical for every model "
        f"in the browser, so these panels are directly comparable", fontsize=11)
    fig.savefig(out / "spikes.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    vmax = float(np.quantile(np.abs(np.concatenate(
        [Y.ravel(), total.ravel(), source.ravel()])), .995))
    rmax = source.shape[1]
    fig, ax = plt.subplots(rmax + 3, n, figsize=(2.65 * n, 2.0 * (rmax + 3)),
                          constrained_layout=True, squeeze=False)
    for j in range(n):
        panels = [(Y[j], "observed"), (total[j], "total")]
        panels += [(source[j, r], f"active term {r + 1} (amplitude-ranked)")
                   for r in range(rmax)]
        panels += [(Y[j] - total[j], "residual")]
        for rr, (arr, label) in enumerate(panels):
            ax[rr, j].imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax[rr, j].set_title(label, fontsize=7.5)
            ax[rr, j].set_ylabel("slot", fontsize=7)
    fig.suptitle("observed waveform = sum of permutation-invariant source terms + residual")
    fig.savefig(out / "spike_decomposition.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def fig_spatial_temporal_shapes(state: dict, summary: dict, codebook: dict,
                                rec, out: Path) -> None:
    """Matched examples emphasizing each spike's learned source constellation."""
    rows, Y, total, source_terms, offs = reconstruction_examples(
        state, summary, codebook, rec)
    n = len(rows)
    omega = np.asarray(state["omega"], np.float32)
    candidate_pos = np.asarray(codebook["candidate_pos"], np.float32)
    colours = ("#4c8dff", "#ff922b", "#12b886", "#ae3ec9")
    lim = 160.0
    grid = np.linspace(-lim, lim, 141, dtype=np.float32)
    X, Ygrid = np.meshgrid(grid, grid, indexing="xy")
    Z, Yz = np.meshgrid(grid, grid, indexing="xy")
    fig, ax = plt.subplots(4, n, figsize=(3.15 * n, 11.5), squeeze=False,
                          constrained_layout=True,
                          gridspec_kw={"height_ratios": [1.2, .8, 1, 1]})
    trace_scale = 16.0 / max(1e-6, float(np.abs(Y).max()))
    for j, row in enumerate(rows):
        spike_id = int(state["spike_index"][row])
        active = np.flatnonzero(state["source_index"][row] >= 0)
        tt = np.arange(Y.shape[2]) * .32
        for c in range(Y.shape[1]):
            ax[0, j].plot(offs[j, c, 0] + tt,
                          offs[j, c, 1] + Y[j, c] * trace_scale,
                          color="#e03131", lw=.8)
            ax[0, j].plot(offs[j, c, 0] + tt,
                          offs[j, c, 1] + total[j, c] * trace_scale,
                          color="#2f9e44", lw=1, ls="--")
        rel = np.mean((Y[j] - total[j]) ** 2) / max(np.mean(Y[j] ** 2), 1e-12)
        ax[0, j].set_title(f"spike {spike_id} · rel.err {rel:.3f}", fontsize=8)
        for k, slot in enumerate(active):
            if "atom_coeff" in state:
                wave = source_terms[j, slot].mean(0)
            else:
                wave = state["source_coeff"][row, slot] @ omega
            q = float(state["source_weight"][row, slot])
            ax[1, j].plot(tt, wave, color=colours[k % len(colours)], lw=1.3,
                          label=f"source {k + 1}, q={q:.2f}")
        ax[1, j].axhline(0, color=".6", lw=.5)
        ax[1, j].legend(fontsize=6, loc="best")

        xy = np.zeros_like(X, dtype=np.float32)
        zy = np.zeros_like(Z, dtype=np.float32)
        for k, slot in enumerate(active):
            idx = int(state["source_index"][row, slot])
            mu = candidate_pos[idx]
            q = float(state["source_weight"][row, slot])
            xy += q * _adapter_spatial_field(
                codebook, state, row, slot, X, Ygrid, "xy")
            zy += q * _adapter_spatial_field(
                codebook, state, row, slot, Z, Yz, "zy")
        peak = max(float(xy.max()), float(zy.max()), 1e-12)
        ax[2, j].imshow(xy, origin="lower", extent=(-lim, lim, -lim, lim),
                        cmap="magma", vmin=0, vmax=peak, aspect="equal")
        ax[3, j].imshow(zy, origin="lower", extent=(-lim, lim, -lim, lim),
                        cmap="magma", vmin=0, vmax=peak, aspect="equal")
        ax[2, j].scatter(offs[j, :, 0], offs[j, :, 1], s=9, marker="s",
                         facecolors="none", edgecolors="white", linewidths=.45)
        for k, slot in enumerate(active):
            mu = candidate_pos[int(state["source_index"][row, slot])]
            q = float(state["source_weight"][row, slot])
            kw = dict(s=35 + 75*q, facecolors="none",
                      edgecolors=colours[k % len(colours)], linewidths=1.5)
            ax[2, j].scatter(mu[0], mu[1], **kw)
            ax[3, j].scatter(mu[2], mu[1], **kw)
        ax[2, j].set(xlabel="x (µm)", ylabel="y (µm)", title="spatial footprint: x–y")
        ax[3, j].set(xlabel="z (µm)", ylabel="y (µm)", title="spatial footprint: z–y")
        for a in ax[:, j]:
            a.tick_params(labelsize=6)
    ax[0, 0].set_ylabel("observed red / recon green", fontsize=8)
    ax[1, 0].set_ylabel("sub-source waveform", fontsize=8)
    fig.suptitle(f"learned per-spike spatial and temporal source shapes — {_title(summary)}")
    fig.savefig(out / "spatial_temporal_shapes.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def fig_localize(state: dict, summary: dict, flat: dict, rec, out: Path) -> None:
    take = _sample_rows(len(flat["pos"]), 100000, 29)
    parent_spike = flat["spike"][take]
    mp = rec.mp_xyz[parent_spike]
    pos = flat["pos"][take]
    fig, ax = plt.subplots(1, 4, figsize=(17.0, 4.2), constrained_layout=True)
    q = flat["q"][take]
    ax[0].hexbin(pos[:, 0], pos[:, 1], C=q, reduce_C_function=np.sum,
                 gridsize=65, cmap="magma", bins="log")
    ax[0].set(xlabel="source x (µm)", ylabel="source y (µm)",
              title="all active source locations")
    for d, label in enumerate("xyz"):
        ax[d + 1].hexbin(mp[:, d], pos[:, d], C=q, reduce_C_function=np.sum,
                         gridsize=60, cmap="magma", bins="log")
        lo = min(float(mp[:, d].min()), float(pos[:, d].min()))
        hi = max(float(mp[:, d].max()), float(pos[:, d].max()))
        ax[d + 1].plot([lo, hi], [lo, hi], "--", lw=1)
        corr = np.corrcoef(mp[:, d], pos[:, d])[0, 1]
        ax[d + 1].set(xlabel=f"monopole {label} (µm)", ylabel=f"source {label} (µm)",
                      title=f"{label}: r={corr:+.3f}")
    fig.suptitle(f"localization comparison — {_title(summary)}")
    fig.savefig(out / "localize.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def _densest_second(spike_idx: np.ndarray, rec) -> int:
    sec = np.floor(rec.spike_times[spike_idx] / rec.fs).astype(int)
    return int(np.bincount(sec).argmax())


def fig_aggregate(state: dict, summary: dict, flat: dict, rec, out: Path) -> None:
    t0 = _densest_second(state["spike_index"], rec)
    use = (flat["time"] >= t0) & (flat["time"] < t0 + 1)
    for name, ylim, figsize in (("aggregate_1s", FULL_Y, (7.0, 11.0)),
                               ("aggregate_1s_zoom", ZOOM_Y, (9.0, 6.0))):
        pos, amp = flat["pos"][use], flat["amp"][use]
        H, xe, ye = np.histogram2d(pos[:, 0], pos[:, 1], bins=(100, int((ylim[1]-ylim[0])/2)),
                                   range=[XY_LIM, ylim], weights=amp)
        nz = H[H > 0]
        vmax = float(np.quantile(nz, .997)) if len(nz) else 1.0
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.imshow(H.T, origin="lower", extent=[*XY_LIM, *ylim], aspect="auto",
                       cmap="magma", norm=PowerNorm(.45, vmin=0, vmax=vmax))
        fig.colorbar(im, ax=ax).set_label("summed conserved source amplitude")
        ax.set(xlabel="lateral x (µm)", ylabel="probe depth y (µm)",
               title=f"all-source cloud, t={t0}–{t0+1} s — {_title(summary)}")
        fig.savefig(out / f"{name}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)
        _match_standard_canvas(out / f"{name}.png")


def fig_source_cloud(state: dict, summary: dict, flat: dict, rec, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 7.0), constrained_layout=True)
    for a, ylim, title in ((ax[0], FULL_Y, "full probe"), (ax[1], ZOOM_Y, "400–900 µm")):
        use = (flat["pos"][:, 1] >= ylim[0]) & (flat["pos"][:, 1] <= ylim[1])
        # Every source contributes q, so the spatial histogram retains one unit
        # of count mass per parent spike even when that mass is split over sites.
        sc = a.hexbin(flat["pos"][use, 0], flat["pos"][use, 1], C=flat["q"][use],
                      reduce_C_function=np.sum, gridsize=(90, 150), bins="log",
                      extent=(*XY_LIM, *ylim), mincnt=1, cmap="magma",
                      rasterized=True)
        a.set_xlim(*XY_LIM); a.set_ylim(*ylim)
        a.set(xlabel="lateral x (µm)", ylabel="probe depth y (µm)", title=title)
    fig.colorbar(sc, ax=ax, label="summed fractional source count mass q")
    fig.suptitle(f"all {len(flat['pos']):,} active source events (hex-aggregated) — "
                 f"{_title(summary)}")
    fig.savefig(out / "source_cloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_diagnostics(state: dict, summary: dict, out: Path) -> None:
    support = state["support_size"]
    multi = support > 1
    qsort = np.sort(state["source_weight"], axis=1)
    secondary = qsort[:, -2] if qsort.shape[1] > 1 else np.zeros(len(qsort))
    separation = state.get("pair_separation", np.full(len(support), np.nan))
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    ax = ax.ravel()
    ax[0].hist(support, bins=np.arange(support.max() + 2) - .5, rwidth=.85)
    ax[0].set(xlabel="effective support", ylabel="spikes", title="support size")
    ax[1].hist(secondary[multi], bins=np.linspace(0, .5, 51))
    ax[1].set(xlabel="second-largest q", ylabel="multi-source spikes",
              title="secondary contribution")
    ax[2].hist(separation[np.isfinite(separation)], bins=70)
    ax[2].set(xlabel="pair separation (µm)", ylabel="spikes", title="separation")
    ax[3].hist(np.log10(np.maximum(state["condition"], 1)), bins=70)
    ax[3].set(xlabel="log10 support Gram condition", ylabel="spikes", title="conditioning")
    leave = state["leaveout_delta"][state["source_index"] >= 0]
    finite = leave[np.isfinite(leave)]
    cap = float(np.quantile(np.abs(finite), .999)) if len(finite) else 1.0
    cap = max(cap, 1e-8)
    ax[4].hist(np.clip(finite, -cap, cap), bins=np.linspace(-cap, cap, 71))
    linthresh = max(cap * 1e-3, 1e-10)
    ax[4].set_xscale("symlog", linthresh=linthresh)
    # Matplotlib's default symmetrical-log locator can place a dense run of
    # decade labels on both sides of zero. Five symmetric ticks remain legible
    # in the matched six-panel layout while preserving sign and dynamic range.
    middle = math.sqrt(cap * linthresh)
    sensitivity_ticks = [-cap, -middle, 0.0, middle, cap]
    ax[4].set_xticks(sensitivity_ticks)
    ax[4].set_xticklabels([f"{x:.1e}" if x else "0" for x in sensitivity_ticks])
    ax[4].axvline(0, color="k", lw=.7)
    ax[4].set(xlabel="signed leave-one-source-out ΔSSE", ylabel="sources",
              title="source sensitivity (negative exposes cross-terms)")
    if "pair_gain" in state:
        ax[5].scatter(secondary[multi], state["pair_gain"][multi], s=2, alpha=.2,
                      rasterized=True)
        ax[5].set(xlabel="secondary q", ylabel="captured-energy gain",
                  title="weight versus incremental fit")
    else:
        # pair_gain is a diagnostic of the beam pair search. Models that select
        # support another way (e.g. matching pursuit over a product dictionary)
        # have no such quantity; the panel says so rather than being dropped, so
        # the row keeps the same panel inventory as every other model.
        ax[5].text(.5, .5, "no pair-search diagnostic\nfor this solver",
                   ha="center", va="center", transform=ax[5].transAxes, fontsize=9)
        ax[5].set(xticks=[], yticks=[], title="weight versus incremental fit")
    fig.suptitle(f"multipole diagnostics — {_title(summary)}")
    fig.savefig(out / "multipole_diagnostics.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def fig_readout_sensitivity(state: dict, summary: dict, flat: dict, rec, out: Path) -> None:
    parent_spike = state["spike_index"]
    t = rec.spike_times[parent_spike] / rec.fs
    all_take = _sample_rows(len(flat["pos"]), 150000, 37)
    one_take = _sample_rows(len(t), 150000, 37)
    panels = ((flat["time"][all_take], flat["pos"][all_take, 1], flat["q"][all_take],
               "all-source cloud (fractional q counts)"),
              (t[one_take], state["pos_dominant"][one_take, 1],
               np.ones(len(one_take)), "dominant source"),
              (t[one_take], state["pos_barycenter"][one_take, 1],
               np.ones(len(one_take)), "q-weighted barycenter"))
    fig, ax = plt.subplots(3, 1, figsize=(13.0, 10.0), constrained_layout=True)
    for a, (tx, yy, weight, title) in zip(ax, panels):
        a.hexbin(tx, yy, C=weight, reduce_C_function=np.sum,
                 gridsize=(300, 160), extent=(0, rec.duration_s, *ZOOM_Y),
                 cmap="magma", bins="log", mincnt=1)
        a.set_ylim(*ZOOM_Y); a.set_ylabel("depth y (µm)"); a.set_title(title)
    ax[-1].set_xlabel("recording time (s)")
    fig.suptitle(f"localization-readout sensitivity — {_title(summary)}")
    fig.savefig(out / "readout_sensitivity.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def fig_real_nonrigid_rasters(state: dict, summary: dict, flat: dict, rec,
                              figures_root: Path, out: Path) -> list[Path]:
    """Dedicated real-DREDge nonrigid rasters matching the uncorrected views."""
    tag = summary["tag"]
    dy = correction(figures_root, tag, "real-nonrigid",
                    rec.spike_times[flat["spike"]], rec.fs,
                    y=flat["pos"][:, 1])
    _, jitter_y, jitter_t = jittered(flat["pos"][:, :2], tag, 17, 1.5)
    corrected_y = flat["pos"][:, 1] - dy + (jitter_y - flat["pos"][:, 1])
    corrected_t = flat["time"] + jitter_t
    palette = shuffled_palette(state["omega"].shape[0])
    made = []
    for suffix, ylim in (("full", FULL_Y), ("zoom", ZOOM_Y)):
        density = out / f"depth_time_density_{suffix}_drn.png"
        basis = out / f"depth_time_basis_{suffix}_drn.png"
        save_density_raster(density, corrected_t, corrected_y, flat["amp"],
                            ylim, f"{tag} · real DREDge nonrigid corrected")
        save_basis_raster(basis, corrected_t, corrected_y, flat["dom"], palette,
                          ylim, f"{tag} · real DREDge nonrigid corrected", False,
                          point_weight=flat["q"])
        made.extend((density, basis))
    return made


def fig_embeddings(state: dict, summary: dict, out: Path,
                   max_points: int = 20000) -> None:
    active = state["source_index"] >= 0
    coef = state["shape_feature"][active] if "shape_feature" in state \
        else state["source_coeff"][active]
    q = state["source_weight"][active]
    take = _sample_rows(len(coef), max_points, 41)
    from spiketensor.viz_centroid_basis import pca_projection
    projected, _ = pca_projection(coef, seed=0)
    pca = projected[take, :2]
    x = coef[take]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    for name, emb, xlabel, ylabel in (("embed_pca", pca, "PC1", "PC2"),):
        fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=q[take], cmap=Q_CMAP,
                        vmin=0, vmax=1, s=3, alpha=.4, rasterized=True)
        fig.colorbar(sc, ax=ax).set_label("source contribution q")
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=f"source-shape {name[6:].upper()} — {_title(summary)}")
        fig.savefig(out / f"{name}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)
        _match_standard_canvas(out / f"{name}.png")
    try:
        import umap
        emb = umap.UMAP(n_neighbors=20, min_dist=.15, random_state=0,
                        n_jobs=1).fit_transform(x)
        method = "UMAP"
    except Exception:
        # Deterministic nonlinear fallback keeps the panel inventory complete in
        # minimal environments while recording the actual method in the title.
        from sklearn.manifold import Isomap
        n_neighbors = min(10, max(2, len(x) - 1))
        emb = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(x)
        method = "Isomap fallback"
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=q[take], cmap=Q_CMAP,
                    vmin=0, vmax=1, s=3, alpha=.4, rasterized=True)
    fig.colorbar(sc, ax=ax).set_label("source contribution q")
    ax.set(xlabel="dimension 1", ylabel="dimension 2",
           title=f"source-shape {method} — {_title(summary)}")
    fig.savefig(out / "embed_umap.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    _match_standard_canvas(out / "embed_umap.png")


def fig_convergence(summary: dict, out: Path) -> None:
    history = summary.get("fit_history", [])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    if history:
        # schema-tolerant: the beam solver logs outer/inference_nmse, the one-hot
        # matching-pursuit solver logs step/nmse; both are outer-iteration histories
        it = [x.get("outer", x.get("step", i + 1)) for i, x in enumerate(history)]
        inference = [x.get("inference_nmse",
                           x.get("nmse", x.get("metrics", {}).get("nmse")))
                     for x in history]
        ax[0].plot(it, inference, "-o", label="post-selection refit")
        fixed = [x.get("fixed_support_basis_nmse", x.get("nmse_after_basis", np.nan))
                 for x in history]
        if np.isfinite(np.asarray(fixed, dtype=float)).any():
            ax[0].plot(it, fixed, "-s", label="exact omega update")
        spatial = [x.get("spatial", {}) for x in history]
        ax[1].plot(it, [x.get("check_sse_before", np.nan) for x in spatial], "-o",
                   label="before spatial block")
        ax[1].plot(it, [x.get("check_sse_after", np.nan) for x in spatial], "-s",
                   label="after spatial block")
    else:
        ax[0].text(.5, .5, "frozen production codebook\n(no alternating updates)",
                   ha="center", va="center", transform=ax[0].transAxes)
    ax[0].set(xlabel="outer iteration", ylabel="fit-pool nMSE", title="exact blocks")
    ax[1].set(xlabel="outer iteration", ylabel="fixed-support SSE", title="spatial block")
    for a in ax:
        a.grid(alpha=.3)
        handles, labels = a.get_legend_handles_labels()
        if handles:
            a.legend(fontsize=7)
    fig.suptitle(f"optimization convergence — {_title(summary)}")
    fig.savefig(out / "convergence.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def render_run(runs: Path, figs: Path, tag: str, max_flat: int = 0,
               quick: bool = False) -> list[str]:
    state, summary, codebook = load_run(runs, tag)
    rec = D.load("np1")
    out = figs / tag
    out.mkdir(parents=True, exist_ok=True)
    flat = flattened_sources(state, rec, max_points=max_flat)
    fig_components(state, summary, out)
    fig_usage(state, summary, out)
    fig_spikes(state, summary, codebook, rec, out)
    fig_spatial_temporal_shapes(state, summary, codebook, rec, out)
    fig_diagnostics(state, summary, out)
    if not quick:
        fig_basis(state, summary, flat, out)
        fig_localize(state, summary, flat, rec, out)
        fig_aggregate(state, summary, flat, rec, out)
        fig_source_cloud(state, summary, flat, rec, out)
        fig_readout_sensitivity(state, summary, flat, rec, out)
        fig_embeddings(state, summary, out)
        fig_convergence(summary, out)
        palette = shuffled_palette(state["omega"].shape[0])
        x, y, jt = jittered(flat["pos"][:, :2], tag, 17, 1.5)
        save_centroid(out / "centroid_basis_full.png", x, y, flat["dom"], palette,
                      rec, FULL_Y, tag, False, point_weight=flat["q"])
        save_centroid(out / "centroid_basis_zoom.png", x, y, flat["dom"], palette,
                      rec, ZOOM_Y, tag, False, point_weight=flat["q"])
        save_density_raster(out / "depth_time_density_full.png",
                            flat["time"] + jt, y, flat["amp"], FULL_Y, tag)
        save_density_raster(out / "depth_time_density_zoom.png",
                            flat["time"] + jt, y, flat["amp"], ZOOM_Y, tag)
        save_basis_raster(out / "depth_time_basis_full.png",
                          flat["time"] + jt, y, flat["dom"], palette, FULL_Y, tag, False,
                          point_weight=flat["q"])
        save_basis_raster(out / "depth_time_basis_zoom.png",
                          flat["time"] + jt, y, flat["dom"], palette, ZOOM_Y, tag, False,
                          point_weight=flat["q"])
    # Tight-layout text (notably C5's explicit rank-inflated labels) can widen
    # the outer canvas even when the axes and data layout are unchanged.  Keep
    # every generic-adapter panel on the authoritative standard canvas.
    for path in out.glob("*.png"):
        _match_standard_canvas(path)
    files = sorted(p.name for p in out.glob("*.png"))
    (out / "panel_manifest.json").write_text(json.dumps({
        "tag": tag, "primary_readout": "flattened all-source cloud",
        "amplitude_conservation": "source amplitude = parent total coefficient-norm sum * q",
        "count_conservation": "source count mass = q; active q values sum to one per parent spike",
        "contribution_color_scale": [0.0, 1.0], "files": files,
    }, indent=2))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/multipole")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/multipole")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--max-flat", type=int, default=0,
                    help="optional display-only source-cloud cap; 0 uses every source")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tags = a.tags or sorted(p.stem[len("summary_"):] for p in a.runs.glob("summary_*.json"))
    for i, tag in enumerate(tags, 1):
        print(f"[{i}/{len(tags)}] render {tag}", flush=True)
        render_run(a.runs, a.figs, tag, max_flat=a.max_flat, quick=a.quick)


if __name__ == "__main__":
    main()
