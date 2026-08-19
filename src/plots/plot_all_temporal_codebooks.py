"""Plot the temporal codebooks from all ten spatial kernels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KERNELS = (
    "monopole", "exponential", "gauss", "lorentz", "power",
    "student", "yukawa", "dog", "gauss_aniso", "mono_aniso",
)


def fit_path(session, kernel):
    suffix = "_optimized" if kernel.endswith("_aniso") else ""
    return session / f"gpu_fit_voxel_1um_{kernel}{suffix}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    fits = []
    limit = np.finfo(np.float32).eps
    for kernel in KERNELS:
        path = fit_path(args.session, kernel)
        if not path.exists():
            raise FileNotFoundError(path)
        fit = np.load(path, allow_pickle=True)
        omega = np.asarray(fit["omega"])
        limit = max(limit, float(np.abs(omega).max()))
        fits.append((kernel, omega, float(fit["nmse"])))

    colors = plt.colormaps["tab10"](np.arange(fits[0][1].shape[0]) % 10)
    fig, axes = plt.subplots(
        5, 2, figsize=(14, 15), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, (kernel, omega, nmse) in zip(axes.flat, fits):
        samples = np.arange(omega.shape[1])
        for q, row in enumerate(omega):
            ax.plot(samples, row, color=colors[q], linewidth=0.9,
                    label=rf"$\Omega_{{{q}}}$")
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.set_ylim(-1.05 * limit, 1.05 * limit)
        ax.set_title(f"{kernel} · nMSE {nmse:.4f}")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("sample")
    for ax in axes[:, 0]:
        ax.set_ylabel("amplitude")
    axes[0, 0].legend(ncol=4, fontsize=7, loc="upper right")
    fig.suptitle("temporal codebooks across spatial kernels")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
