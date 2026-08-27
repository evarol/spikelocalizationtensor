#!/bin/bash
#SBATCH --job-name=plot_global_q_existing
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
OUTPUT=$ROOT/out/global_codebook_per_q_16344946

if [[ -e "$OUTPUT" ]]; then
    echo "output already exists: $OUTPUT" >&2
    exit 2
fi

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -s -- "$ROOT" "$RECORDING" "$OUTPUT" <<'INNER'
set -euo pipefail

ROOT=$1
RECORDING=$2
OUTPUT=$3
source /ext3/env.sh
cd "$ROOT"

Q_VALUES=(8 16 24 32)
RUN_NAMES=(
    global_codebook_pursuit_q8_matched_16344946_0
    global_codebook_pursuit_q16_matched_16344946_1
    global_codebook_pursuit_q24_matched_16344946_2
    global_codebook_pursuit_q32_matched_16344946_3
)

for index in "${!Q_VALUES[@]}"; do
    Q=${Q_VALUES[$index]}
    RUN=$ROOT/runs/dataset1_p1/${RUN_NAMES[$index]}
    Q_OUTPUT=$OUTPUT/q${Q}
    ADAPTER=$Q_OUTPUT/.adapter
    mkdir -p "$ADAPTER/reconstruction_session"

    python - "$RUN" "$ADAPTER" <<'PY'
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
adapter = Path(sys.argv[2])
omega = np.load(run / "omega.npy")
np.savez(
    adapter / "global_fit.npz",
    sources=np.load(run / "sources.npy", mmap_mode="r"),
    alpha=np.load(run / "alpha.npy", mmap_mode="r"),
    omega=omega,
    temporal_idx=np.load(run / "temporal_idx.npy", mmap_mode="r"),
)
chunk_path = run / "chunks" / "chunk_000000.npz"
with np.load(chunk_path, allow_pickle=False) as chunk:
    session = adapter / "reconstruction_session"
    for source_name, output_name in (
        ("residual_waveforms", "neighborhood_waveforms.npy"),
        ("local_coords", "local_coords.npy"),
        ("neighbor_ids", "neighbor_ids.npy"),
        ("neighbor_counts", "neighbor_counts.npy"),
    ):
        np.save(session / output_name, chunk[source_name])
    input_energy = np.asarray(chunk["input_energy"], dtype=np.float64)
    captured_energy = np.asarray(chunk["captured_energy"], dtype=np.float64)
    nmse = 1 - captured_energy.sum() / max(
        input_energy.sum(), np.finfo(np.float64).tiny
    )
    np.savez(
        adapter / "reconstruction_fit.npz",
        sources=chunk["sources"],
        profile_idx=chunk["profile_idx"],
        omega=omega,
        temporal_idx=chunk["temporal_idx"],
        alpha=chunk["alpha"],
        nmse=np.asarray(nmse),
    )
PY

    python src/plots/plot_amplitude_scatter.py \
        --session "$RUN" \
        --fit "$ADAPTER/global_fit.npz" \
        --recording "$RECORDING" \
        --method-label "global Q=$Q pursuit" \
        --out "$Q_OUTPUT/localizations_xy_yz_xz.png"

    python src/plots/plot_reconstructions.py \
        --session "$ADAPTER/reconstruction_session" \
        --fit "$ADAPTER/reconstruction_fit.npz" \
        --kernel monopole \
        --sample-size 30000 \
        --n-examples 6 \
        --out "$Q_OUTPUT/reconstructions"

    for chunk_index in 0 1 2 3; do
        python src/plots/plot_residual_pass_diagnostics.py \
            --run "$RUN" \
            --chunk-index "$chunk_index" \
            --out "$Q_OUTPUT/recording_energy_loss_after_each_pass_chunk${chunk_index}.png"
    done

    python src/plots/plot_temporal_codebook.py \
        --fit "$ADAPTER/global_fit.npz" \
        --out "$Q_OUTPUT/codebook_values.png"

    python src/plots/plot_temporal_codebook_usage.py \
        --fit "$ADAPTER/global_fit.npz" \
        --out "$Q_OUTPUT/codebook_usage_percentages.png"

    rm -r "$ADAPTER"
    echo "Q=$Q existing plot suite complete: $Q_OUTPUT"
done
INNER

echo "Completed: $(date)"
