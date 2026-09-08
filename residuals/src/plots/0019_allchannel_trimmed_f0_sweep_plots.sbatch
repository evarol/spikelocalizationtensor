#!/bin/bash
#SBATCH --job-name=plot0019-trimsweep
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
: "${SLURM_ARRAY_TASK_ID:?must run as a SLURM array job}"

RULES=(min min min min min mean mean mean mean mean kofn kofn kofn kofn kofn flat flat flat flat flat)
F0S=(0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30)
TAGS=(05 10 15 20 30 05 10 15 20 30 05 10 15 20 30 05 10 15 20 30)

i=$SLURM_ARRAY_TASK_ID
RULE=${RULES[$i]}
F0=${F0S[$i]}
TAG=${TAGS[$i]}

RUN_NAME=0019_allchannel_trimmed_${RULE}_f0${TAG}_q32
RUN_PATH=$PROJECT_ROOT/residuals/runs/dataset1_p1/$RUN_NAME
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/$RUN_NAME

if [[ ! -f "$RUN_PATH/summary.json" ]]; then
    echo "missing completed run: $RUN_PATH" >&2
    exit 2
fi

mkdir -p "$PLOT_OUTPUT"
cd "$PROJECT_ROOT"

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        set -euo pipefail
        source /ext3/env.sh
        export PYTHONPATH=$PWD/residuals/src:$PWD/residuals/src/plots
        python -m py_compile \
            residuals/src/plots/0016_onehot_lattice_plots.py \
            residuals/src/plots/build_plot_gallery.py \
            residuals/src/plots/plot_0014_codebook_usage.py \
            residuals/src/plots/plot_0019_allchannel_cones.py \
            residuals/src/plots/plot_0019_recording_replay.py \
            residuals/src/plots/plot_0019_full_recording_replay.py \
            residuals/src/plots/plot_raw_residual_reconstructions.py \
            residuals/src/plots/plot_spiketensor_residual_pursuit.py \
            residuals/src/plots/plot_temporal_codebook_depth_time_raster.py
        python residuals/src/plots/0016_onehot_lattice_plots.py \
            --run "$1" --out "$2"
        python residuals/src/plots/plot_0014_codebook_usage.py \
            --run "$1" --out "$2/temporal_codebook_usage.png"
        python residuals/src/plots/plot_0019_allchannel_cones.py \
            --run "$1" --out "$2/temporal_prototype_cones.png"
        python residuals/src/plots/plot_raw_residual_reconstructions.py \
            --run "$1" --out "$2/reconstructions" --chunk-index 0 \
            --max-examples 6
        python residuals/src/plots/plot_spiketensor_residual_pursuit.py \
            --run "$1" --out "$2/spiketensor"
        python residuals/src/plots/plot_temporal_codebook_depth_time_raster.py \
            --run "$1" --out "$2/depth_time_omega_raster.png" \
            --max-points 1000000 --overwrite
        python residuals/src/plots/plot_0019_recording_replay.py \
            --run "$1" --out "$2/recording_replay_chunk0.png" --chunk-index 0 --overwrite
        MOST=$(python - "$1" <<PYEOF
import sys
import numpy as np
from pathlib import Path
best, best_frac = None, -1.0
for c in sorted(Path(sys.argv[1]).glob("pass_00/chunk_*.npz")):
    with np.load(c, allow_pickle=False) as a:
        if "captured_energy" in a.files and "input_energy" in a.files:
            ce = a["captured_energy"]
            ie = a["input_energy"]
            if len(ie) and ie.sum() > 0:
                frac = ce.sum() / ie.sum()
                if frac > best_frac:
                    best_frac, best = frac, int(c.stem.split("_")[1])
print(best if best is not None else 0)
PYEOF
)
        python residuals/src/plots/plot_0019_recording_replay.py \
            --run "$1" \
            --out "$2/recording_replay_chunk$(printf "%06d" "$MOST").png" \
            --chunk-index "$MOST" --overwrite
        python residuals/src/plots/plot_0019_full_recording_replay.py \
            --run "$1" --out "$2/recording_replay_full_recording.png" \
            --samples-per-bin 5000 --overwrite
        python residuals/src/plots/build_plot_gallery.py \
            --run "$1" --plots "$2" --out "$2/index.html" \
            --title "trimmed-gate 0019 all-channel $3 bar $4 (q32)"
    ' bash "$RUN_PATH" "$PLOT_OUTPUT" "$RULE" "$F0"

echo "trimmed 0019 $RULE bar $F0 q32 plot suite complete: $PLOT_OUTPUT"