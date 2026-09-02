#!/bin/bash
#SBATCH --job-name=plot0019-qsweep
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
: "${VARIANT:?set VARIANT=base|flat10|step20|mean20|kofn20}"
: "${Q:?set Q=16|32|64}"

case "$VARIANT" in
    base)   DIRTAG=fraction20_step10; LABEL=base;;
    flat10) DIRTAG=fraction10_step0;  LABEL="flat 10%";;
    step20) DIRTAG=fraction10_step20; LABEL="steep-escalation 10%";;
    mean20) DIRTAG=fraction20_step10_meanchannel; LABEL="mean-rule 20%";;
    kofn20) DIRTAG=fraction20_step10_kofn875;     LABEL="k-of-n 20%";;
    *) echo "unknown VARIANT: $VARIANT" >&2; exit 2;;
esac

RUN_NAME=0019_allchannel_pass3_round1_${DIRTAG}_fitted8_q${Q}
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
            --title "0019 all-channel $3 reconstruction pursuit (q$4)"
    ' bash "$RUN_PATH" "$PLOT_OUTPUT" "$LABEL" "$Q"

echo "0019 all-channel $LABEL q$Q plot suite complete: $PLOT_OUTPUT"
