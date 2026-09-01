#!/bin/bash
#SBATCH --job-name=plot0019-fullrec
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
RUN_PATH=${1:-$PROJECT_ROOT/residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8}
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/$(basename "$RUN_PATH")

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
        python -m py_compile residuals/src/plots/plot_0019_full_recording_replay.py
        python residuals/src/plots/plot_0019_full_recording_replay.py \
            --run "$1" --out "$2/recording_replay_full_recording.png" \
            --samples-per-bin 5000 --overwrite
    ' bash "$RUN_PATH" "$PLOT_OUTPUT"

echo "0019 full-recording replay complete: $PLOT_OUTPUT/recording_replay_full_recording.png"
