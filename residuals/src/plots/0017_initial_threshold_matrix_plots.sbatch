#!/bin/bash
#SBATCH --job-name=plot0017-matrix
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general
#SBATCH --array=0-5

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN_NAMES=(
    0017_initial_threshold6_learned_fitted8
    0017_initial_threshold5_learned_fitted8
    0017_initial_threshold7_learned_fitted8
    0017_initial_threshold8_learned_fitted8
    0017_initial_threshold6_isolated30_learned_fitted8
    0017_initial_threshold6_onepass_learned_fitted8
)
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
RUN_PATH=$PROJECT_ROOT/residuals/runs/dataset1_p1/$RUN_NAME
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/$RUN_NAME

if [[ ! -f "$RUN_PATH/summary.json" ]]; then
    echo "missing completed run: $RUN_PATH" >&2
    exit 2
fi

mkdir -p "$PLOT_OUTPUT"
cd "$PROJECT_ROOT"

singularity exec --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        set -euo pipefail
        source /ext3/env.sh
        export PYTHONPATH=$PWD/residuals/src:$PWD/residuals/src/plots
        python -m py_compile \
            residuals/src/plots/0016_onehot_lattice_plots.py \
            residuals/src/plots/plot_0014_codebook_usage.py \
            residuals/src/plots/plot_raw_residual_reconstructions.py \
            residuals/src/plots/plot_spiketensor_residual_pursuit.py \
            residuals/src/plots/plot_temporal_codebook_depth_time_raster.py
        python residuals/src/plots/0016_onehot_lattice_plots.py \
            --run "$1" --out "$2"
        python residuals/src/plots/plot_0014_codebook_usage.py \
            --run "$1" --out "$2/temporal_codebook_usage.png"
        python residuals/src/plots/plot_raw_residual_reconstructions.py \
            --run "$1" --out "$2/reconstructions" --chunk-index 0 \
            --max-examples 6
        python residuals/src/plots/plot_spiketensor_residual_pursuit.py \
            --run "$1" --out "$2/spiketensor"
        python residuals/src/plots/plot_temporal_codebook_depth_time_raster.py \
            --run "$1" --out "$2/depth_time_omega_raster.png" \
            --max-points 1000000 --overwrite
    ' bash "$RUN_PATH" "$PLOT_OUTPUT"

echo "0017 matrix plot suite complete: $PLOT_OUTPUT"
