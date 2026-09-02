#!/bin/bash
#SBATCH --job-name=plot0019-step20
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
RUN_PATH=$PROJECT_ROOT/residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction10_step20_fitted8
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/0019_allchannel_pass3_round1_fraction10_step20_fitted8

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
        python residuals/src/plots/plot_0019_recording_replay.py \
            --run "$1" --out "$2/recording_replay_chunk001629.png" --chunk-index 1629 --overwrite
        python residuals/src/plots/build_plot_gallery.py \
            --run "$1" --plots "$2" --out "$2/index.html" \
            --title "0019 all-channel steep-escalation 10% reconstruction pursuit"
    ' bash "$RUN_PATH" "$PLOT_OUTPUT"

echo "0019 all-channel steep-escalation 10% plot suite complete: $PLOT_OUTPUT"

if [[ -z "${SKIP_FULLREC:-}" ]]; then
    sbatch --job-name=plot0019-step20-fullrec \
        --output=$PROJECT_ROOT/residuals/slurm_logs/%x_%j.out \
        --error=$PROJECT_ROOT/residuals/slurm_logs/%x_%j.err \
        "$PROJECT_ROOT/residuals/src/plots/0019_full_recording_replay.sbatch" \
        "$RUN_PATH"
fi
