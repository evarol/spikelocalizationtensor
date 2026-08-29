#!/bin/bash
#SBATCH --job-name=plot-0014-full-passes
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN=$ROOT/residuals/runs/dataset1_p1/0014_xyzsig_full_score8
OUT=$ROOT/residuals/out/0014_xyzsig_full_score8
cd "$ROOT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && export PYTHONPATH=\$PWD/residuals/src:\$PWD/residuals/src/plots && \
python residuals/src/plots/plot_temporal_codebook_depth_time_raster.py --run \"$RUN\" --out \"$OUT/depth_time_omega_raster.png\" --overwrite && \
python residuals/src/plots/plot_0014_full_recording_passes.py --run \"$RUN\" --out \"$OUT/full_recording_residuals_and_rasters.png\""
