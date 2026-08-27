#!/bin/bash
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN=${RUN:?RUN must name the completed recording run directory}
OUT=${OUT:?OUT must name the plot output directory}
cd "$ROOT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && export PYTHONPATH=\$PWD/residuals/src:\$PWD/residuals/src/plots && python residuals/src/plots/plot_raw_residual_reconstructions.py --run \"$RUN\" --out \"$OUT/reconstructions\" --chunk-index 0 --max-examples 4 && python residuals/src/plots/plot_raw_residual_localizations.py --run \"$RUN\" --out \"$OUT/localizations_by_pass.png\" && python residuals/src/plots/plot_residual_pass_diagnostics.py --run \"$RUN\" --out \"$OUT/pass_diagnostics_chunk0.png\" --chunk-index 0 && python residuals/src/plots/plot_temporal_codebook_depth_time_raster.py --run \"$RUN\" --out \"$OUT/depth_time_omega_raster.png\" && python residuals/src/plots/plot_spiketensor_residual_pursuit.py --run \"$RUN\" --out \"$OUT/spiketensor\""
