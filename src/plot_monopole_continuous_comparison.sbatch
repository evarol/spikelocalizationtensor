#!/bin/bash
#SBATCH --job-name=plot_monopole_continuous
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/runs/dataset1_p1
GRID=$SESSION/gpu_fit_voxel_1um_masked_monopole.npz
CONTINUOUS=$SESSION/gpu_fit_voxel_1um_masked_monopole_continuous.npz
OUT=$ROOT/out/plots/gpu_fit_voxel_1um_masked_monopole_continuous/projections

if [[ ! -e "$GRID" || ! -e "$CONTINUOUS" ]]; then
    echo "monopole fit is missing" >&2
    exit 2
fi
if [[ -e "$OUT" ]]; then
    echo "plot directory already exists: $OUT" >&2
    exit 2
fi

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_projection_histograms.py --session $SESSION \
        --fit $GRID --continuous $CONTINUOUS --out $OUT"
