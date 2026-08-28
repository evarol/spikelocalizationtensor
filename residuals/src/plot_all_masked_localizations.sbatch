#!/bin/bash
#SBATCH --job-name=plot_all_masked_localizations
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/residuals/runs/dataset1_p1
MONO=/scratch/ap7151/sln-v2/runs/dataset1_p1/monopolar_true/localizations.npy
OUT=$ROOT/residuals/out/plots/gpu_fit_voxel_1um_masked_all_kernels

if [[ -e "$OUT" ]]; then
    echo "plot directory already exists: $OUT" >&2
    exit 2
fi

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_all_localizations.py --session $SESSION \
        --monopole $MONO --out $OUT --masked"
