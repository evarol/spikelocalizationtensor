#!/bin/bash
#SBATCH --job-name=fit_dataset1_p1
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/fit_dataset1_p1_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/fit_dataset1_p1_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/residuals/runs/dataset1_p1
RESULT=$SESSION/gpu_fit.npz

echo "=== GPU spike model fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Session: $SESSION"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python src/fit_dataset.py \
        $SESSION $RESULT --device cuda"

echo "Completed: $(date)"
