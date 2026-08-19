#!/bin/bash
#SBATCH --job-name=fit_masked_kernels
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

KERNELS=(monopole exponential gauss lorentz power student yukawa dog)
KERNEL=${KERNELS[$SLURM_ARRAY_TASK_ID]}
ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/runs/dataset1_p1
RESULT=$SESSION/gpu_fit_voxel_1um_masked_${KERNEL}.npz

if [[ -e "$RESULT" ]]; then
    echo "result already exists: $RESULT" >&2
    exit 2
fi

echo "=== masked-channel closed-form-gain 1 um voxel $KERNEL fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python src/fit_dataset.py \
        $SESSION $RESULT --device cuda --kernel $KERNEL --refine-levels 6"

echo "Completed: $(date)"
