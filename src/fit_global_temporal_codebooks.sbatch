#!/bin/bash
#SBATCH --job-name=fit_global_codebook
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --array=0-3

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/runs/dataset1_p1
Q_VALUES=(8 16 24 32)
Q=${Q_OVERRIDE:-${Q_VALUES[$SLURM_ARRAY_TASK_ID]}}
RESULT=$SESSION/global_codebook_q${Q}.npz

if [[ ! "$Q" =~ ^(8|16|24|32)$ ]]; then
    echo "Q must be one of 8, 16, 24, or 32; got $Q" >&2
    exit 2
fi
if [[ ! -e "$SESSION/neighborhood_waveforms.npy" ]]; then
    echo "missing training waveforms: $SESSION/neighborhood_waveforms.npy" >&2
    exit 2
fi
if [[ -e "$RESULT" ]]; then
    echo "result already exists: $RESULT" >&2
    exit 2
fi

echo "=== global Q=$Q temporal-codebook fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Session: $SESSION"
echo "Output: $RESULT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/fit_dataset.py $SESSION $RESULT --device cuda \
        --kernel monopole --q $Q --n-scales 10 --n-sites 16 \
        --n-iters 8 --refine-levels 6"

echo "Completed: $(date)"
