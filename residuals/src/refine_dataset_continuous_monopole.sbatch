#!/bin/bash
#SBATCH --job-name=refine_cont_monopole
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/residuals/runs/dataset1_p1
FIT=$SESSION/gpu_fit_voxel_1um_masked_monopole.npz
RESULT=$SESSION/gpu_fit_voxel_1um_masked_monopole_continuous.npz

if [[ ! -e "$FIT" ]]; then
    echo "fit does not exist: $FIT" >&2
    exit 2
fi
if [[ -e "$RESULT" || -e "${RESULT%.npz}.json" ]]; then
    echo "continuous result already exists: $RESULT" >&2
    exit 2
fi

echo "=== continuous xyz refinement of frozen masked monopole fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/refine_dataset_continuous.py $SESSION $FIT $RESULT \
        --kernel monopole --device cuda --spike-chunk 131072"

echo "Completed: $(date)"
