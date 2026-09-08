#!/bin/bash
#SBATCH --job-name=master-codebook-026
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
CHILD_PID=

handle_usr1() {
    echo "USR1/TERM received; stopping the active process and requeueing $SLURM_JOB_ID"
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

trap handle_usr1 USR1 TERM

cd "$PROJECT_ROOT"
echo "0026 master codebook: harvest 812 IBL clips -> init -> alternating fit (Q32)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
        python -m py_compile residuals/src/preprocessing/026_master_codebook.py
        python residuals/src/preprocessing/026_master_codebook.py \
            all \
            --data-root /scratch/ap7151/_RAW_DATA/ibl-data \
            --output residuals/runs/ibl_bwm/master_codebook_q32 \
            --harvest-workers 8 \
            --resume
    ' &
CHILD_PID=$!
wait "$CHILD_PID"
