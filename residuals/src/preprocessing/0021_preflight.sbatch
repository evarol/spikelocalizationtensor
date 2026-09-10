#!/bin/bash
#SBATCH --job-name=preflight-0021
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
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
RECORDING=${RECORDING:-/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin}
STEM=$(basename "$RECORDING" | sed 's/\.imec0\.ap\..*$//; s/\.ap\..*$//; s/\.[^.]*$//')

echo "0021 preflight: recording=$RECORDING stem=$STEM"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
        python -m py_compile residuals/src/preprocessing/0021_preflight.py
        python residuals/src/preprocessing/0021_preflight.py \
            '$RECORDING' \
            --output residuals/runs/0021_preflight/$STEM \
            --priors 2,4 \
            --shift off,on \
            --resume
    " &
CHILD_PID=$!
wait "$CHILD_PID"
