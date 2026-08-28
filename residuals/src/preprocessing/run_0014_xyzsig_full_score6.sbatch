#!/bin/bash
#SBATCH --job-name=xyzsig0014-full-s6
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_62_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/varol-version
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
OUTPUT=$ROOT/runs/dataset1_p1/0014_xyzsig_full_score6
CHILD_PID=

handle_usr1() {
    echo "USR1 received; stopping the active process and requeueing $SLURM_JOB_ID"
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

trap handle_usr1 USR1

if [[ ! -e "$RECORDING" ]]; then
    echo "missing recording: $RECORDING" >&2
    exit 2
fi

cd "$ROOT"
echo "full corrected score-6 XYZ-sigma pursuit: $OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python -m py_compile residuals/src/preprocessing/residuals_0012.py residuals/src/preprocessing/0014_xyzsig_residual.py && python residuals/src/preprocessing/0014_xyzsig_residual.py all $RECORDING $OUTPUT --threshold 6 --min-fitted-projection 6 --outer-passes 4 --fit-batch-size 2048 --chunk-seconds 1 --resume" &
CHILD_PID=$!
wait "$CHILD_PID"
