#!/bin/bash
#SBATCH --job-name=crossfit0019-his
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN_DIR=$PROJECT_ROOT/residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8
HIS_RESULTS=/scratch/ap7151/_SYMLINKS/am15577-paths/UnitMatch/SLT_ICLR/extraction/results/dataset1_p1
OUTPUT=$PROJECT_ROOT/residuals/runs/dataset1_p1/0019_crossfit_hisspikes_fraction20
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

if [[ ! -e "$HIS_RESULTS/spike_times.npy" ]]; then
    echo "missing his spike list: $HIS_RESULTS" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
echo "cross-fit: 0019 fitter on his $(wc -c < "$HIS_RESULTS/spike_times.npy") bytes of spike times"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python -m py_compile \
            residuals/src/preprocessing/0019_cross_hisspikes_fit.py
        python residuals/src/preprocessing/0019_cross_hisspikes_fit.py \
            --run-dir "$1" \
            --his-results "$2" \
            --output "$3" \
            --device cuda \
            --fit-batch-size 2048 \
            --check-events 20000
    ' bash "$RUN_DIR" "$HIS_RESULTS" "$OUTPUT" &
CHILD_PID=$!
wait "$CHILD_PID"
