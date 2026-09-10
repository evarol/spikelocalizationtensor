#!/bin/bash
#SBATCH --job-name=0030-kilosort-macaque3
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RECORDING_PATH=$PROJECT_ROOT/residuals/runs/primate/0030_macaque3_spikeglx/macaque3_kilosort.imec0.ap.bin
RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/kilosort_iblsorter_macaque3${RUN_SUFFIX:+_$RUN_SUFFIX}
CHILD_PID=

handle_stop() {
    echo "signal received; stopping the active process and requeueing $SLURM_JOB_ID"
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

# TERM is caught too: the low-GPU-utilization policy kill (session 0025) sends
# SIGTERM directly, bypassing the USR1-only walltime trap.
trap handle_stop USR1 TERM

if [[ ! -e "$RECORDING_PATH" ]]; then
    echo "missing recording: $RECORDING_PATH" >&2
    exit 2
fi
if [[ -e "$RUN_OUTPUT/.complete" ]]; then
    echo "run already complete: $RUN_OUTPUT" >&2
    exit 0
fi

mkdir -p "$RUN_OUTPUT"
cd "$PROJECT_ROOT"
echo "kilosort iblsorter run: $RUN_OUTPUT (stop_after=${STOP_AFTER:-<none>})"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        export PYTHONUNBUFFERED=1
        python -m py_compile residuals/src/preprocessing/run_kilosort_iblsorter.py
        python residuals/src/preprocessing/run_kilosort_iblsorter.py \
            "$1" "$2" ${STOP_AFTER:+--stop-after $STOP_AFTER}
        touch "$2/.complete"
    ' bash "$RECORDING_PATH" "$RUN_OUTPUT" &
CHILD_PID=$!
wait "$CHILD_PID"
