#!/bin/bash
#SBATCH --job-name=threshold0017-full
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RECORDING_PATH=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
OMEGA_PRIOR_PATH=${OMEGA_PRIOR_PATH:-}
if [[ -n "$OMEGA_PRIOR_PATH" ]]; then
    RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/dataset1_p1/0017_initial_threshold6_spiketensor_prior_fitted8
else
    RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/dataset1_p1/0017_initial_threshold6_learned_fitted8
fi
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

if [[ ! -e "$RECORDING_PATH" ]]; then
    echo "missing recording: $RECORDING_PATH" >&2
    exit 2
fi
if [[ -n "$OMEGA_PRIOR_PATH" && ! -e "$OMEGA_PRIOR_PATH" ]]; then
    echo "missing Omega prior: $OMEGA_PRIOR_PATH" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
echo "full-recording 0017 initial threshold discovery: $RUN_OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python -m py_compile \
            residuals/src/preprocessing/0016_onehot_lattice_peeling.py \
            residuals/src/preprocessing/0017_initial_threshold_peeling.py
        prior_args=()
        if [[ -n "$3" ]]; then
            prior_args=(--omega-prior "$3")
        fi
        python residuals/src/preprocessing/0017_initial_threshold_peeling.py \
            all "$1" "$2" \
            --threshold 6 \
            --discovery-temporal-radius-samples 5 \
            --min-fitted-projection 8 \
            --peeling-rounds 60 \
            --event-merge-ms 0.5 \
            --min-improved-channels 2 \
            --min-channel-improvement-fraction 0.05 \
            --max-channel-normalized-rmse 3 \
            --fit-batch-size 2048 \
            --chunk-seconds 1 \
            --resume \
            "${prior_args[@]}"
    ' bash "$RECORDING_PATH" "$RUN_OUTPUT" "$OMEGA_PRIOR_PATH" &
CHILD_PID=$!
wait "$CHILD_PID"
