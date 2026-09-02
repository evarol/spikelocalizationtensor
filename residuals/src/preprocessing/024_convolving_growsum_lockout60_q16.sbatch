#!/bin/bash
#SBATCH --job-name=convolving024-growsum-lockout60-q16
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
RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/dataset1_p1/024_convolving_growsum_lockout60_q16
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

cd "$PROJECT_ROOT"
echo "full-recording 024 convolving growsum lockout-60 pursuit: $RUN_OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python -m py_compile \
            residuals/src/preprocessing/0016_onehot_lattice_peeling.py \
            residuals/src/preprocessing/0019_allchannel_peeling.py \
            residuals/src/preprocessing/024_convolving_detection_peeling.py
        python residuals/src/preprocessing/024_convolving_detection_peeling.py \
            all "$1" "$2" \
            --threshold 5 \
            --exclude-sweep-ms 1 \
            --min-fitted-projection 8 \
            --peeling-rounds 1 \
            --recording-passes 3 \
            --spatial-score mean-channel-rmse \
            --all-channel-improvement \
            --all-channel-min-fraction 0.2 \
            --pass-fraction-step 0.1 \
            --event-merge-ms 0.5 \
            --max-channel-normalized-rmse 3 \
            --fit-batch-size 2048 \
            --chunk-seconds 1 \
            --prototype-count 2 \
            --prototype-cone-deg 35 \
            --detector convolving \
            --q 16 \
            --score-merge growsum \
            --growsum-channels 7 \
            --proposal-threshold 6.0 \
            --proposal-lockout 60 \
            --resume
    ' bash "$RECORDING_PATH" "$RUN_OUTPUT" &
CHILD_PID=$!
wait "$CHILD_PID"
