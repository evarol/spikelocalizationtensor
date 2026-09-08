#!/bin/bash
#SBATCH --job-name=allch0019-trimsweep
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RECORDING_PATH=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
: "${SLURM_ARRAY_TASK_ID:?must run as a SLURM array job}"
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

trap handle_usr1 USR1 TERM

if [[ ! -e "$RECORDING_PATH" ]]; then
    echo "missing recording: $RECORDING_PATH" >&2
    exit 2
fi

RULES=(min min min min min mean mean mean mean mean kofn kofn kofn kofn kofn flat flat flat flat flat)
F0S=(0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30 0.05 0.10 0.15 0.20 0.30)
TAGS=(05 10 15 20 30 05 10 15 20 30 05 10 15 20 30 05 10 15 20 30)

i=$SLURM_ARRAY_TASK_ID
RULE=${RULES[$i]}
F0=${F0S[$i]}
TAG=${TAGS[$i]}

case "$RULE" in
    min)  EXTRA=(--all-channel-min-fraction "$F0" --pass-fraction-step 0.1);;
    mean) EXTRA=(--all-channel-rule mean-channel --all-channel-min-fraction "$F0" --pass-fraction-step 0.1);;
    kofn) EXTRA=(--all-channel-rule k-of-n --all-channel-required-share 0.875 --all-channel-min-fraction "$F0" --pass-fraction-step 0.1);;
    flat) EXTRA=(--all-channel-min-fraction "$F0" --pass-fraction-step 0.0);;
    *) echo "unknown RULE: $RULE" >&2; exit 2;;
esac

RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/dataset1_p1/0019_allchannel_trimmed_${RULE}_f0${TAG}_q32

cd "$PROJECT_ROOT"
echo "trimmed-gate 0019 f0 sweep: rule=$RULE f0=$F0 q=32 run=$RUN_OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python -m py_compile \
            residuals/src/preprocessing/0016_onehot_lattice_peeling.py \
            residuals/src/preprocessing/0019_allchannel_peeling.py
        python residuals/src/preprocessing/0019_allchannel_peeling.py \
            all "$1" "$2" \
            --threshold 5 \
            --exclude-sweep-ms 1 \
            --min-fitted-projection 8 \
            --peeling-rounds 1 \
            --recording-passes 3 \
            --spatial-score mean-channel-rmse \
            --all-channel-improvement \
            --event-merge-ms 0.5 \
            --max-channel-normalized-rmse 3 \
            --fit-batch-size 2048 \
            --chunk-seconds 1 \
            --prototype-count 2 \
            --prototype-cone-deg 35 \
            --q 32 \
            --resume \
            "${@:3}"
    ' bash "$RECORDING_PATH" "$RUN_OUTPUT" "${EXTRA[@]}" &
CHILD_PID=$!
wait "$CHILD_PID"
