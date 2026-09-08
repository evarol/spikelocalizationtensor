#!/bin/bash
#SBATCH --job-name=0025-primate-run
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
RAW=/scratch/ap7151/_RAW_DATA/primate
: "${SUBJECT:?set SUBJECT=human1|macaque1|macaque2|macaque3}"
: "${VARIANT:?set VARIANT=base|mean20}"
: "${SMOKE:=0}"
CHILD_PID=

case "$SUBJECT" in
    human1)   RECORDING_PATH=$RAW/human1_sub-Pt01_ecephys.nwb;;
    macaque1) RECORDING_PATH=$RAW/macaque1_sub-L11104_ecephys.nwb;;
    macaque2) RECORDING_PATH=$RAW/macaque2_sub-C_20220121_ecephys.nwb;;
    macaque3) RECORDING_PATH=$RAW/macaque3_sub-C_20220218_ecephys.nwb;;
    *) echo "unknown SUBJECT: $SUBJECT" >&2; exit 2;;
esac

case "$VARIANT" in
    base)   EXTRA=();;
    mean20) EXTRA=(--all-channel-rule mean-channel);;
    *) echo "unknown VARIANT: $VARIANT" >&2; exit 2;;
esac

if [[ "$SMOKE" == "1" ]]; then
    RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/0025_${SUBJECT}_smoke_${VARIANT}_q32
    DURATION=(--duration-seconds 30)
else
    RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/0025_${SUBJECT}_0019_${VARIANT}_q32
    DURATION=()
fi

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

if [[ ! -e "$RECORDING_PATH" ]]; then
    echo "missing recording: $RECORDING_PATH" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
echo "0025 primate run: subject=$SUBJECT variant=$VARIANT smoke=$SMOKE run=$RUN_OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python -m py_compile residuals/src/preprocessing/0025_primate_peeling.py
        python residuals/src/preprocessing/0025_primate_peeling.py \
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
            --q 32 \
            --resume \
            "${@:3}"
    ' bash "$RECORDING_PATH" "$RUN_OUTPUT" "${EXTRA[@]}" "${DURATION[@]}" &
CHILD_PID=$!
wait "$CHILD_PID"