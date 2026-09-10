#!/bin/bash
#SBATCH --job-name=0025-dredge-prep
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RAW=/scratch/ap7151/_RAW_DATA/primate
: "${SUBJECT:?set SUBJECT=human1|macaque1|macaque2|macaque3}"
: "${VARIANT:=base}"

case "$SUBJECT" in
    human1)   RECORDING_PATH=$RAW/human1_sub-Pt01_ecephys.nwb;;
    macaque1) RECORDING_PATH=$RAW/macaque1_sub-L11104_ecephys.nwb;;
    macaque2) RECORDING_PATH=$RAW/macaque2_sub-C_20220121_ecephys.nwb;;
    macaque3) RECORDING_PATH=$RAW/macaque3_sub-C_20220218_ecephys.nwb;;
    *) echo "unknown SUBJECT: $SUBJECT" >&2; exit 2;;
esac

RESIDUAL_RUN=$PROJECT_ROOT/residuals/runs/primate/0025_${SUBJECT}_0019_${VARIANT}_q32
RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/0025_dredge_ap_${SUBJECT}_${VARIANT}

if [[ ! -f "$RESIDUAL_RUN/global_sources.npy" ]]; then
    echo "missing residual run: $RESIDUAL_RUN" >&2
    exit 2
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
echo "0025 dredge prepare stage: subject=$SUBJECT variant=$VARIANT run=$RUN_OUTPUT"

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python residuals/src/preprocessing/0025_dredge_ap.py \
            prepare \
            --residual-run "$1" \
            --output-dir "$2"
    ' bash "$RESIDUAL_RUN" "$RUN_OUTPUT"