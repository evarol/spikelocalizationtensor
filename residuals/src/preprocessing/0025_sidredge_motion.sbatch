#!/bin/bash
#SBATCH --job-name=0025-sidredge-motion
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RAW=/scratch/ap7151/_RAW_DATA/primate
: "${SUBJECT:?set SUBJECT=human1|macaque1|macaque2|macaque3}"
: "${MAXDISP:=2500}"
CHILD_PID=

case "$SUBJECT" in
    human1)   RECORDING_PATH=$RAW/human1_sub-Pt01_ecephys.nwb; SERIES=acquisition/ElectricalSeriesRaw;;
    macaque1) RECORDING_PATH=$RAW/macaque1_sub-L11104_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAPImec;;
    macaque2) RECORDING_PATH=$RAW/macaque2_sub-C_20220121_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAP;;
    macaque3) RECORDING_PATH=$RAW/macaque3_sub-C_20220218_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAP;;
    *) echo "unknown SUBJECT: $SUBJECT" >&2; exit 2;;
esac

RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/0025_sidredge_${SUBJECT}
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/0025_sidredge_${SUBJECT}

if [[ ! -f "$RUN_OUTPUT/_peaks_done" ]]; then
    echo "peaks stage not complete: $RUN_OUTPUT" >&2
    exit 2
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

cd "$PROJECT_ROOT"
echo "0025 SI dredge motion stage: subject=$SUBJECT run=$RUN_OUTPUT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        set -e
        source /ext3/env.sh
        python -m py_compile residuals/src/preprocessing/0025_sidredge.py
        python residuals/src/preprocessing/0025_sidredge.py \
            motion \
            --recording-path "$1" \
            --electrical-series-path "$2" \
            --output-dir "$3" \
            --max-disp-um "$5" \
            --device cuda
        python residuals/src/preprocessing/0025_sidredge.py \
            raster \
            --output-dir "$3" \
            --plot-dir "$4" \
            --max-disp-um "$5"
    ' bash "$RECORDING_PATH" "$SERIES" "$RUN_OUTPUT" "$PLOT_OUTPUT" "$MAXDISP" &
CHILD_PID=$!
wait "$CHILD_PID"