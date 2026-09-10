#!/bin/bash
#SBATCH --job-name=0025-sidredge-peaks
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --account=torch_pr_60_tandon_priority

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RAW=/scratch/ap7151/_RAW_DATA/primate
: "${SUBJECT:?set SUBJECT=human1|macaque1|macaque2|macaque3}"

case "$SUBJECT" in
    human1)   RECORDING_PATH=$RAW/human1_sub-Pt01_ecephys.nwb; SERIES=acquisition/ElectricalSeriesRaw;;
    macaque1) RECORDING_PATH=$RAW/macaque1_sub-L11104_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAPImec;;
    macaque2) RECORDING_PATH=$RAW/macaque2_sub-C_20220121_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAP;;
    macaque3) RECORDING_PATH=$RAW/macaque3_sub-C_20220218_ecephys.nwb; SERIES=acquisition/ElectricalSeriesAP;;
    *) echo "unknown SUBJECT: $SUBJECT" >&2; exit 2;;
esac

RUN_OUTPUT=$PROJECT_ROOT/residuals/runs/primate/0025_sidredge_${SUBJECT}

cd "$PROJECT_ROOT"
echo "0025 SI dredge peaks stage: subject=$SUBJECT run=$RUN_OUTPUT"

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        python residuals/src/preprocessing/0025_sidredge.py \
            peaks \
            --recording-path "$1" \
            --electrical-series-path "$2" \
            --output-dir "$3" \
            --n-jobs 12
    ' bash "$RECORDING_PATH" "$SERIES" "$RUN_OUTPUT"