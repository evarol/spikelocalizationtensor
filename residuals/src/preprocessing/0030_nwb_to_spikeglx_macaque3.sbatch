#!/bin/bash
#SBATCH --job-name=0030-nwb-to-spikeglx-macaque3
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=torch_pr_60_tandon_priority

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
NWB_PATH=/scratch/ap7151/_RAW_DATA/primate/macaque3_sub-C_20220218_ecephys.nwb
OUT_DIR=$PROJECT_ROOT/residuals/runs/primate/0030_macaque3_spikeglx
OUT_BIN=$OUT_DIR/macaque3_kilosort.imec0.ap.bin

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

if [[ -e "$OUT_BIN" ]]; then
    echo "already converted: $OUT_BIN" >&2
    exit 0
fi

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        export PYTHONUNBUFFERED=1
        python -m py_compile residuals/src/preprocessing/0030_nwb_to_spikeglx.py
        python residuals/src/preprocessing/0030_nwb_to_spikeglx.py \
            "$1" "$2" --electrical-series-path acquisition/ElectricalSeriesAP
    ' bash "$NWB_PATH" "$OUT_BIN"
