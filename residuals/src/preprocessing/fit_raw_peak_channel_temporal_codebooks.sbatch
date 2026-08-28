#!/bin/bash
#SBATCH --job-name=fit_peak_channel_q
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
SESSION=$ROOT/residuals/runs/dataset1_p1
OUTPUT_DIR=${OUTPUT_DIR:-$SESSION/raw_peak_channel_codebooks_${SLURM_JOB_ID}}

if [[ ! -e "$RECORDING" ]]; then
    echo "missing raw recording: $RECORDING" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "output already exists: $OUTPUT_DIR" >&2
    exit 2
fi

echo "=== fresh peak-channel temporal codebook fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Recording: $RECORDING"
echo "Training interval: reproducibly sampled chunks in [60 seconds, recording end)"
echo "Q values: 4 8 16 32 64"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/fit_raw_peak_channel_temporal_codebooks.py \
        $RECORDING $OUTPUT_DIR --q-values 4 8 16 32 64 \
        --start-seconds 60 --chunk-seconds 1 --detection-threshold 6 \
        --isolation-ms 1 --max-events 100000 --max-events-per-chunk 2048 \
        --seed 42 --n-iters 10 --assignment-batch-size 65536 --device cuda"

echo "Completed: $(date)"
