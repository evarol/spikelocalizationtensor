#!/bin/bash
#SBATCH --job-name=calibrate_global_q
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
SESSION=$ROOT/runs/dataset1_p1
OUTPUT=$SESSION/global_codebook_thresholds.json
CODEBOOKS=(
    "$SESSION/global_codebook_q8.npz"
    "$SESSION/global_codebook_q16.npz"
    "$SESSION/global_codebook_q24.npz"
    "$SESSION/global_codebook_q32.npz"
)

for path in "${CODEBOOKS[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "missing codebook: $path" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT" ]]; then
    echo "output already exists: $OUTPUT" >&2
    exit 2
fi

echo "=== matched-candidate threshold calibration ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Output: $OUTPUT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/calibrate_global_codebook_thresholds.py \
        $RECORDING $OUTPUT ${CODEBOOKS[*]} \
        --reference-q 8 --reference-threshold 6 \
        --start-seconds 0 --duration-seconds 8 --chunk-seconds 1 \
        --n-scales 10 --template-time-batch 4096 --device cuda"

echo "Completed: $(date)"
