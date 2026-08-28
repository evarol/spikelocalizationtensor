#!/bin/bash
#SBATCH --job-name=raw_residual_pursuit_q8
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
OMEGA_FIT=$ROOT/residuals/runs/dataset1_p1/gpu_fit_voxel_1um_masked_monopole.npz
DURATION_SECONDS=${DURATION_SECONDS:-4}
LEARNING_CHUNKS=${LEARNING_CHUNKS:-2}
OUTPUT=$ROOT/residuals/runs/dataset1_p1/raw_template_residual_pursuit_q8_${SLURM_JOB_ID}

if [[ ! -e "$RECORDING" || ! -e "$OMEGA_FIT" ]]; then
    echo "missing recording or temporal-codebook fit" >&2
    exit 2
fi
if [[ -e "$OUTPUT" ]]; then
    echo "output already exists: $OUTPUT" >&2
    exit 2
fi

echo "=== IBL-style pursuit with learned Q8 temporal codebook ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Duration: $DURATION_SECONDS seconds"
echo "Learning chunks: $LEARNING_CHUNKS"
echo "Output: $OUTPUT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/raw_residual.py $RECORDING $OMEGA_FIT $OUTPUT \
        --duration-seconds $DURATION_SECONDS --device cuda \
        --kernel monopole --n-scales 10 --n-sites 16 --refine-levels 6 \
        --continuous-max-iterations 80 --continuous-backtracks 30 \
        --threshold 6 --min-captured-fraction 0.05 \
        --chunk-seconds 1 --pursuit-rounds 60 \
        --max-peaks-per-round 10000 --fit-batch-size 1024 \
        --localization-config-batch-size 32 --template-time-batch 4096 \
        --codebook-learning-chunks $LEARNING_CHUNKS \
        --codebook-momentum 0.9 --codebook-min-events-per-row 32 \
        --codebook-learning-seed 42 --profile-stages"

echo "Completed: $(date)"
