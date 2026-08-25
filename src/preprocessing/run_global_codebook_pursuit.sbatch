#!/bin/bash
#SBATCH --job-name=global_q_pursuit
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --array=0-3

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
SESSION=$ROOT/runs/dataset1_p1
CODEBOOK_DIR=${CODEBOOK_DIR:?set CODEBOOK_DIR to the completed raw codebook run}
Q_VALUES=(8 16 24 32)
Q=${Q_OVERRIDE:-${Q_VALUES[$SLURM_ARRAY_TASK_ID]}}
THRESHOLD_MODE=${THRESHOLD_MODE:-matched}
CODEBOOK=$CODEBOOK_DIR/global_codebook_q${Q}.npz
CALIBRATION=$CODEBOOK_DIR/global_codebook_thresholds.json
OUTPUT=$SESSION/global_codebook_pursuit_q${Q}_${THRESHOLD_MODE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

if [[ ! "$Q" =~ ^(8|16|24|32)$ ]]; then
    echo "Q must be one of 8, 16, 24, or 32; got $Q" >&2
    exit 2
fi
if [[ ! "$THRESHOLD_MODE" =~ ^(fixed|matched)$ ]]; then
    echo "THRESHOLD_MODE must be fixed or matched; got $THRESHOLD_MODE" >&2
    exit 2
fi
if [[ ! -e "$RECORDING" || ! -e "$CODEBOOK" ]]; then
    echo "missing recording or codebook: $RECORDING $CODEBOOK" >&2
    exit 2
fi
if [[ "$THRESHOLD_MODE" == matched && ! -e "$CALIBRATION" ]]; then
    echo "missing threshold calibration: $CALIBRATION" >&2
    exit 2
fi
if [[ -e "$OUTPUT" ]]; then
    echo "output already exists: $OUTPUT" >&2
    exit 2
fi

echo "=== frozen global Q=$Q pursuit ($THRESHOLD_MODE threshold) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Codebook: $CODEBOOK"
echo "Output: $OUTPUT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/run_global_codebook_pursuit.py \
        $RECORDING $CODEBOOK $OUTPUT \
        --threshold-mode $THRESHOLD_MODE \
        --threshold-calibration $CALIBRATION --fixed-threshold 6 \
        --duration-seconds 8.748 --chunk-seconds 2.187 \
        --pursuit-rounds 60 --max-peaks-per-round 10000 \
        --fit-batch-size 1024 --localization-config-batch-size 32 \
        --template-time-batch 4096 --continuous-max-iterations 80 \
        --continuous-backtracks 30 --save-waveforms --profile-stages \
        --device cuda"

echo "Completed: $(date)"
