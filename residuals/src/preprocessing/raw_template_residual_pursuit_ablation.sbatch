#!/bin/bash
#SBATCH --job-name=raw_pursuit_ablation
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
INITIAL_OMEGA=$ROOT/runs/dataset1_p1/gpu_fit_voxel_1um_masked_monopole.npz
MODE=${MODE:?set MODE to frozen, learned, or learned_stopped}
CHUNK_SECONDS=2.187
DURATION_SECONDS=8.748

case "$MODE" in
    frozen)
        OMEGA_FIT=$INITIAL_OMEGA
        LEARNING_CHUNKS=0
        ENERGY_STOP=0
        ;;
    learned)
        OMEGA_FIT=$INITIAL_OMEGA
        LEARNING_CHUNKS=2
        ENERGY_STOP=0
        ;;
    learned_stopped)
        OMEGA_FIT=${LEARNED_OMEGA_PATH:?set LEARNED_OMEGA_PATH for learned_stopped}
        LEARNING_CHUNKS=0
        ENERGY_STOP=0.005
        ;;
    *)
        echo "unknown MODE: $MODE" >&2
        exit 2
        ;;
esac

OUTPUT=$ROOT/runs/dataset1_p1/raw_template_residual_${MODE}_65610_${SLURM_JOB_ID}

if [[ ! -e "$RECORDING" || ! -e "$OMEGA_FIT" ]]; then
    echo "missing recording or temporal codebook: $RECORDING $OMEGA_FIT" >&2
    exit 2
fi
if [[ -e "$OUTPUT" ]]; then
    echo "output already exists: $OUTPUT" >&2
    exit 2
fi

echo "=== IBL-style pursuit ablation: $MODE ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Chunk: 65,610 samples = 729 x 90 = $CHUNK_SECONDS seconds at 30 kHz"
echo "Duration: 262,440 samples = 4 chunks = $DURATION_SECONDS seconds"
echo "Learning chunks: $LEARNING_CHUNKS"
echo "Round energy stop: $ENERGY_STOP"
echo "Codebook input: $OMEGA_FIT"
echo "Output: $OUTPUT"
echo "Start: $(date)"

SECONDS=0
singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/raw_residual.py $RECORDING $OMEGA_FIT $OUTPUT \
        --duration-seconds $DURATION_SECONDS --device cuda \
        --kernel monopole --n-scales 10 --n-sites 16 --refine-levels 6 \
        --continuous-max-iterations 80 --continuous-backtracks 30 \
        --threshold 6 --min-captured-fraction 0.05 \
        --chunk-seconds $CHUNK_SECONDS --pursuit-rounds 60 \
        --pursuit-min-round-energy-drop-fraction $ENERGY_STOP \
        --max-peaks-per-round 10000 --fit-batch-size 1024 \
        --localization-config-batch-size 32 --template-time-batch 4096 \
        --codebook-learning-chunks $LEARNING_CHUNKS \
        --codebook-momentum 0.9 --codebook-min-events-per-row 32 \
        --codebook-learning-seed 42 --profile-stages"
ELAPSED_SECONDS=$SECONDS

printf '{\n  "mode": "%s",\n  "elapsed_seconds": %d\n}\n' \
    "$MODE" "$ELAPSED_SECONDS" > "$OUTPUT/runtime.json"
echo "Elapsed seconds: $ELAPSED_SECONDS"
echo "Completed: $(date)"
