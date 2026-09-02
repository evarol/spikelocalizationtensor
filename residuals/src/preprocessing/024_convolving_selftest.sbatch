#!/bin/bash
#SBATCH --job-name=convolving024-selftest
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RECORDING_PATH=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
SMOKE_CONV=$PROJECT_ROOT/residuals/runs/dataset1_p1/024_selftest_smoke_convolving
SMOKE_BIPOLAR=$PROJECT_ROOT/residuals/runs/dataset1_p1/024_selftest_smoke_bipolar

cd "$PROJECT_ROOT"
echo "024 validation: py_compile, CPU self-test, CUDA smokes"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        set -euo pipefail
        python -m py_compile \
            residuals/src/preprocessing/0016_onehot_lattice_peeling.py \
            residuals/src/preprocessing/0019_allchannel_peeling.py \
            residuals/src/preprocessing/024_convolving_detection_peeling.py
        echo "=== py_compile ok; running 024 CPU self-test ==="
        python residuals/src/preprocessing/024_convolving_detection_peeling.py \
            --self-test --device cpu
        echo "=== CUDA smoke: convolving growsum ==="
        python residuals/src/preprocessing/024_convolving_detection_peeling.py \
            all "$1" "$2" \
            --start-seconds 60 --duration-seconds 10 \
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
            --detector convolving \
            --score-merge growsum \
            --proposal-threshold 6.0 \
            --proposal-lockout 60
        echo "=== CUDA smoke: bipolar routing parity ==="
        python residuals/src/preprocessing/024_convolving_detection_peeling.py \
            all "$1" "$3" \
            --start-seconds 60 --duration-seconds 10 \
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
            --prototype-cone-deg 35
    ' bash "$RECORDING_PATH" "$SMOKE_CONV" "$SMOKE_BIPOLAR"
