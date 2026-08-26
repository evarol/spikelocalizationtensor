#!/bin/bash
#SBATCH --job-name=peak_q8_delta_chi2
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin
CODEBOOK=$ROOT/runs/dataset1_p1/raw_peak_channel_codebooks_16358267/global_codebook_q8.npz
OUTPUT=$ROOT/runs/dataset1_p1/peak_channel_q8_delta_chi2_${SLURM_JOB_ID}

if [[ ! -e "$RECORDING" || ! -e "$CODEBOOK" ]]; then
    echo "missing recording or Q8 codebook" >&2
    exit 2
fi

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/run_global_codebook_pursuit.py \
        $RECORDING $CODEBOOK $OUTPUT \
        --threshold-mode fixed --fixed-threshold 6 \
        --duration-seconds 8.748 --chunk-seconds 2.187 \
        --pursuit-rounds 60 --pursuit-min-round-energy-drop-fraction 0.002 \
        --min-captured-fraction 0 --min-delta-chi2 36 \
        --max-peaks-per-round 10000 --fit-batch-size 1024 \
        --localization-config-batch-size 32 --template-time-batch 4096 \
        --continuous-max-iterations 80 --continuous-backtracks 30 \
        --save-waveforms --profile-stages --device cuda"
