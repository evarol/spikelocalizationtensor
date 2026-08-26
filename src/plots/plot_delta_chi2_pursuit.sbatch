#!/bin/bash
#SBATCH --job-name=plot_delta_chi2
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
BASELINE=$ROOT/runs/dataset1_p1/peak_channel_codebook_pursuit_q8_fixed_16358281_1
GATED=$ROOT/runs/dataset1_p1/peak_channel_q8_delta_chi2_16375246
OUTPUT=$ROOT/out/plots/peak_channel_q8_delta_chi2_16375246.png

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_delta_chi2_pursuit.py \
        --baseline $BASELINE --gated $GATED --min-delta-chi2 36 --out $OUTPUT"
