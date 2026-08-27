#!/bin/bash
#SBATCH --job-name=plot_pursuit_ablation
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
FROZEN_OUTPUT=${FROZEN_OUTPUT:?set FROZEN_OUTPUT}
LEARNED_OUTPUT=${LEARNED_OUTPUT:?set LEARNED_OUTPUT}
STOPPED_OUTPUT=${STOPPED_OUTPUT:?set STOPPED_OUTPUT}
COMPARISON_OUTPUT=${COMPARISON_OUTPUT:?set COMPARISON_OUTPUT}

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_pursuit_ablation.py \
        --frozen $FROZEN_OUTPUT \
        --learned $LEARNED_OUTPUT \
        --stopped $STOPPED_OUTPUT \
        --comparison $COMPARISON_OUTPUT/comparison.json \
        --out-dir $COMPARISON_OUTPUT/plots"
