#!/bin/bash
#SBATCH --job-name=compare_pursuit_ablation
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
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
FROZEN_JOB_ID=${FROZEN_JOB_ID:?set FROZEN_JOB_ID}
LEARNED_JOB_ID=${LEARNED_JOB_ID:?set LEARNED_JOB_ID}
STOPPED_JOB_ID=${STOPPED_JOB_ID:?set STOPPED_JOB_ID}

mkdir -p "$COMPARISON_OUTPUT"

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/compare_pursuit_ablation.py \
        $FROZEN_OUTPUT $LEARNED_OUTPUT $STOPPED_OUTPUT \
        --tolerance-samples 3 --expected-chunk-samples 65610 \
        --output $COMPARISON_OUTPUT/comparison.json"

sacct -j "$FROZEN_JOB_ID,$LEARNED_JOB_ID,$STOPPED_JOB_ID" \
    --format=JobID,JobName%32,State,Elapsed,AllocTRES%100,TRESUsageInAve%160,TRESUsageInMax%160 \
    -P > "$COMPARISON_OUTPUT/accounting.txt"
