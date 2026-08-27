#!/bin/bash
#SBATCH --job-name=whiten_spike_examples
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RUN=${RUN:-$ROOT/runs/dataset1_p1/raw_template_residual_batched_4s_16257633}
CHUNK_INDEX=${CHUNK_INDEX:-0}
SEED=${SEED:-42}
OUT=${OUT:-$ROOT/out/plots/whitening_spike_examples_4s_${SLURM_JOB_ID}.png}

echo "=== whitening diagnostic for poor reconstruction examples ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Run: $RUN"
echo "Chunk index: $CHUNK_INDEX"
echo "Seed: $SEED"
echo "Output: $OUT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_whitening_spike_examples.py \
        --run $RUN --chunk-index $CHUNK_INDEX --seed $SEED \
        --out $OUT"

echo "Completed: $(date)"
