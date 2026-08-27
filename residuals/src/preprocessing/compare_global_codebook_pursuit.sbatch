#!/bin/bash
#SBATCH --job-name=compare_global_q
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
RUN_Q8=${RUN_Q8:?set RUN_Q8}
RUN_Q16=${RUN_Q16:?set RUN_Q16}
RUN_Q24=${RUN_Q24:?set RUN_Q24}
RUN_Q32=${RUN_Q32:?set RUN_Q32}
OUTPUT=${OUTPUT:?set OUTPUT}

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/preprocessing/compare_global_codebook_pursuit.py \
        $RUN_Q8 $RUN_Q16 $RUN_Q24 $RUN_Q32 \
        --reference-q 8 --tolerance-samples 3 --output $OUTPUT"
