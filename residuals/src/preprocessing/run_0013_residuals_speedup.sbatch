#!/bin/bash
#SBATCH --job-name=rho0013-speedup
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN="$ROOT/residuals/runs/dataset1_p1/unwhitened_rho_0011_vectorized_full"
OUTPUT="$ROOT/residuals/runs/dataset1_p1/0013_rho_speedup/identity_fast_path_chunk100"
cd "$ROOT"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c 'source /ext3/env.sh && export PYTHONPATH=$PWD/residuals/src && python residuals/src/preprocessing/0013_residuals_speedup.py "$1" "$2"' \
    /bin/bash "$RUN" "$OUTPUT"
