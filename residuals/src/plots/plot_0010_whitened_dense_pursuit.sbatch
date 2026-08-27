#!/bin/bash
#SBATCH --job-name=plot_0010_dense
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

RUN_ID=${1:?pass the parent pursuit job id}
ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN=$ROOT/residuals/runs/dataset1_p1/whitened_dense_0010_${RUN_ID}

singularity exec --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python residuals/src/plots/plot_0010_whitened_dense_pursuit.py --run $RUN --out $RUN/plots/summary.png"
