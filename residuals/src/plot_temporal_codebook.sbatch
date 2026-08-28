#!/bin/bash
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

KERNEL=${1:?kernel is required}
case "$KERNEL" in
    monopole|exponential|gauss|lorentz|power|student|yukawa|dog|gauss_aniso|mono_aniso) ;;
    *) echo "unsupported kernel: $KERNEL" >&2; exit 2 ;;
esac

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
FIT=$ROOT/residuals/runs/dataset1_p1/gpu_fit_voxel_1um_${KERNEL}.npz
OUT=$ROOT/residuals/out/plots/gpu_fit_voxel_1um_${KERNEL}/temporal_codebook.png

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_temporal_codebook.py --fit $FIT --out $OUT"
