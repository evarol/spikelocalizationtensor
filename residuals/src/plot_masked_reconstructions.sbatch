#!/bin/bash
#SBATCH --job-name=plot_masked_recon
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

KERNELS=(monopole exponential gauss lorentz power student yukawa dog gauss_aniso mono_aniso)
KERNEL=${KERNELS[$SLURM_ARRAY_TASK_ID]}
ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/residuals/runs/dataset1_p1
FIT=$SESSION/gpu_fit_voxel_1um_masked_${KERNEL}.npz
OUT=$ROOT/residuals/out/plots/gpu_fit_voxel_1um_masked_${KERNEL}/reconstruction

if [[ ! -e "$FIT" ]]; then
    echo "fit does not exist: $FIT" >&2
    exit 2
fi
if [[ -e "$OUT" ]]; then
    echo "reconstruction plot directory already exists: $OUT" >&2
    exit 2
fi

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python \
        src/plots/plot_reconstructions.py --session $SESSION --fit $FIT \
        --kernel $KERNEL --out $OUT"
