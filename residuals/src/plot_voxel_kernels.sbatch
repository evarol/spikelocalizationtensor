#!/bin/bash
#SBATCH --job-name=plot_voxel_kernels
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
FIT=$SESSION/gpu_fit_voxel_1um_${KERNEL}.npz
OUT=$ROOT/residuals/out/plots/gpu_fit_voxel_1um_${KERNEL}
MONO=/scratch/ap7151/sln-v2/runs/dataset1_p1/monopolar_true/localizations.npy
RECORDING=/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1

if [[ ! -e "$FIT" ]]; then
    echo "fit does not exist: $FIT" >&2
    exit 2
fi
if [[ -e "$OUT" ]]; then
    echo "plot directory already exists: $OUT" >&2
    exit 2
fi

echo "=== 1 um voxel $KERNEL fit plots ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && \
        python src/plots/plot_gpu_fit.py --session $SESSION --fit $FIT --out $OUT && \
        python src/plots/plot_probe_global.py --session $SESSION --fit $FIT \
            --recording $RECORDING --out $OUT/probe_global.png && \
        python src/plots/plot_projection_histograms.py --session $SESSION --fit $FIT \
            --monopole $MONO --out $OUT/projections"

echo "Completed: $(date)"
