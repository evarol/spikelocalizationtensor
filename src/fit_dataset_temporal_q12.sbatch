#!/bin/bash
#SBATCH --job-name=fit_temporal_q12
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
SESSION=$ROOT/runs/dataset1_p1
WAVEFORMS=$SESSION/neighborhood_waveforms.npy
RESULT=$SESSION/gpu_fit_voxel_1um_masked_monopole_q12.npz

if [[ ! -e "$WAVEFORMS" ]]; then
    echo "missing waveforms: $WAVEFORMS" >&2
    exit 2
fi
if [[ -e "$RESULT" ]]; then
    echo "result already exists: $RESULT" >&2
    exit 2
fi

echo "=== 90-sample, 12-generic temporal codebook fit ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Output: $RESULT"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && python - <<'PY'
import numpy as np

path = '$WAVEFORMS'
shape = np.load(path, mmap_mode='r').shape
if len(shape) != 3 or shape[2] != 90:
    raise SystemExit(f'expected (spikes, channels, 90) waveforms, got {shape}')
print(f'validated waveform array {shape}', flush=True)
PY
python src/fit_dataset.py \
    $SESSION $RESULT --device cuda --kernel monopole --q 12 \
    --n-scales 10 --n-sites 16 --n-iters 8 --refine-levels 6"

echo "Completed: $(date)"
