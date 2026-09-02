#!/bin/bash
#SBATCH --job-name=plot0022-kilosort
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
RUN_PATH=$PROJECT_ROOT/residuals/runs/dataset1_p1/kilosort_iblsorter
PLOT_OUTPUT=$PROJECT_ROOT/residuals/out/0022_kilosort_baseline
KILOSORT_JOB_ID=16740477

if [[ ! -e "$RUN_PATH/.complete" ]]; then
    echo "missing completed kilosort run: $RUN_PATH" >&2
    exit 2
fi

mkdir -p "$PLOT_OUTPUT"
cd "$PROJECT_ROOT"

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
        set -euo pipefail
        source /ext3/env.sh
        python -m py_compile residuals/src/plots/kilosort_baseline_plots.py
        python residuals/src/plots/kilosort_baseline_plots.py \
            --run "$1" --out "$2" --job-id "$3" --overwrite
    ' bash "$RUN_PATH" "$PLOT_OUTPUT" "$KILOSORT_JOB_ID"

echo "0022 kilosort plot suite complete: $PLOT_OUTPUT"
