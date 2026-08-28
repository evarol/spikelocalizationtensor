#!/bin/bash
#SBATCH --job-name=replot_global_q_energy
#SBATCH --output=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --account=torch_pr_60_general

set -euo pipefail

ROOT=/scratch/ap7151/iclr-paper/ANALYTIC_APPROACH
Q_VALUES=(8 16 24 32)
RUN_NAMES=(
    global_codebook_pursuit_q8_matched_16344946_0
    global_codebook_pursuit_q16_matched_16344946_1
    global_codebook_pursuit_q24_matched_16344946_2
    global_codebook_pursuit_q32_matched_16344946_3
)
Q=${Q_VALUES[$SLURM_ARRAY_TASK_ID]}
RUN=$ROOT/residuals/runs/dataset1_p1/${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
OUTPUT=$ROOT/residuals/out/global_codebook_per_q_16344946/q${Q}

if [[ ! -d "$RUN" || ! -d "$OUTPUT" ]]; then
    echo "missing run or output directory: $RUN $OUTPUT" >&2
    exit 2
fi

singularity exec --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && cd $ROOT && \
        for chunk_index in 0 1 2 3; do \
            python src/plots/plot_residual_pass_diagnostics.py \
                --run $RUN --chunk-index \$chunk_index \
                --out $OUTPUT/recording_energy_loss_after_each_pass_chunk\${chunk_index}.png; \
        done"

echo "Q=$Q energy plots replaced: $(date)"
