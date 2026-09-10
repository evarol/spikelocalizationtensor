#!/bin/bash
#SBATCH --job-name=codebook-0021
#SBATCH --output=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/ap7151/iclr-paper/varol-version/residuals/slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_60_general
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=/scratch/ap7151/iclr-paper/varol-version
CHILD_PID=

handle_usr1() {
    echo "USR1/TERM received; stopping the active process and requeueing $SLURM_JOB_ID"
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

trap handle_usr1 USR1 TERM

cd "$PROJECT_ROOT"
VARIANT=${VARIANT:-4prior_shift}
Q=${Q:-32}
CONE=${CONE:-35}

case "$VARIANT" in
    2prior)          PRIORS=2; SHIFT=--no-shift-learning; FREEZE=--no-freeze-prototypes; TAG=q${Q}_p2_fixed ;;
    2prior_shift)    PRIORS=2; SHIFT=--shift-learning;    FREEZE=--no-freeze-prototypes; TAG=q${Q}_p2_shift ;;
    4prior)          PRIORS=4; SHIFT=--no-shift-learning; FREEZE=--no-freeze-prototypes; TAG=q${Q}_p4_fixed ;;
    4prior_tight)    PRIORS=4; SHIFT=--no-shift-learning; FREEZE=--freeze-prototypes;    TAG=q${Q}_p4f${CONE}_fixed ;;
    4prior_shift_tight) PRIORS=4; SHIFT=--shift-learning; FREEZE=--freeze-prototypes;    TAG=q${Q}_p4f${CONE}_shift ;;
    4prior_shift|*)  PRIORS=4; SHIFT=--shift-learning;    FREEZE=--no-freeze-prototypes; TAG=q${Q}_p4_shift ;;
esac

echo "0021 master codebook variant=$VARIANT q=$Q priors=$PRIORS cone=${CONE}deg freeze=${FREEZE} output=residuals/runs/ibl_bwm/master_codebook_0021_$TAG"

singularity exec --nv --overlay /scratch/${USER}/_ENVS/ibl-sorter.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
        python -m py_compile residuals/src/preprocessing/0021_shift_invariant_codebook.py
        python residuals/src/preprocessing/0021_shift_invariant_codebook.py \
            all \
            --shards residuals/runs/ibl_bwm/master_codebook_q32/shards \
            --output residuals/runs/ibl_bwm/master_codebook_0021_$TAG \
            --q $Q \
            --priors $PRIORS \
            --prototype-cone-deg $CONE \
            $FREEZE \
            $SHIFT \
            --resume
    " &
CHILD_PID=$!
wait "$CHILD_PID"
