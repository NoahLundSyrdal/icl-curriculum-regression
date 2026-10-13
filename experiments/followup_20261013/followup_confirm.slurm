#!/usr/bin/env bash
#SBATCH --job-name=icl-next-confirm
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --array=0-359%15
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
icl_extra=()
if [[ "${ICL_FOLLOWUP_RESUME:-0}" == 1 ]]; then icl_extra+=(--resume); fi
srun .venv/bin/python -u followup_experiment.py --cohort confirmation --task-index "${SLURM_ARRAY_TASK_ID:?}" "${icl_extra[@]}"
