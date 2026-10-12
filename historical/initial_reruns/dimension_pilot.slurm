#!/usr/bin/env bash
#SBATCH --job-name=icl-dim-pilot
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%6
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
icl_task=${SLURM_ARRAY_TASK_ID:?}
if ((icl_task<0 || icl_task>5)); then exit 2; fi
icl_arms=(curriculum shuffled hard)
icl_seed=$((100+icl_task/3))
icl_arm=${icl_arms[$((icl_task%3))]}
icl_extra=()
if [[ "${ICL_DIM_RESUME:-0}" == 1 ]]; then icl_extra+=(--resume); fi
srun .venv/bin/python -u dimension_experiment.py --seed "$icl_seed" --arm "$icl_arm" "${icl_extra[@]}"
