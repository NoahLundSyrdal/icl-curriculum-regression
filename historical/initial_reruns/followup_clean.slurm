#!/usr/bin/env bash
#SBATCH --job-name=icl-clean
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%5
#SBATCH --signal=USR1@180
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
icl_extra=()
if [[ "${ICL_CLEAN_RESUME:-0}" == 1 ]]; then icl_extra+=(--resume); fi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'
srun .venv/bin/python -u clean_control.py --seed "${SLURM_ARRAY_TASK_ID:?}" "${icl_extra[@]}"
srun .venv/bin/python -u checkpoint_diagnostics.py --seed "$SLURM_ARRAY_TASK_ID" --clean
