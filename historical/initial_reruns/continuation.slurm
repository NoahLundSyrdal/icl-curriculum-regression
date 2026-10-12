#!/usr/bin/env bash
#SBATCH --job-name=icl-tail
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%20
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment directory}"
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
ICL_PYTHON="${ICL_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
ICL_TAIL_STEPS="${ICL_TAIL_STEPS:-10000}"
ICL_TAIL_OUTPUT="${ICL_TAIL_OUTPUT:-results/continuation_$ICL_TAIL_STEPS}"
ICL_TAIL_RESUME="${ICL_TAIL_RESUME:-0}"
ICL_TAIL_SOURCE="${ICL_TAIL_SOURCE:-results/full}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ ! -x "$ICL_PYTHON" ]]; then
    echo "ICL_PYTHON must point to the verified CUDA Python environment." >&2
    exit 2
fi
icl_task_id="${SLURM_ARRAY_TASK_ID:?This script requires an array index}"
if (( icl_task_id < 0 || icl_task_id > 19 )); then
    echo "Array indices must be in 0..19." >&2
    exit 2
fi
icl_histories=(curriculum shuffled)
icl_tails=(high mixed)
icl_seed=$((icl_task_id / 4))
icl_history="${icl_histories[$(((icl_task_id % 4) / 2))]}"
icl_tail="${icl_tails[$((icl_task_id % 2))]}"
icl_extra=()
if [[ "$ICL_TAIL_RESUME" == "1" ]]; then icl_extra+=(--resume); fi

nvidia-smi
"$ICL_PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))'
srun "$ICL_PYTHON" -u continue_experiment.py \
    --source-root "$ICL_TAIL_SOURCE" --output "$ICL_TAIL_OUTPUT" \
    --history "$icl_history" --tail "$icl_tail" --seed "$icl_seed" \
    --steps "$ICL_TAIL_STEPS" --device cuda "${icl_extra[@]}"
