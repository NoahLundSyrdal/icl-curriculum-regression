#!/usr/bin/env bash
#SBATCH --job-name=icl-order
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment directory}"

# This Python module was verified on IDUN on 5 October 2026. Override as needed.
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
ICL_PYTHON="${ICL_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
ICL_PRESET="${ICL_PRESET:-pilot}"
ICL_RESUME="${ICL_RESUME:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ ! -x "$ICL_PYTHON" ]]; then
    echo "Set ICL_PYTHON to your CUDA-enabled Python interpreter." >&2
    exit 2
fi
if [[ "$ICL_PRESET" != "smoke" && "$ICL_PRESET" != "pilot" && "$ICL_PRESET" != "full" ]]; then
    echo "ICL_PRESET must be smoke, pilot, or full." >&2
    exit 2
fi

icl_modes=(clean curriculum shuffled reverse)
icl_task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( icl_task_id < 0 || icl_task_id > 19 )); then
    echo "Array indices must be in 0..19 (five seeds, four modes)." >&2
    exit 2
fi
icl_seed=$((icl_task_id / 4))
icl_mode="${icl_modes[$((icl_task_id % 4))]}"
icl_extra=()
if [[ "$ICL_RESUME" == "1" ]]; then icl_extra+=(--resume); fi

nvidia-smi
"$ICL_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
srun "$ICL_PYTHON" -u run_experiment.py \
    --preset "$ICL_PRESET" --mode "$icl_mode" --seed "$icl_seed" \
    --device cuda --output "results/$ICL_PRESET" "${icl_extra[@]}"
