#!/usr/bin/env bash
#SBATCH --job-name=icl-cont-summary
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/icl-cont-summary-%j.out
#SBATCH --error=logs/icl-cont-summary-%j.err

# Submit from the experiment directory after mkdir -p logs, passing --account.
# Use an afterok dependency covering all continuation array tasks.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
ICL_PYTHON="${ICL_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
"$ICL_PYTHON" summarize_continuation.py "${ICL_TAIL_OUTPUT:-results/continuation_10000}" \
    --source-directory "${ICL_SOURCE_OUTPUT:-results/full}"
