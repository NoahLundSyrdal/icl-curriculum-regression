#!/usr/bin/env bash
#SBATCH --job-name=icl-summary
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/icl-summary-%j.out
#SBATCH --error=logs/icl-summary-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
.venv/bin/python summarize.py "results/${ICL_PRESET:-full}"
