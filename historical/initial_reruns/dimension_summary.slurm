#!/usr/bin/env bash
#SBATCH --job-name=icl-dim-summary
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
.venv/bin/python summarize_dimension.py
