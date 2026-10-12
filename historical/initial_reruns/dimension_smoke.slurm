#!/usr/bin/env bash
#SBATCH --job-name=icl-dim-smoke
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
srun .venv/bin/python -u dimension_smoke.py
