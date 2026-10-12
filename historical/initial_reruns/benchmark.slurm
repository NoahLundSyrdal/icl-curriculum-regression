#!/usr/bin/env bash
#SBATCH --job-name=icl-benchmark
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/icl-benchmark-%j.out
#SBATCH --error=logs/icl-benchmark-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun .venv/bin/python -u benchmark.py
