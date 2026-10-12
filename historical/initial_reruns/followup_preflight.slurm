#!/usr/bin/env bash
#SBATCH --job-name=icl-h1-checks
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
.venv/bin/python -m unittest -v test_followup.py
.venv/bin/python checkpoint_diagnostics.py --prepare

