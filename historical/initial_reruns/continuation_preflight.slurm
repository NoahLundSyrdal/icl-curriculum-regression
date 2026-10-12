#!/usr/bin/env bash
#SBATCH --job-name=icl-tail-checks
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python -m unittest -v test_continuation.py test_summary_continuation.py
.venv/bin/python preflight_continuation.py --source results/full \
    --output "submissions/continuation-preflight-${ICL_TAIL_STEPS:-10000}.json"
