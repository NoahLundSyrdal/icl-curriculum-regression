#!/usr/bin/env bash
set -euo pipefail
module purge
module load "${ICL_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p logs
if [[ ! -x .venv/bin/python ]]; then python -m venv .venv; fi
.venv/bin/python -m pip install --quiet torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --quiet -r requirements.txt
.venv/bin/python -c 'import torch, transformers; print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__, "CUDA wheel:", torch.version.cuda)'
