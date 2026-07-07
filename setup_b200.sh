#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_b200.sh
# One-shot bootstrap that adapts this project to run on the NVIDIA B200 pod.
#
#   GPU     : NVIDIA B200 (Blackwell, sm_100), MIG ~45 GB
#   Driver  : 570.124.06  /  CUDA driver 12.8
#   Python  : 3.10  (conda env `ts`)
#   Project : /arun/temporal-straightening
#   Data    : /arun/data   (point_maze at /arun/data/point_maze)
#
# Usage (inside the activated conda env, from the project root):
#     conda activate ts          # or a clone: conda create --name ts_b200 --clone ts
#     bash setup_b200.sh
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/arun/data}"

echo "==> [1/4] Removing CUDA 12.1 wheels that are incompatible with Blackwell..."
# Pinned for CUDA 12.1 / cuDNN 8.9 -> no sm_100 kernels. The correct CUDA 12.8 /
# cuDNN 9.x libraries are pulled in as dependencies of the torch install below.
pip uninstall -y \
  torch torchvision triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 || true

echo "==> [2/4] Installing Blackwell-capable PyTorch (CUDA 12.8 build)..."
# torch 2.7.x is the first release with official Blackwell (sm_100) support.
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

echo "==> [3/4] Configuring DATASET_DIR=${DATASET_ROOT} ..."
export DATASET_DIR="${DATASET_ROOT}"
if ! grep -qs "export DATASET_DIR=${DATASET_ROOT}" "${HOME}/.bashrc" 2>/dev/null; then
  echo "export DATASET_DIR=${DATASET_ROOT}" >> "${HOME}/.bashrc"
  echo "    added DATASET_DIR to ~/.bashrc (persists across sessions)"
fi
if [ ! -d "${DATASET_ROOT}/point_maze" ]; then
  echo "    WARNING: ${DATASET_ROOT}/point_maze not found -- check the dataset path."
fi

echo "==> [4/4] Verifying the GPU is visible to PyTorch..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (10, 0) on B200
PY

cat <<'EOF'

==> Setup complete. If capability printed (10, 0) with a B200 device name, you're good.

Next steps:
  export DATASET_DIR=/arun/data        # already added to ~/.bashrc

  # quick smoke test (short run) to confirm data loads and the 45 GB slice holds:
  python train.py --config-name train.yaml env=point_maze \
      training.epochs=1 training.save_every_x_iterations=50 debug=True

  # full paper run:
  python train.py --config-name train.yaml env=point_maze
EOF
