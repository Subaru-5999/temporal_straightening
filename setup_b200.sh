#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_b200.sh
# Adapts the `ts` conda env (built for H100 / CUDA 12.1 / Python 3.9) to run
# on an NVIDIA B200 (Blackwell, sm_100) pod with CUDA driver 12.8 / Python 3.10.
#
# Run this INSIDE the activated `ts` env on the pod:
#     conda activate ts
#     bash setup_b200.sh
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> Removing CUDA 12.1 runtime wheels that are incompatible with Blackwell..."
# These were pinned for CUDA 12.1 and cuDNN 8.9 (no sm_100 kernels).
# The correct CUDA 12.8 / cuDNN 9.x libs are pulled in as torch dependencies below.
pip uninstall -y \
  torch torchvision triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 || true

echo "==> Installing Blackwell-capable PyTorch (CUDA 12.8 build)..."
# torch 2.7.x is the first release with official Blackwell (sm_100) support.
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

echo "==> Verifying the GPU is visible to PyTorch..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (10, 0) on B200
PY

echo "==> Done. If capability prints (10, 0) and a B200 name, the stack matches the driver."
