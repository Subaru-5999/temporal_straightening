#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_b200.sh
# One-shot bootstrap that adapts this project to run on the NVIDIA B200 pod.
#
#   GPU     : NVIDIA B200 (Blackwell, sm_100), MIG ~45 GB
#   Driver  : 570.124.06  /  CUDA driver 12.8
#   Python  : 3.10  (conda env `ts`)
#   Project : /workspace/arun/temporal-straightening
#   Data    : /workspace/arun/data   (point_maze at /workspace/arun/data/point_maze)
#
# Usage (inside the activated conda env, from the project root):
#     conda activate ts          # or a clone: conda create --name ts_b200 --clone ts
#     bash setup_b200.sh
#
# Override the data root if yours differs:
#     DATASET_ROOT=/some/other/data bash setup_b200.sh
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/workspace/arun/data}"

echo "==> [1/5] Removing CUDA 12.1 wheels that are incompatible with Blackwell..."
# Pinned for CUDA 12.1 / cuDNN 8.9 -> no sm_100 kernels. The correct CUDA 12.8 /
# cuDNN 9.x libraries are pulled in as dependencies of the torch install below.
pip uninstall -y \
  torch torchvision triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 || true

echo "==> [2/5] Installing Blackwell-capable PyTorch (CUDA 12.8 build)..."
# torch 2.7.x is the first release with official Blackwell (sm_100) support.
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

echo "==> [3/5] Configuring DATASET_DIR=${DATASET_ROOT} ..."
export DATASET_DIR="${DATASET_ROOT}"
if ! grep -qs "export DATASET_DIR=${DATASET_ROOT}" "${HOME}/.bashrc" 2>/dev/null; then
  echo "export DATASET_DIR=${DATASET_ROOT}" >> "${HOME}/.bashrc"
  echo "    added DATASET_DIR to ~/.bashrc (persists across sessions)"
fi
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "    ERROR: DATASET_ROOT '${DATASET_ROOT}' does not exist."
  echo "    Set the correct path, e.g.: DATASET_ROOT=/workspace/arun/data bash setup_b200.sh"
  exit 1
fi
if [ ! -d "${DATASET_ROOT}/point_maze" ]; then
  echo "    WARNING: ${DATASET_ROOT}/point_maze not found -- check the dataset path before training."
fi

echo "==> [4/5] Pre-caching the DINOv2 backbone (so fresh runs work if internet later drops)..."
# All dino* encoder configs use dinov2_vits14. Downloading it now (internet available
# during setup) caches it under ~/.cache/torch/hub so training won't need to fetch it.
python - <<'PY' || echo "    WARNING: DINOv2 pre-cache failed (offline?). First training run will need internet once."
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("    cached dinov2_vits14 (num_features =", m.num_features, ")")
PY

echo "==> [5/5] Verifying the GPU is visible to PyTorch..."
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
  export DATASET_DIR=/workspace/arun/data   # already added to ~/.bashrc

  # quick smoke test (short run) to confirm data loads and the 45 GB slice holds:
  python train.py --config-name train.yaml env=point_maze \
      training.epochs=1 training.save_every_x_iterations=50 debug=True

  # full paper run:
  python train.py --config-name train.yaml env=point_maze

  # ---- Offline / interrupted-run resume ----
  # The FIRST run downloads the DINOv2 backbone (needs internet, once). After a
  # checkpoint exists, training resumes with NO internet: the encoder + DINOv2
  # weights are stored inside the checkpoint.
  #
  # If internet is unreliable, run wandb offline so logging never blocks training:
  #   export WANDB_MODE=offline
  #
  # Auto-resume: just rerun the SAME command -- it picks up model_latest.pth.
  # Resume from a SPECIFIC checkpoint you choose:
  #   python train.py --config-name train.yaml env=point_maze \
  #       training.resume_from=/workspace/arun/temporal-straightening/checkpoints/test/<run>/checkpoints/model_10.pth
EOF
