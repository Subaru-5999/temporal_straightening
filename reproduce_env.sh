#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# reproduce_env.sh
# One-command reproduction of the FULL B200 environment for this project, plus
# an optional fetch of the trained model from the Hugging Face Hub.
#
# What "the environment" is (all version-controlled in this GitHub repo):
#   1. Blackwell-capable PyTorch (torch 2.7.0 + cu128) + training deps   -> setup_b200.sh
#   2. Simulator / planning deps (MuJoCo 210, gym, mujoco-py, d4rl, PushT) -> setup_planning.sh
#
# The trained model is NOT in git (too large); it lives on the Hugging Face Hub
# and is fetched here with your API key (read token is enough to pull):
#   export HF_TOKEN=hf_xxx     # https://huggingface.co/settings/tokens
#
# Usage (from the repo root on a fresh B200 pod):
#   # env only:
#   bash reproduce_env.sh
#
#   # env + pull the trained model from Hugging Face (needs HF_TOKEN):
#   export HF_TOKEN=hf_xxx
#   bash reproduce_env.sh --with-model
#
#   # override paths / repo:
#   DATASET_ROOT=/workspace/temporal_s/data \
#   HF_REPO_ID=gravycrazy/temporal_straightening \
#   bash reproduce_env.sh --with-model
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- configurable knobs (env-overridable) ---------------------------------
DATASET_ROOT="${DATASET_ROOT:-/workspace/temporal_s/data}"
HF_REPO_ID="${HF_REPO_ID:-gravycrazy/temporal_straightening}"
# Where restored checkpoints land (matches ckpt_base_path=.../checkpoints/repro):
CKPT_RESTORE_DIR="${CKPT_RESTORE_DIR:-checkpoints/repro/test}"

WITH_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --with-model) WITH_MODEL=1 ;;
    *) echo "Unknown arg: $arg (supported: --with-model)"; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=============================================================="
echo " Reproducing environment for temporal-straightening on B200"
echo "   repo dir     : $HERE"
echo "   DATASET_ROOT : $DATASET_ROOT"
echo "   HF repo      : $HF_REPO_ID"
echo "   with-model   : $WITH_MODEL"
echo "=============================================================="

# ---- 1) Blackwell torch + training deps -----------------------------------
echo ""
echo ">>> [1/3] Training-tier env (torch 2.7 cu128 + deps) via setup_b200.sh ..."
DATASET_ROOT="$DATASET_ROOT" bash setup_b200.sh

# ---- 2) Simulator / planning deps -----------------------------------------
echo ""
echo ">>> [2/3] Planning-tier env (MuJoCo/gym/mujoco-py/d4rl/PushT) via setup_planning.sh ..."
bash setup_planning.sh

# ---- 3) Optional: fetch the trained model from the Hugging Face Hub --------
echo ""
if [ "$WITH_MODEL" = "1" ]; then
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo ">>> [3/3] --with-model requested but HF_TOKEN is not set."
    echo "    Set it and re-run just the pull:"
    echo "      export HF_TOKEN=hf_xxx"
    echo "      python hf_backup.py pull ${HF_REPO_ID} ${CKPT_RESTORE_DIR}"
  else
    echo ">>> [3/3] Fetching trained model from Hugging Face (${HF_REPO_ID}) ..."
    pip install -q -U huggingface_hub
    python hf_backup.py pull "${HF_REPO_ID}" "${CKPT_RESTORE_DIR}"
    echo "    Restored under: ${CKPT_RESTORE_DIR}/"
    ls -1 "${CKPT_RESTORE_DIR}" 2>/dev/null || true
  fi
else
  echo ">>> [3/3] Skipping model fetch (env only). To get the model later:"
  echo "      export HF_TOKEN=hf_xxx"
  echo "      python hf_backup.py pull ${HF_REPO_ID} ${CKPT_RESTORE_DIR}"
fi

cat <<EOF

==============================================================
 Environment reproduction complete.

 Verify torch sees the B200:
   python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
   # expect: 2.7.0+cu128 (10, 0)

 Then evaluate the PushT run (open-loop + MPC, 3 seeds):
   export DATASET_DIR=${DATASET_ROOT}
   bash eval_pusht_3seeds.sh ${CKPT_RESTORE_DIR}/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
==============================================================
EOF
