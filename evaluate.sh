#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# evaluate.sh  <ckpt_base_path>  <model_name>  [model_epoch]
# Runs both planners (open-loop GD + MPC) on a trained checkpoint and prints
# the two success rates that fill a Table-1 row.
#
# Example:
#   bash evaluate.sh \
#     /workspace/arun/temporal-straightening/checkpoints/test \
#     umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05 \
#     latest
#
# For PushT add: EXTRA="objective.alpha=1"  (and for MPC also objective.mode=staged)
# ---------------------------------------------------------------------------
set -euo pipefail

CKPT_BASE="${1:?usage: evaluate.sh <ckpt_base_path> <model_name> [model_epoch]}"
MODEL_NAME="${2:?usage: evaluate.sh <ckpt_base_path> <model_name> [model_epoch]}"
MODEL_EPOCH="${3:-latest}"
EXTRA="${EXTRA:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== Open-loop (plan_gd) ==="
python plan.py --config-name plan_gd.yaml \
  ckpt_base_path="${CKPT_BASE}" model_name="${MODEL_NAME}" model_epoch="${MODEL_EPOCH}" ${EXTRA}

echo "=== MPC (plan_gd_mpc) ==="
python plan.py --config-name plan_gd_mpc.yaml \
  ckpt_base_path="${CKPT_BASE}" model_name="${MODEL_NAME}" model_epoch="${MODEL_EPOCH}" ${EXTRA}

echo ""
echo "=== success_rate values found (multiply by 100 for %) ==="
grep -rh "success_rate" plan_outputs_gd/ 2>/dev/null | tail -n 1 || echo "  (open-loop logs.json not found)"
grep -rh "success_rate" plan_outputs_gd_mpc/ 2>/dev/null | tail -n 1 || echo "  (mpc logs.json not found)"
echo "Tip: run 'python collect_results.py' to aggregate everything into a table."
