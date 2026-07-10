#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_pusht_3seeds.sh  <run_dir>
# Reproduces a Table-1 PushT cell the way the paper does it: mean +/- std over
# THREE data-sampling seeds (100/200/300). The `seed` arg controls which 50 test
# start/goal pairs are drawn (plan.py line 134: eval_seed = seed*n + 1).
#
# Open-loop uses objective.mode=last (terminal MSE within H).
# MPC uses objective.mode=staged (terminal within H, weighted beyond H).
# Both use objective.alpha=1 (PushT plans on target images AND proprio).
#
# Usage:
#   bash eval_pusht_3seeds.sh /workspace/arun/temporal-straightening/checkpoints/repro/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
# ---------------------------------------------------------------------------
set -uo pipefail

RUN_DIR="${1:?usage: eval_pusht_3seeds.sh <run_dir>}"
RUN_DIR="$(readlink -f "$RUN_DIR")"
NAME="$(basename "$RUN_DIR")"
SEEDS=(100 200 300)

export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export PLAN_SERIAL_ENV="${PLAN_SERIAL_ENV:-1}"

# Start clean so each logs.json holds exactly one entry per seed (no stale dupes).
echo ">>> Removing any previous PushT plan outputs for a clean 3-seed collection..."
rm -rf plan_outputs_gd/*pusht*/ plan_outputs_gd_mpc/*pusht*/ 2>/dev/null

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== OPEN-LOOP  seed=$s ==================="
  python plan.py --config-name plan_gd.yaml \
    ckpt_base_path="$RUN_DIR" model_name="$NAME" model_epoch=latest \
    decode_for_viz=false objective.alpha=1 seed=$s \
    || echo "!!! open-loop seed $s failed"
done

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== MPC  seed=$s ==================="
  python plan.py --config-name plan_gd_mpc.yaml \
    ckpt_base_path="$RUN_DIR" model_name="$NAME" model_epoch=latest \
    decode_for_viz=false objective.alpha=1 objective.mode=staged seed=$s \
    || echo "!!! MPC seed $s failed"
done

echo ""
echo "############### RESULTS (mean +/- std over 3 seeds) ###############"
python - <<'PY'
import glob, json, statistics as st
def collect(pattern):
    vals=[]
    for f in glob.glob(pattern, recursive=True):
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: vals.append(json.loads(line)["final_eval/success_rate"])
            except Exception: pass
    return vals
for label, pat in [("OPEN-LOOP","plan_outputs_gd/**/logs.json"),
                   ("MPC","plan_outputs_gd_mpc/**/logs.json")]:
    v=collect(pat)
    v=[x for x in v]  # keep all entries (one per seed)
    if v:
        m=st.mean(v)
        s=st.pstdev(v) if len(v)>1 else 0.0
        print(f"{label:10s} seeds={ [round(x,4) for x in v] }  ->  mean {m*100:.2f} +/- {s*100:.2f} %")
    else:
        print(f"{label:10s} no results found")
PY
