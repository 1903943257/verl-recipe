#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")" && pwd -P)"
RECIPE_ROOT="$(cd -- "\${SCRIPT_DIR}/.." && pwd -P)"
VERL_ROOT="\${VERL_ROOT:?Set VERL_ROOT to the patched VERL checkout used for training}"
CAPTURE="\${TPR_REAL_ACTOR_MINIBATCH:?Set TPR_REAL_ACTOR_MINIBATCH to an actor_update_mini_batch capture}"
MODEL_PATH="\${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B}"
ROWS="\${TPR_OFFLINE_ROWS:-0,1}"
TURN_MIN_TOKENS="\${TPR_TURN_MIN_SEGMENT_TOKENS:-128}"
RUN_MODES="\${TPR_OFFLINE_RUN_MODES:-both}"
OUTPUT_DIR="\${TPR_OFFLINE_OUTPUT_DIR:-/workspace/tpr_experiments/recipe_sp_ringcp4_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="\${TPR_OFFLINE_LOG_DIR:-\${OUTPUT_DIR}/logs}"

[[ -f "\${VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py" ]] || {
  echo "ERROR: invalid VERL_ROOT: \${VERL_ROOT}" >&2
  exit 2
}
[[ -f "\${VERL_ROOT}/verl/workers/utils/losses.py" ]] || {
  echo "ERROR: VERL loss module not found under \${VERL_ROOT}" >&2
  exit 2
}
[[ -f "\${CAPTURE}" ]] || { echo "ERROR: actor capture not found: \${CAPTURE}" >&2; exit 2; }
[[ -f "\${MODEL_PATH}/config.json" ]] || { echo "ERROR: Qwen3-4B config not found: \${MODEL_PATH}" >&2; exit 2; }
command -v torchrun >/dev/null || { echo "ERROR: torchrun is not available" >&2; exit 2; }

export ASCEND_RT_VISIBLE_DEVICES="\${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a VISIBLE_DEVICES <<< "\${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "\${#VISIBLE_DEVICES[@]}" -ne 8 ]]; then
  echo "ERROR: TP2 x CP4 replay requires exactly 8 visible NPUs; got \${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 2
fi

export RECIPE_ROOT VERL_ROOT
export PYTHONPATH="\${RECIPE_ROOT}/tpr:\${VERL_ROOT}\${PYTHONPATH:+:\${PYTHONPATH}}"
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS="\${TURN_MIN_TOKENS}"

case "\${RUN_MODES}" in
  both) modes=(unshared tpr) ;;
  unshared|tpr) modes=("\${RUN_MODES}") ;;
  *)
    echo "ERROR: TPR_OFFLINE_RUN_MODES must be both, unshared, or tpr" >&2
    exit 2
    ;;
esac

mkdir -p "\${OUTPUT_DIR}" "\${LOG_DIR}"
if compgen -G "\${OUTPUT_DIR}/*_rank*.pt" >/dev/null; then
  echo "ERROR: replay snapshots already exist in \${OUTPUT_DIR}; choose a new TPR_OFFLINE_OUTPUT_DIR" >&2
  exit 2
fi

python - <<'PY'
import os
from pathlib import Path

recipe = Path(os.environ["RECIPE_ROOT"]).resolve()
verl_root = Path(os.environ["VERL_ROOT"]).resolve()
import verl_tpr
import verl

runtime_path = Path(verl_tpr.__file__).resolve()
verl_path = Path(verl.__file__).resolve()
if recipe.joinpath("tpr") not in runtime_path.parents:
    raise SystemExit(f"Wrong TPR runtime imported: {runtime_path}")
if verl_root not in verl_path.parents:
    raise SystemExit(f"Wrong VERL checkout imported: {verl_path}")
from verl.workers.utils.losses import ppo_loss
print(f"Recipe Runtime: {runtime_path}")
print(f"VERL: {verl_path}")
print(f"Native PPO loss: {ppo_loss}")
PY

echo "Recipe offline Actor replay: TP2 x RingCP4 x DP1 x SP=True"
echo "VERL_ROOT=\${VERL_ROOT}"
echo "RECIPE_ROOT=\${RECIPE_ROOT}"
echo "CAPTURE=\${CAPTURE}"
echo "MODEL_PATH=\${MODEL_PATH}"
echo "ROWS=\${ROWS}"
echo "OUTPUT_DIR=\${OUTPUT_DIR}"

for mode in "\${modes[@]}"; do
  echo "TPR_RECIPE_REPLAY_START mode=\${mode}"
  torchrun \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port="\${MASTER_PORT:-29797}" \
    "\${SCRIPT_DIR}/tests/offline_actor_replay.py" \
    --mode "\${mode}" \
    --capture "\${CAPTURE}" \
    --model "\${MODEL_PATH}" \
    --output "\${OUTPUT_DIR}" \
    --rows "\${ROWS}" \
    --turn-min-segment-tokens "\${TURN_MIN_TOKENS}" \
    2>&1 | tee "\${LOG_DIR}/\${mode}.log"
done

if [[ "\${RUN_MODES}" == "both" ]]; then
  python "\${SCRIPT_DIR}/tests/offline_actor_replay.py" --compare "\${OUTPUT_DIR}" \
    2>&1 | tee "\${LOG_DIR}/comparison.log"
fi

echo "TPR_RECIPE_REPLAY_DONE output=\${OUTPUT_DIR}"
