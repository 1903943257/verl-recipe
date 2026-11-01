# Tree Prefix Reuse (TPR)

TPR reuses shared trajectory prefixes during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Quick Start

Run in an existing Ascend training environment (8 NPUs) with VERL/Megatron/MindSpeed, vLLM-Ascend, Ray, a Qwen3 checkpoint, and an **AMD64 Docker daemon** accessible to Uni-Agent. Clone the VERL, verl-recipe, and Uni-Agent repositories first.

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export VERL_ROOT=/path/to/verl
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=/path/to/swe_rebench_pelita875.parquet

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
python "$RECIPE_ROOT/tpr/scripts/prepare_swe_rebench_smoke.py" --output "$TRAIN_FILE"
docker pull --platform linux/amd64 swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest

bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher loads `verl_tpr/` directly from the Recipe checkout; no separate TPR installation is needed. Use a clean checkout when applying the patch.

The example runs **Uni-Agent + SWE-ReBench** with TP2 × CP4 × DP1, N=8, and one PPO step. Set `TPR_ENABLED=False` to use the native Full training path. Other training settings can be adjusted in `run_uniagent_rebench_ppo.sh`.
