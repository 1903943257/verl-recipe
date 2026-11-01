# Tree Prefix Reuse (TPR)

TPR reuses shared trajectory prefixes during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Quick Start

Use an existing Ascend environment with Megatron-Core, MindSpeed, a source checkout of Megatron-Bridge, vLLM-Ascend, Ray, Uni-Agent and an accessible **AMD64 Docker daemon**. This example uses 8 NPUs.

**On the host**, with your actual host paths:

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export VERL_ROOT=/path/to/verl

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
```

**Inside the Ascend training container**, using the mounted paths:

```bash
export RECIPE_ROOT=/workspace/verl-recipe
export VERL_ROOT=/workspace/verl
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export MEGATRON_BRIDGE_SRC=/workspace/Megatron-Bridge/src
export TRAIN_FILE="$HOME/data/tpr/swe_rebench_pelita875.parquet"

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" --output "$TRAIN_FILE"
docker pull --platform linux/amd64 swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher detects local Bridge source at `/workspace/Megatron-Bridge/src` or `/Megatron-Bridge/src`; set `MEGATRON_BRIDGE_SRC` to override. VERL initializes MindSpeed NPU compatibility patches before importing Megatron-Bridge. Standalone Bridge imports without MindSpeed initialization may fail with a missing `transformer_engine`.

For offline runs, supply an existing SWE-ReBench Parquet file, a preloaded Docker sandbox image with Claude Code, and a local `TASK_CONFIG` with `image_map` and `pull_policy: never`.

The script loads `verl_tpr/` directly. Defaults: Uni-Agent + SWE-ReBench, **TP2 × CP4 × DP1**, N=8, one PPO step. Set `TPR_ENABLED=False` for a native Full baseline.
