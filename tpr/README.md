# Tree Prefix Reuse (TPR)

TPR reuses shared trajectory prefixes in VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Quick Start

Use an existing Ascend NPU training environment (Megatron-Core 0.16.1, MindSpeed 0.16.0, `mbridge==0.15.1`), 8 available NPUs, an accessible AMD64 Docker daemon, Uni-Agent, and a local Qwen3 checkpoint. The sandbox must have Claude Code installed or be able to install it.

**On the host** (Git operations; use your actual host paths):

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe
VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
```

**In the Ascend training container** (mounted paths):

```bash
export VERL_ROOT=/workspace/verl
export RECIPE_ROOT=/workspace/verl-recipe
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export TRAIN_FILE="$HOME/data/tpr/swe_rebench_pelita875.parquet"

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" --output "$TRAIN_FILE"
docker pull --platform linux/amd64 swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

For an offline environment, use an existing Parquet and a locally loaded task image. The script imports `verl_tpr/` directly; no separate TPR installation is needed.

Defaults: Uni-Agent + SWE-ReBench, **TP2 × CP4 × DP1**, N=8, one PPO step. Set `TPR_ENABLED=False` for native Full training. Other options are in `run_uniagent_rebench_ppo.sh`.
