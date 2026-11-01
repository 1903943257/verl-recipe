# Tree Prefix Reuse (TPR)

TPR reuses shared trajectory prefixes during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Quick Start

Use an existing Ascend training environment with VERL/Megatron/MindSpeed, vLLM-Ascend, Ray, Uni-Agent and an accessible **AMD64 Docker daemon**. The example uses 8 NPUs.

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export VERL_ROOT=/path/to/verl
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B-128K

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow

mkdir -p "$HOME/data/tpr"
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" \
  --output "$HOME/data/tpr/swe_rebench_pelita875.parquet"
docker pull --platform linux/amd64 swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The script loads `verl_tpr/` directly from this Recipe, so no separate TPR installation is required. Use a clean checkout when applying the patch.

The default example runs **Uni-Agent + SWE-ReBench**, TP2 × CP4 × DP1, N=8, one PPO step. Set `TPR_ENABLED=False` for the native Full baseline; other settings are in `run_uniagent_rebench_ppo.sh`.
