# Tree Prefix Reuse (TPR)

TPR enables training-side prefix reuse in VERL's Megatron engine.

## Required VERL version

Use the revision in [REQUIRED_VERL.txt](REQUIRED_VERL.txt) and apply [verl-tpr.patch](verl-tpr.patch) to the VERL checkout.

## Environment

- Python 3.11 and an Ascend NPU environment.
- VERL, Megatron-LM, MindSpeed, vLLM-Ascend and Ray.
- Uni-Agent and Docker CLI.
- The training container must access the host Docker daemon through `/var/run/docker.sock`.
- Eight available NPUs for the default configuration.

## Install

From the workspace containing the VERL, Recipe and Uni-Agent checkouts:

```bash
export RECIPE_ROOT=/workspace/verl-recipe
export VERL_ROOT=/workspace/verl
export UNIAGENT_ROOT=/workspace/uni-agent

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

python -m pip install -e "$RECIPE_ROOT/tpr"
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
```

## Data

Prepare the public SWE-ReBench Pelita-875 task:

```bash
mkdir -p ~/data/tpr
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" \
    --instance-id ASPP__pelita-875 \
    --output ~/data/tpr/swe_rebench_pelita875.parquet
```

The prepared task uses the official image `swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest` (`linux/amd64`, image ID `sha256:933103ac2f63e4d3ffcc8ffe769dfa38c7d4e85efc6ccfc3dcdba0b93617eaa6`).

## Training

Set the model path and launch:

```bash
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=~/data/tpr/swe_rebench_pelita875.parquet

bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher checks the task images before training: it pulls the image from the official registry, verifies its image ID and architecture, reports host/image architecture differences, and runs a Docker execution probe. On ARM64 hosts, amd64 emulation must be available.

The default configuration is one-node TP2 × CP4 × DP1, Ring CP, N=8, concurrency=8 and one PPO step. Override settings with environment variables when needed.
