# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

This recipe is pinned to upstream `verl` commit [`2ab35292039a51175a4b3d3b3b2d76e951a0d3e4`](https://github.com/verl-project/verl/commit/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4). The integration patch is specific to this source revision; see [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the canonical pin and installation command.

## Overview

- **Model:** [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Dataset:** [SWE-ReBench](https://huggingface.co/datasets/nebius/SWE-rebench)
- **Training:** Uni-Agent + Claude Code with GRPO advantages
- **Parallelism:** Tensor Parallelism (TP), Context Parallelism (CP), and Data Parallelism (DP)

The `verl_tpr/` package implements prefix reuse, while `verl-tpr.patch` connects it to the VERL Megatron Engine.

## Requirements

An Ascend environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge is required. The example uses eight NPUs.

The training container must have Docker CLI access to a host Docker daemon for `linux/amd64` SWE-ReBench sandboxes. Uni-Agent installs Claude Code inside the sandbox automatically when network access is available.

## Installation

On the **host**, run these commands in a directory mounted as `/workspace` inside the training container:

```bash
mkdir -p tpr-workspace
cd tpr-workspace

git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-recipe.git
git clone https://github.com/verl-project/uni-agent.git

VERL_COMMIT=$(sed -n 's/^COMMIT=//p' verl-recipe/tpr/REQUIRED_VERL.txt)
git -C verl checkout "$VERL_COMMIT"
git -C verl apply --check ../verl-recipe/tpr/verl-tpr.patch
git -C verl apply ../verl-recipe/tpr/verl-tpr.patch
```

Apply the patch once to a clean VERL checkout; do not apply it twice.

Inside the **training container**:

```bash
cd /workspace/tpr-workspace
export VERL_ROOT=/workspace/tpr-workspace/verl
export RECIPE_ROOT=/workspace/tpr-workspace/verl-recipe
export UNIAGENT_ROOT=/workspace/tpr-workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow huggingface_hub
hf download Qwen/Qwen3-4B --local-dir "$MODEL_PATH"
```

If the model is already downloaded, set `MODEL_PATH` to its directory. The launcher automatically detects Megatron-Bridge in the standard locations and initializes MindSpeed compatibility before importing it.

## Data preparation

The helper downloads a task from the `filtered` split of `nebius/SWE-rebench` and converts it into Uni-Agent's Parquet format.

```bash
export TRAIN_FILE=/workspace/tpr-workspace/data/swe_rebench.parquet
mkdir -p /workspace/tpr-workspace/data
python "$RECIPE_ROOT/tpr/prepare_swe_rebench.py" --output "$TRAIN_FILE"

docker pull --platform linux/amd64 \
  swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
```

Use `--instance-id` to select another task and make its corresponding Docker image available. The default task configuration is included in this recipe.

## Training

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher runs one training step by default. Set `TOTAL_TRAINING_STEPS` for longer training, or `TPR_ENABLED=False` for the original Full path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.

TPR has no separate recompute switch. To enable the supported native Megatron recompute mode, pass the ordinary transformer-config overrides to the launcher:

```bash
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh" \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
```

When `tpr_enabled=True` and `recompute_granularity` is configured, the Recipe installs the TPR-aware recompute adapter automatically. The currently implemented mode is `full/uniform/1`; unsupported recompute configurations fail validation rather than silently falling back.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_TP`, `TRAIN_PP`, `TRAIN_CP`, `TRAIN_SP` | `2`, `1`, `4`, `False` | Actor parallelism; DP is derived from the total NPU count |
| `N`, `CONCURRENCY` | `8`, `8` | Trajectories and concurrent sessions |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | `18432`, `2048` | Sequence lengths |
| `TPR_ENABLED` | `True` | Enable the TPR execution path |
| `TOTAL_TRAINING_STEPS` | `1` | Training steps |
