# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Overview

- **Model:** [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Dataset:** [SWE-ReBench](https://huggingface.co/datasets/nebius/SWE-rebench)
- **Training:** Uni-Agent + Claude Code with GRPO advantages
- **Parallelism:** Ascend NPU, TP2 × CP4 × DP1

The `verl_tpr/` package implements prefix reuse, while `verl-tpr.patch` connects it to the VERL Megatron Engine.

## Requirements

An Ascend environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge source is required. The configuration uses eight NPUs (Megatron-Core 0.16.1, MindSpeed 0.16.0, torch-npu 2.9.0).

The training container must access a Docker daemon for `linux/amd64` SWE-ReBench sandboxes. Uni-Agent installs Claude Code inside the sandbox automatically if network access is available; otherwise the sandbox image must already contain `claude`.

## Installation

**On the host:** place this `verl-recipe` checkout in a directory mounted as `/workspace` inside the training container. Run the following commands from its parent directory:

```bash
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/uni-agent.git

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' verl-recipe/tpr/REQUIRED_VERL.txt)
git -C verl checkout "$VERL_COMMIT"
git -C verl apply --check ../verl-recipe/tpr/verl-tpr.patch
git -C verl apply ../verl-recipe/tpr/verl-tpr.patch
```

Apply the patch once to a clean checkout. See [REQUIRED_VERL.txt](REQUIRED_VERL.txt) for the pinned VERL version.

**Inside the training container:**

```bash
cd /workspace
export VERL_ROOT=$PWD/verl
export RECIPE_ROOT=$PWD/verl-recipe
export UNIAGENT_ROOT=$PWD/uni-agent
export MEGATRON_BRIDGE_SRC=/workspace/Megatron-Bridge/src
export MODEL_PATH=/workspace/models/Qwen3-4B

python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow huggingface_hub
hf download Qwen/Qwen3-4B --local-dir "$MODEL_PATH"
```

Use existing local model weights if already downloaded. Set `MEGATRON_BRIDGE_SRC` to your Bridge source directory; the launcher initializes MindSpeed compatibility patches before importing it.

## Data preparation

The helper downloads a task from the `filtered` split of `nebius/SWE-rebench` and converts it into Uni-Agent's Parquet format.

```bash
export TRAIN_FILE=/workspace/data/swe_rebench.parquet
python "$RECIPE_ROOT/tpr/prepare_swe_rebench.py" --output "$TRAIN_FILE"

docker pull --platform linux/amd64 \
  swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
```

Use `--instance-id` to select a different task and provide its corresponding Docker image.

## Training

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher runs one training step by default. Set `TOTAL_TRAINING_STEPS` for longer training, or `TPR_ENABLED=False` for the original Full path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_TP`, `TRAIN_CP`, `TRAIN_SP` | `2`, `4`, `False` | Actor parallelism |
| `N`, `CONCURRENCY` | `8`, `8` | Trajectories and concurrent sessions |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | `18432`, `2048` | Sequence lengths |
| `ACTOR_RECOMPUTE` | `True` | Activation recomputation |
| `TOTAL_TRAINING_STEPS` | `1` | Training steps |
| `TASK_CONFIG` | Bundled YAML | Sandbox and agent settings |

Keep `TRAIN_SP=False` when TP and CP are both greater than 1. For restricted-network deployments, provide the model, dataset and a Claude Code-enabled Docker image locally, and configure the image mapping in `TASK_CONFIG`.
