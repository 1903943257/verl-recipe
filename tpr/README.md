# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).


## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).

## Overview

- **Model:** [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Dataset:** [SWE-ReBench](https://huggingface.co/datasets/nebius/SWE-rebench)
- **Training:** Uni-Agent + Claude Code with GRPO advantages
- **Parallelism:** Tensor Parallelism (TP), Context Parallelism (CP), and Data Parallelism (DP)

Ring CP enables Prefix QKV Merge and physical padding by default; no additional merge switches are required. The `verl_tpr/` package implements prefix reuse, while `verl-tpr.patch` connects it to the VERL Megatron Engine.

## Requirements

This example uses Uni-Agent to run SWE-ReBench tasks with Claude Code as the coding agent. The training environment requires Ascend, Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge; the example is configured for eight NPUs.

The included Uni-Agent task configuration uses Docker sandbox for SWE-ReBench and launches the image as `linux/amd64`. The training container needs Docker CLI configured to start the sandbox. Uni-Agent installs Claude Code inside the sandbox automatically when network access is available.

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
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher runs 10 training steps by default. Set `TOTAL_TRAINING_STEPS` to change the run length, or set `TPR_ENABLED=False` to use the original Full path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.

## Configuration

The training script exposes the following commonly adjusted settings. Data parallelism (DP) is calculated automatically from the total NPU count.

| Setting | Default | Description |
| --- | ---: | --- |
| `NNODES` / `N_GPUS_PER_NODE` | `1` / `8` | Number of nodes and NPUs per node |
| `TRAIN_TP` / `TRAIN_CP` | `2` / `4` | Actor tensor and context parallel sizes; Sequence Parallel is enabled by default |
| DP | Automatic | Derived from the total NPU count, TP, and CP |
| `N` / `CONCURRENCY` | `8` / `8` | Trajectories per prompt and concurrent agent sessions |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | `16384` / `16384` | Maximum prompt and response lengths |
| `TPR_ENABLED` | `True` | Enable TPR; set to `False` to use the original Full path |
| `TOTAL_TRAINING_STEPS` | `10` | Number of training steps |
