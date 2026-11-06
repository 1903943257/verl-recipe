# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required versions

- VERL and Uni-Agent revisions are pinned in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- The example uses an Ascend NPU environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge. It assumes eight NPUs.
- The SWE-ReBench example uses Uni-Agent's Docker sandbox. The PPO training container needs Docker CLI access to a trusted host Docker daemon.
- The public task images use `linux/amd64`. Network access is required to pull missing task images and, when needed, install Claude Code in the sandbox.

## Quick Start

This example assumes you already have a VERL checkout. Install the pinned Uni-Agent revision and prepare a local Qwen3-4B checkpoint.

Set the paths to your existing checkouts and model:

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B
```

Apply the TPR patch to a clean VERL checkout:

```bash
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
```

Apply the patch only once.

## Prepare SWE-ReBench data

Use Uni-Agent's official preprocessing command. It loads the public `nebius/SWE-rebench` `filtered` split, converts it to Uni-Agent's native Task Config schema, and writes the Parquet file with the per-instance public task image references:

```bash
export DATA_DIR="$HOME/data/tpr"
mkdir -p "$DATA_DIR"
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir "$DATA_DIR"
export TRAIN_FILE="$DATA_DIR/swe_rebench_filtered.parquet"
```

For a small prefix smoke, use the upstream `--max-instances` option and a separate output directory:

```bash
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir "$HOME/data/tpr/smoke" \
  --max-instances 2
```

No Recipe-owned dataset converter is required. The public task configuration is [`task_config_swe_rebench_docker.yaml`](task_config_swe_rebench_docker.yaml); each row supplies its own task image, so the YAML does not enumerate task IDs or define an `image_map`.

## Start training

```bash
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The example runs 10 training steps by default. Set `TPR_ENABLED=False` to use the original Full training path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.
