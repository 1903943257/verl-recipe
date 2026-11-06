# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required versions

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the pinned VERL and Uni-Agent revisions.

## Requirements

- An Ascend NPU environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge. The example uses eight NPUs.
- Uni-Agent installed at the revision specified in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Docker CLI in the PPO training container, with access to a trusted host Docker daemon.
- The SWE-ReBench task images use `linux/amd64`. The public workflow requires network access to pull missing task images and, when needed, install Claude Code in the sandbox.

## Quick Start

This example assumes you already have a VERL checkout and a local Qwen3-4B checkpoint.

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

Use Uni-Agent's official preprocessing command. It loads the public `nebius/SWE-rebench` `filtered` split and writes the Parquet file in Uni-Agent's native Task Config format, including each task's official public image reference:

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

The public task configuration is [`task_config_swe_rebench_docker.yaml`](task_config_swe_rebench_docker.yaml). Each Parquet row supplies its own official task image, so the public YAML does not enumerate task IDs or define an `image_map`.

## Local two-task offline smoke

The following selector and Docker configuration are for local development. They are retained on this development branch and are not included in the public Recipe branch.

First, select the two instances from the Parquet produced by Uni-Agent's official preprocessing. The selector only filters existing rows; it does not duplicate Uni-Agent's dataset conversion:

```bash
python "$RECIPE_ROOT/tpr/select_swe_rebench_instances.py" \
  --input "$DATA_DIR/swe_rebench_filtered.parquet" \
  --instance-ids-file "$RECIPE_ROOT/tpr/tests/fixtures/swe_rebench_two_tasks.txt" \
  --output "$DATA_DIR/swe_rebench_two_tasks.parquet"
```

The fixture selects `ASPP__pelita-875` and `Algebra8__pyopenapi3-83`.

Build the local PyOpenAPI3-83 image:

```bash
bash "$RECIPE_ROOT/tpr/docker/pyopenapi3-83/build.sh"
```

The build produces `tpr-pyopenapi3-83-claude:2.1.308`. The Pelita image must also be available locally as `tpr-pelita875-claude:2.1.308`. Both images must be loaded into the Docker daemon used by the PPO container.

Launch the local offline smoke with the local YAML:

```bash
TRAIN_FILE="$DATA_DIR/swe_rebench_two_tasks.parquet" \
TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker_local.yaml" \
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The local YAML maps the official per-row image references to the two preloaded local tags and uses `pull_policy: never`.

## Start the public-image example

```bash
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher uses [`task_config_swe_rebench_docker.yaml`](task_config_swe_rebench_docker.yaml) by default and runs 10 training steps. Set `TPR_ENABLED=False` to use the original Full training path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.
