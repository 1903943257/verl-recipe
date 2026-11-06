# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required versions

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the pinned VERL and Uni-Agent revisions.

## Requirements

- An Ascend NPU environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge. The example uses eight NPUs.
- Uni-Agent installed at the revision specified in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Docker CLI in the PPO training container, with access to a trusted host Docker daemon.
- The SWE-ReBench task images use `linux/amd64`.
- The public-image workflow needs network access from the training host to pull missing task images and, when needed, install Claude Code in the sandbox.
- For an offline training host, prepare the task Parquet on a separate machine with Hugging Face access, then transfer the selected Parquet to the training host.

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

Uni-Agent's official preprocessing command downloads the public `nebius/SWE-rebench` `filtered` split from Hugging Face, converts it to Uni-Agent's native Task Config schema, and writes per-instance official task image references into the Parquet file.

**Do not run this download step on an offline training server.** Run it on a network-connected workstation with the pinned Uni-Agent revision and its dependencies installed.

On the network-connected workstation:

```bash
export UNIAGENT_ROOT=/path/to/uni-agent
export RECIPE_ROOT=/path/to/verl-recipe
export DATA_DIR="$HOME/data/tpr"

mkdir -p "$DATA_DIR"

# Use the Uni-Agent revision pinned by this Recipe.
git -C "$UNIAGENT_ROOT" fetch origin
git -C "$UNIAGENT_ROOT" checkout d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f

# Run the official dataset conversion.
cd "$UNIAGENT_ROOT"
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir "$DATA_DIR"
```

The official command creates:

```text
$DATA_DIR/swe_rebench_filtered.parquet
```

For this local two-task smoke, run the development-only selector on the same workstation. It filters the official Parquet and preserves the selected rows' original Task Config fields and image references; it does not reimplement dataset conversion:

```bash
python "$RECIPE_ROOT/tpr/select_swe_rebench_instances.py" \
  --input "$DATA_DIR/swe_rebench_filtered.parquet" \
  --instance-ids-file "$RECIPE_ROOT/tpr/tests/fixtures/swe_rebench_two_tasks.txt" \
  --output "$DATA_DIR/swe_rebench_two_tasks.parquet"
```

The fixture selects `ASPP__pelita-875` and `Algebra8__pyopenapi3-83`. Transfer only the selected two-row Parquet to the offline training server:

```bash
scp "$DATA_DIR/swe_rebench_two_tasks.parquet" \
  <user>@<training-host>:/path/to/tpr-data/
```

On the training server, set `DATA_DIR` to the destination directory and verify that the file arrived:

```bash
export DATA_DIR=/path/to/tpr-data
ls -lh "$DATA_DIR/swe_rebench_two_tasks.parquet"
```

No Hugging Face access is needed on the training server for this two-task run.

## Local two-task offline smoke

The following selector and Docker configuration are for local development. They are retained on this development branch and are not included in the public Recipe branch.

Build the local PyOpenAPI3-83 image:

```bash
bash "$RECIPE_ROOT/tpr/docker/pyopenapi3-83/build.sh"
```

The build produces `tpr-pyopenapi3-83-claude:2.1.308`. The Pelita image must also be available locally as `tpr-pelita875-claude:2.1.308`. Both images must be loaded into the Docker daemon used by the PPO container.

Launch the local offline smoke with the local YAML. For eight NPUs, TP=2 and CP=2 imply DP=2; SP is enabled by the Recipe launcher:

```bash
export TRAIN_FILE="$DATA_DIR/swe_rebench_two_tasks.parquet"
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker_local.yaml"
export TPR_ENABLED=True
export NNODES=1
export N_GPUS_PER_NODE=8
export TRAIN_TP=2
export TRAIN_CP=2
export TRAIN_PROMPT_BSZ=2
export GEN_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2

bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The local YAML maps the official per-row image references to the two preloaded local tags and uses `pull_policy: never`.

## Start the public-image example

For a network-connected training host, use the full official Parquet and the public Docker YAML:

```bash
export TRAIN_FILE="$HOME/data/tpr/swe_rebench_filtered.parquet"
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher defaults to the public task configuration and runs 10 training steps. Set `TPR_ENABLED=False` to use the original Full training path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.
