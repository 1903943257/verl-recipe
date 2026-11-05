# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).

## Requirements

- An Ascend NPU environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge. The example uses eight NPUs.
- Uni-Agent: [Install Uni-Agent](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html). Use the revision specified in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Docker CLI for the included SWE-ReBench sandbox configuration, plus access to a trusted host Docker daemon.
- The default SWE-ReBench sandbox uses `linux/amd64`. When network access is unavailable, preload the task images and use [`task_config_two_tasks_offline.yaml`](task_config_two_tasks_offline.yaml) for the two-task local smoke.

## Quick Start

This example assumes you already have a VERL checkout. Install Uni-Agent using the instructions above, and prepare a local Qwen3-4B checkpoint.

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

By default, the preparation script follows Uni-Agent's upstream preprocessing and writes the complete `nebius/SWE-rebench` `filtered` split in the current Task Config format:

```bash
export TRAIN_FILE="$HOME/data/tpr/swe_rebench_filtered.parquet"
mkdir -p "$(dirname "$TRAIN_FILE")"
python "$RECIPE_ROOT/tpr/prepare_swe_rebench.py" --output "$TRAIN_FILE"
```

For a small prefix smoke, use `--max-instances N`. For an explicit local subset, provide one instance ID per line:

```bash
python "$RECIPE_ROOT/tpr/prepare_swe_rebench.py" \
  --instance-ids-file "$RECIPE_ROOT/tpr/tests/fixtures/swe_rebench_two_tasks.txt" \
  --output "$HOME/data/tpr/swe_rebench_two_tasks.parquet"
```

The two-task fixture selects `ASPP__pelita-875` and `Algebra8__pyopenapi3-83`. It preserves the upstream per-instance image references in each row; the offline YAML maps those references to preloaded local images.

For the offline two-task run, first load the built PyOpenAPI3 image into the host Docker daemon, confirm the existing Pelita image is present, then launch with:

```bash
TRAIN_FILE="$HOME/data/tpr/swe_rebench_two_tasks.parquet" \
TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_two_tasks_offline.yaml" \
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The offline YAML expects these local tags:

- `tpr-pelita875-claude:2.1.308`
- `tpr-pyopenapi3-83-claude:2.1.308`

## Start training

```bash
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The example runs 10 training steps by default. Set `TPR_ENABLED=False` to use the original Full training path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.
