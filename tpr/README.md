# Tree Prefix Reuse (TPR)

TPR enables training-side prefix reuse in VERL's Megatron engine.

## Required VERL version

Use the revision in [REQUIRED_VERL.txt](REQUIRED_VERL.txt) and apply [verl-tpr.patch](verl-tpr.patch) to the VERL checkout.

## Environment

- Python 3.11 and a working Ascend NPU training environment.
- VERL, Megatron-LM, MindSpeed, vLLM-Ascend and Ray.
- Uni-Agent installed in a sibling checkout named `un-iagent` or `uni-agent`.
- Docker CLI connected to the host Docker daemon through `/var/run/docker.sock`.
- Eight available NPUs for the default configuration.

The task Parquet references the official image `swerebench/sweb.eval.x86_64.aspp_1776_pelita-875` (`linux/amd64`). On an ARM64 Docker host, enable amd64 QEMU/binfmt emulation or use an x86_64 Docker host. The task config requests `--platform=linux/amd64`. Verify the host can run it with `docker run --rm --platform linux/amd64 --entrypoint bash swerebench/sweb.eval.x86_64.aspp_1776_pelita-875 -lc 'uname -m'` (expected: `x86_64`).

## Install

From the workspace containing the VERL, Recipe and Uni-Agent checkouts:

```bash
export RECIPE_ROOT=/workspace/verl-recipe
export VERL_ROOT=/workspace/verl
export UNIAGENT_ROOT=/workspace/un-iagent

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

python -m pip install -e "$RECIPE_ROOT/tpr"
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
```

## Data

For the Pelita smoke run, use `pelita875_hint_h2.parquet` (not the `_arm64` variant). Keep its official `sandbox.image` reference unchanged; do not rewrite the Parquet or build a custom task image.

## Training

Set the model and dataset paths, then launch:

```bash
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=/path/to/pelita875_hint_h2.parquet

bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The launcher defaults to one-node TP2 × CP4 × DP1, Ring CP, N=8, concurrency=8 and one PPO step. Override settings with environment variables when needed.
