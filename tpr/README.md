# Tree Prefix Reuse (TPR)

Tree Prefix Reuse (TPR) avoids repeated computation on shared prefixes across agent trajectories during Megatron Actor training. This recipe integrates TPR with [VERL](https://github.com/verl-project/verl) and [Uni-Agent](https://github.com/verl-project/uni-agent). See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

See [REQUIRED_VERL.txt](REQUIRED_VERL.txt). The integration patch targets VERL commit `2ab35292039a51175a4b3d3b2d76e951a0d3e4`; it must be applied to that checkout.

## Overview

Uni-Agent runs a software engineering task in a Docker sandbox, collects multiple agent trajectories, and returns training samples to VERL. TPR groups shared trajectory prefixes and executes the Actor forward/backward pass over the resulting trees. The example uses **Qwen3-4B**, **SWE-ReBench**, and the VERL PPO trainer with GRPO advantages.

The recipe provides a TPR runtime (`verl_tpr/`), a small VERL integration patch (`verl-tpr.patch`), a dataset conversion helper, and a training launch script.

## Requirements

- An existing Ascend training container with PyTorch/NPU, Megatron-Core, MindSpeed, Ray, and vLLM-Ascend. The verified software stack includes Megatron-Core 0.16.1, MindSpeed 0.16.0, and torch-npu 2.9.0.
- A local **Qwen3-4B** Hugging Face checkpoint and Megatron-Bridge source. On Ascend, MindSpeed patches must initialize before Megatron-Bridge is imported; the launcher checks this.
- Docker CLI in the training container, connected to a host Docker daemon. SWE-ReBench task containers use `linux/amd64` and must have Claude Code available (or be able to install it).
- Eight available NPUs for the example configuration (TP2 × CP4 × DP1).

The examples below assume that a **host workspace directory is mounted at `/workspace` inside the training container**. Run Git commands on the **host**, and all subsequent Python/Docker/training commands **inside the training container**.

## Installation

### 1. Check out the repositories and apply the VERL patch (host)

Run this from the **host directory mounted as `/workspace` in your training container**:

```bash
mkdir -p e2e_test

git clone https://github.com/verl-project/verl.git e2e_test/verl
git clone -b fix/tpr-pr165-precommit-20261029 \
  https://github.com/1903943257/verl-recipe.git e2e_test/verl-recipe
git clone https://github.com/verl-project/uni-agent.git e2e_test/uni-agent

VERL_ROOT="$PWD/e2e_test/verl"
RECIPE_ROOT="$PWD/e2e_test/verl-recipe"
VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")

git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" diff --check
```

The patch changes only two VERL files: `verl/workers/config/engine.py` and `verl/workers/engine/megatron/transformer_impl.py`. Apply it **once** to a clean checkout. Until [verl-recipe PR #165](https://github.com/verl-project/verl-recipe/pull/165) is merged, the commands use the Recipe development branch.

### 2. Prepare the training environment (container)

```bash
export WORK_ROOT=/workspace/e2e_test
export VERL_ROOT="$WORK_ROOT/verl"
export RECIPE_ROOT="$WORK_ROOT/verl-recipe"
export UNIAGENT_ROOT="$WORK_ROOT/uni-agent"
export MODEL_PATH=/workspace/hf_models/Qwen3-4B
export MEGATRON_BRIDGE_SRC=/workspace/Megatron-Bridge/src

# Install missing Python dependencies in your existing Ascend image.
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow huggingface_hub

# Download the model if it is not already available locally.
if [ ! -f "$MODEL_PATH/config.json" ]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir=os.environ["MODEL_PATH"])
PY
fi

test -f "$MEGATRON_BRIDGE_SRC/megatron/bridge/__init__.py"
export PYTHONPATH="$MEGATRON_BRIDGE_SRC:$RECIPE_ROOT/tpr:$VERL_ROOT:$UNIAGENT_ROOT:$PYTHONPATH"

python - <<'PY'
import mindspeed.megatron_adaptor
from verl.models.mcore.bridge import AutoBridge
from verl_tpr.adapter import install_tpr_module_spec, run_tpr_forward_backward_batch
print("MindSpeed / Megatron-Bridge / TPR imports: PASS")
PY
```

If Megatron-Bridge is mounted under `/Megatron-Bridge/src`, set `MEGATRON_BRIDGE_SRC` accordingly. Do not import `megatron.bridge` before the MindSpeed NPU compatibility patches have initialized.

## Data preparation

The data source is [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench), split `filtered`. The included [prepare_swe_rebench_smoke.py](prepare_swe_rebench_smoke.py) downloads a selected task through Hugging Face and converts it into the **Parquet format expected by Uni-Agent**, including the task prompt, repository metadata, tests and Docker image identifier.

For a reproducible single-task example:

```bash
export TRAIN_FILE="$WORK_ROOT/data/swe_rebench.parquet"
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" \
  --instance-id ASPP__pelita-875 --output "$TRAIN_FILE"
```

To use another SWE-ReBench task, change `--instance-id` and ensure its matching Docker sandbox image is available. The helper selects **one task**, not the full dataset. Preparing a larger training dataset requires converting additional task records and making their corresponding sandbox images available; a single-task example is intended to demonstrate the integration.

## Docker sandbox

The example task requires this AMD64 image:

```bash
docker pull --platform linux/amd64 \
  swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest

export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
```

The Docker CLI must talk to the **host daemon**. Uni-Agent launches the sandbox and installs Claude Code there if it is missing; this requires network access to the Claude Code installer. Installing `claude` only in the training container is insufficient.

### Offline sandbox setup (optional)

If the training environment has no registry/installer access, provide a Docker image archive and a previously downloaded Claude Code Linux x64 package. The original task image is loaded locally and a new sandbox image is built without network access.

```bash
# Copy both offline archives to /workspace/e2e_test/offline first.
IMAGE_ARCHIVE="$WORK_ROOT/offline/image.tar.gz"
CLAUDE_ARCHIVE="$WORK_ROOT/offline/anthropic-ai-claude-code-linux-x64-2.1.308.tgz"
test -f "$IMAGE_ARCHIVE" && test -f "$CLAUDE_ARCHIVE"
docker load -i "$IMAGE_ARCHIVE"

BUILD_DIR="$WORK_ROOT/claude_image"
mkdir -p "$BUILD_DIR"
cp "$CLAUDE_ARCHIVE" "$BUILD_DIR/claude.tgz"

cat > "$BUILD_DIR/Dockerfile" <<'EOF'
FROM swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
USER root
COPY claude.tgz /tmp/claude.tgz
RUN mkdir -p /tmp/claude-install \
    && tar -xzf /tmp/claude.tgz -C /tmp/claude-install \
    && install -m 755 /tmp/claude-install/package/claude /usr/local/bin/claude \
    && rm -rf /tmp/claude-install /tmp/claude.tgz
ENV DISABLE_AUTOUPDATER=1
EOF

docker build --network=none --pull=false \
  -t tpr-pelita875-claude:2.1.308 "$BUILD_DIR"
docker run --rm --pull never --platform linux/amd64 \
  --entrypoint bash tpr-pelita875-claude:2.1.308 \
  -lc 'claude --version'
```

Map the original image name from the dataset to the new local image. Keep the original image identifier in the Parquet:

```bash
export TASK_CONFIG="$WORK_ROOT/task_config_offline.yaml"

python - <<'PY'
import os
from pathlib import Path
import yaml

src = Path(os.environ["RECIPE_ROOT"]) / "tpr/task_config_swe_rebench_docker.yaml"
cfg = yaml.safe_load(src.read_text())
sandbox = cfg[0]["sandbox"]
sandbox["image_map"] = [{
    "from": "swerebench/sweb.eval.x86_64.aspp_1776_pelita-875",
    "to": "tpr-pelita875-claude:2.1.308",
}]
sandbox["sandbox_kwargs"]["pull_policy"] = "never"
Path(os.environ["TASK_CONFIG"]).write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
)
PY
```

For a fully offline run, the model checkpoint, converted Parquet, Python dependencies, and both archives must already be present locally. Image mapping is applied when Uni-Agent constructs `SandboxConfig`, not during the initial task-config merge.

## Training

With the variables above still exported:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export TPR_ENABLED=True
export RUN_ROOT="$WORK_ROOT/runs/tpr_$(date +%Y%m%d_%H%M%S)"

bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

By default the launcher uses eight NPUs, TP2 × CP4 × DP1, eight sampled trajectories and one Actor training step. Increase `TOTAL_TRAINING_STEPS` for a longer run. The output is under `$RUN_ROOT`; a new run needs a new directory.

To check that the Actor update finished:

```bash
grep 'training/global_step:1' "$RUN_ROOT/launcher.log" | tail -1
```

The TPR run reports metrics under `actor/tpr/`, including `actor/tpr/topology_token_reuse_ratio`. For the native Full path, set `TPR_ENABLED=False` and choose another `RUN_ROOT`. This launch example is an integration check, **not a Full-vs-TPR performance benchmark**.

## Configuration

The main entry point is [run_uniagent_rebench_ppo.sh](run_uniagent_rebench_ppo.sh).

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_TP`, `TRAIN_CP`, `TRAIN_SP` | `2`, `4`, `False` | Actor parallelism |
| `N`, `CONCURRENCY` | `8`, `8` | Trajectories and concurrent sessions |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | `18432`, `2048` | Sequence lengths |
| `TPR_ENABLED` | `True` | TPR or Full Actor path |
| `ACTOR_RECOMPUTE` | `True` | Enable full recomputation |
| `TOTAL_TRAINING_STEPS` | `1` | Training steps |
| `TASK_CONFIG` | Bundled YAML | Uni-Agent Docker sandbox configuration |

The packaged TPR runtime requires `TRAIN_SP=False` when TP and CP are both greater than 1.

## Troubleshooting

- **`No module named 'megatron.bridge'`:** Set `MEGATRON_BRIDGE_SRC` to the source root containing `megatron/bridge/`.
- **`No module named 'transformer_engine'` on a direct Bridge import:** Import `mindspeed.megatron_adaptor` first. The VERL launcher performs this ordered check.
- **Docker tries to pull while offline:** Load the sandbox image locally, configure `image_map`, and set `pull_policy: never`.
- **`claude` missing in sandbox:** Use the derived sandbox image; installing the CLI in the outer training container does not install it in the sandbox.
- **Patch fails:** Confirm the pinned VERL commit and a clean checkout. Apply the patch only once.
