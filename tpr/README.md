# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. This recipe integrates TPR with Uni-Agent on SWE-ReBench. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

See [REQUIRED_VERL.txt](REQUIRED_VERL.txt). The patch is pinned to **VERL `2ab35292039a51175a4b3d3b2d76e951a0d3e4`**, not a moving `main` branch.

## Overview

This example uses Uni-Agent and Claude Code to solve a SWE-ReBench task inside a Docker sandbox. It collects multiple trajectories, reuses shared prefixes in the Megatron Actor, and performs a training update through VERL's PPO trainer with GRPO advantages.

The default smoke test runs on **8 Ascend NPUs (TP2 × CP4 × DP1), N=8, one training step**, using Qwen3-4B-128K and `ASPP__pelita-875`.

## Requirements

- An existing Ascend NPU training container with VERL dependencies, Megatron-Core, MindSpeed, vLLM-Ascend, and Ray. Verified versions include Megatron-Core `0.16.1`, MindSpeed `0.16.0`, and `torch-npu 2.9.0`.
- A local Qwen3-4B-128K checkpoint, Uni-Agent, and **Megatron-Bridge source**. On Ascend, MindSpeed must install TE compatibility patches **before** importing Megatron-Bridge. The launcher checks this order automatically.
- Docker CLI in the training container, access to the host Docker daemon, and a **linux/amd64** SWE-ReBench sandbox with Claude Code. The sandbox is separate from the training container.
- Eight free NPUs (use devices 0–7 in this example). The host directory for the source checkouts must be mounted as `/workspace` inside the training container.

**Run Git commands on the host. Run Python, Docker CLI and NPU training commands inside the training container.** The paths below are examples; change them to match the mount on your machine.

## Installation

### 1. Clone and patch VERL (host)

```bash
HOST_ROOT=/path/to/shared/workspace
mkdir -p "$HOST_ROOT/e2e_test"

git clone https://github.com/verl-project/verl.git "$HOST_ROOT/e2e_test/verl"
git clone -b fix/tpr-pr165-precommit-20261029 \
  https://github.com/1903943257/verl-recipe.git \
  "$HOST_ROOT/e2e_test/verl-recipe"
git clone https://github.com/verl-project/uni-agent.git \
  "$HOST_ROOT/e2e_test/uni-agent"

VERL_ROOT="$HOST_ROOT/e2e_test/verl"
RECIPE_ROOT="$HOST_ROOT/e2e_test/verl-recipe"
VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")

git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" diff --check
```

Apply the patch **once to a clean checkout**. It changes only `verl/workers/config/engine.py` and `verl/workers/engine/megatron/transformer_impl.py`. The fork branch above is the available development location while [verl-recipe PR #165](https://github.com/verl-project/verl-recipe/pull/165) is unmerged.

### 2. Configure the training container

```bash
export VERL_ROOT=/workspace/e2e_test/verl
export RECIPE_ROOT=/workspace/e2e_test/verl-recipe
export UNIAGENT_ROOT=/workspace/e2e_test/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export MEGATRON_BRIDGE_SRC=/workspace/Megatron-Bridge/src

# Install these only if missing (or use preinstalled/offline wheels).
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow

export PYTHONPATH="$MEGATRON_BRIDGE_SRC:$RECIPE_ROOT/tpr:$VERL_ROOT:$UNIAGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import mindspeed.megatron_adaptor  # Install NPU/TE compatibility patches first.
from verl.models.mcore.bridge import AutoBridge
from verl_tpr.adapter import install_tpr_module_spec, run_tpr_forward_backward_batch
print("MindSpeed + Megatron-Bridge + TPR: PASS")
PY
```

`MEGATRON_BRIDGE_SRC` must contain `megatron/bridge/__init__.py`. The launcher also auto-detects `/workspace/Megatron-Bridge/src` and `/Megatron-Bridge/src`.

## Data and Docker sandbox

### 3. Prepare SWE-ReBench data

**Online (Hugging Face accessible):**

```bash
export TRAIN_FILE=/workspace/e2e_test/data/swe_rebench_pelita875.parquet
mkdir -p "$(dirname "$TRAIN_FILE")"
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" \
  --instance-id ASPP__pelita-875 --output "$TRAIN_FILE"
```

**Offline:** use an existing Parquet for `ASPP__pelita-875` (the script cannot fetch the dataset offline):

```bash
export TRAIN_FILE=/workspace/tpr_data/swe_rebench_pelita875.parquet
test -f "$TRAIN_FILE"
```

### 4. Prepare an AMD64 sandbox image

**Online:** pull the official task image. Claude Code must be available inside the sandbox or installable there with network access.

```bash
docker pull --platform linux/amd64 \
  swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
```

**Offline (verified):** load the original image, then build a derived image with a predownloaded Claude Code Linux x64 binary archive. Paths under `/path/to` are inputs you provide.

```bash
# Skip this if the original image is already loaded.
docker load -i /path/to/image.tar.gz

BUILD_DIR=/workspace/e2e_test/claude_image
mkdir -p "$BUILD_DIR"
cp /path/to/anthropic-ai-claude-code-linux-x64-2.1.308.tgz \
  "$BUILD_DIR/claude.tgz"

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
  -lc 'claude --version; git -C /testbed rev-parse HEAD'
```

The binary archive must contain `package/claude`. Do **not** change the original image field in the Parquet. Instead, map it to the locally built image using a copy of the task config:

```bash
export TASK_CONFIG=/workspace/e2e_test/task_config_pelita875_offline.yaml

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

# Mapping happens at SandboxConfig validation, not at TaskConfigResolver alone.
python - <<'PY'
import os
import pandas as pd
from uni_agent.tasks.config import TaskConfigResolver
from uni_agent.sandbox import SandboxConfig

info = pd.read_parquet(os.environ["TRAIN_FILE"]).iloc[0]["extra_info"]
task = info["tools_kwargs"]["task"]
resolved = TaskConfigResolver.from_file(os.environ["TASK_CONFIG"]).resolve(task)
image = SandboxConfig.model_validate(resolved["sandbox"]).image
print("Resolved sandbox image:", image)
assert image == "tpr-pelita875-claude:2.1.308"
PY
```

A working `claude --version` verifies the CLI installation; during rollout, Uni-Agent provides the model gateway for inference.

## Run training

### 5. One-step E2E smoke (training container)

```bash
# Keep VERL_ROOT, RECIPE_ROOT, UNIAGENT_ROOT, MODEL_PATH, TRAIN_FILE,
# and TASK_CONFIG exported from the previous steps.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TRAIN_TP=2 TRAIN_CP=4 TRAIN_SP=False
export TPR_ENABLED=True N=8 TOTAL_TRAINING_STEPS=1

export RUN_ROOT="/workspace/e2e_test/runs/tpr_$(date +%Y%m%d_%H%M%S)"
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

Logs are written to `$RUN_ROOT/launcher.log` and `$RUN_ROOT/agent/`. Use a fresh `RUN_ROOT` for each run.

### 6. Check the result

```bash
grep 'training/global_step:1' "$RUN_ROOT/launcher.log" | tail -1
grep 'Final validation metrics' "$RUN_ROOT/launcher.log" | tail -1
```

A successful run completes step 1 and reports `actor/tpr/` metrics. `Final validation metrics: None` is normal because the example disables periodic validation. The generated trajectories can yield different rewards, losses, and gradients each run.

**Observed single-node smoke result (November 2026, 8 NPUs, TP2 × CP4 × DP1):**

| Metric | Value |
| --- | ---: |
| `training/global_step` | 1 |
| `actor/tpr/topology_token_reuse_ratio` | 3.4145× |
| `actor/loss` | 0.2047 |
| `actor/grad_norm` | 1.2791 |
| `timing_s/update_actor` | 41.04 s |
| `timing_s/step` (including rollout) | 1080.70 s |
| `actor/perf/max_memory_allocated_gb` | 11.98 GiB |

This establishes an **end-to-end functional smoke**, not a Full-vs-TPR acceleration claim. To run the native Full baseline, set `TPR_ENABLED=False` and use another `RUN_ROOT`.

## Configuration

The key settings are in [run_uniagent_rebench_ppo.sh](run_uniagent_rebench_ppo.sh).

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAIN_TP`, `TRAIN_CP`, `TRAIN_SP` | `2`, `4`, `False` | Training parallelism |
| `N`, `CONCURRENCY` | `8`, `8` | Sampling and concurrent sessions |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | `18432`, `2048` | Sequence limits |
| `ACTOR_RECOMPUTE` | `True` | Uniform full recomputation |
| `TPR_ENABLED` | `True` | TPR vs native Full |
| `TOTAL_TRAINING_STEPS` | `1` | Smoke-test length |
| `TASK_CONFIG` | Bundled Docker YAML | Sandbox/agent settings |

Keep `TRAIN_SP=False` for this packaged TPR runtime when both TP and CP are greater than 1.

## Troubleshooting

- **`No module named 'megatron.bridge'`:** Check `MEGATRON_BRIDGE_SRC` and the mount. The launcher exports it through `PYTHONPATH` for Ray workers.
- **`No module named 'transformer_engine'` on direct import:** Initialize `mindspeed.megatron_adaptor` first; Ascend uses its compatibility modules.
- **Docker pull fails offline:** Load the image and use `pull_policy: never` with `image_map`.
- **Claude Code is missing:** It must exist inside the SWE-ReBench sandbox, not just the outer training container.
- **`git apply` fails:** Verify the pinned VERL commit and clean checkout; do not apply the patch twice.
