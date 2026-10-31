# Fresh Ascend NPU setup and one-step TPR Recipe run

This guide records the environment that completed a real one-step Uni-Agent + SWE-ReBench PPO run on 2026-10-30, then gives the steps needed to reproduce it on another ARM64 Ascend host.

It distinguishes two Docker images:

1. **Training container image**: contains the Ascend/CANN/PyTorch/torch-npu/VERL/Megatron/MindSpeed/vLLM-Ascend/Ray stack.
2. **Task Sandbox image**: one container per SWE-ReBench task. For the validated Pelita run this is the ARM64 image `uniagent-swe-pelita-875:latest`, which contains the testbed, Claude Code and Pelita test dependencies.

Do not substitute one image for the other.

## 0. What was actually validated

| Item | Validated value |
|---|---|
| Host architecture | `aarch64` |
| Training resources | 8 Ascend NPUs, one node |
| Training topology | TP=2, CP=4 (Ring), PP=1, DP=1, SP=False |
| VERL source | `/workspace/verl-tpr-recipe-smoke` |
| Recipe checkout | `/workspace/verl-recipe-pr165-fix-verify` |
| Uni-Agent checkout | `/workspace/uni-agent` |
| Model | `/workspace/hf_models/Qwen3-4B-128K` |
| Task Parquet | `/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet` |
| Task config | `/workspace/verl-recipe-pr165-fix-verify/tpr/task_config_swe_rebench_docker.yaml` |
| Task image | `uniagent-swe-pelita-875:latest` (`linux/arm64`) |
| Rollout | N=8, concurrency=8 |
| Sequence budget | prompt=18,432; response=2,048; total=20,480 |
| TPR | `TPR_ENABLED=True`; CP=4 automatically uses Ring |
| Training | one PPO step completed |

Observed metrics from that run included `training/global_step=1`, `actor/tpr/topology_token_reuse_ratio≈3.2828`, `timing_s/gen≈2252.49s`, `timing_s/update_actor≈162.89s`, and `timing_s/step≈2419.57s`.

This validates that the Recipe patch, runtime, launcher, Uni-Agent rollout, task reward path and one Actor update can complete together. It is a smoke test, not a Full-vs-TPR benchmark.

### Known follow-ups (not yet fixed by this guide)

- Some Claude Code sessions reached the 20,480-token trajectory capacity and then produced additional Gateway Chains. We have not yet validated `DISABLE_AUTO_COMPACT=1`; the current run did not use a working Session Hard Cap patch.
- The Sandbox Reward path may execute the instance's `metadata.install` command. The previous Pelita run spent several minutes in this install step. Runtime Sandbox proxy configuration and a verified install-skip policy are still pending.
- Do not set `SESSION_HARD_CAP=True` in the example below. The earlier patch was not part of the successful run and is not considered validated.

## 1. Host prerequisites: ARM64, Ascend and Docker

Run these commands on the **host**, not inside the training container:

~~~bash
uname -m
# Expected: aarch64

npu-smi info
docker version
docker info
~~~

The host must already have the matching Ascend driver/firmware installed. Do not install kernel drivers inside the training container.

On Ubuntu, if Docker Engine is not installed:

~~~bash
sudo apt-get update
sudo apt-get install -y docker.io git curl rsync
sudo systemctl enable --now docker
sudo docker version
~~~

For openEuler/EulerOS, use the site's approved Docker package procedure instead of copying Ubuntu package commands.

## 2. Preserve or transfer the validated Ascend training image

The exact training image is environment-specific. Do not assume that an arbitrary public VERL/Ascend image has the same CANN, torch-npu, MindSpeed, Megatron and vLLM-Ascend versions as the validated environment.

On the host where the working training container exists, identify its image and architecture:

~~~bash
TRAIN_CONTAINER=<name-or-id-of-the-working-training-container>
TRAIN_IMAGE_ID=$(docker inspect "$TRAIN_CONTAINER" --format '{{.Image}}')
docker inspect "$TRAIN_CONTAINER" --format 'configured_image={{.Config.Image}} image_id={{.Image}}'
docker image inspect "$TRAIN_IMAGE_ID" --format '{{.Os}}/{{.Architecture}} {{.Id}}'
~~~

Save that image for transfer to a fresh host:

~~~bash
docker save "$TRAIN_IMAGE_ID" -o /path/to/ascend-training-image.tar
~~~

On the destination ARM64 host:

~~~bash
docker load -i /path/to/ascend-training-image.tar
docker image ls
~~~

If rebuilding the training image from upstream sources instead, follow the official VERL Ascend installation/build guide and pin the exact software versions. The public candidate image mentioned in earlier drafts was not the image used for the successful run and is not certified by this Recipe validation.

## 3. Host: create the training container

The example below assumes the host workspace is `/ssd_3/jyb/project/uniagent_linear`, and that the transferred training image already contains the validated Ascend software stack.

~~~bash
export HOST_WORK=/ssd_3/jyb/project/uniagent_linear
export TPR_TRAIN_IMAGE=<loaded-ascend-training-image-tag>
mkdir -p "$HOST_WORK"

docker image inspect "$TPR_TRAIN_IMAGE" --format '{{.Os}}/{{.Architecture}}'

NPU_SMI=$(command -v npu-smi)
DEVICE_ARGS=()
for i in $(seq 0 7); do
  DEVICE_ARGS+=(--device="/dev/davinci\${i}")
done

docker run -d --name tpr-recipe-8npu \
  --network host \
  --ipc host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "\${DEVICE_ARGS[@]}" \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v "$NPU_SMI:/usr/local/bin/npu-smi:ro" \
  -v /var/log/npu:/usr/slog \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOST_WORK:/workspace" \
  -w /workspace \
  "$TPR_TRAIN_IMAGE" sleep infinity

docker exec -it tpr-recipe-8npu bash
~~~

The Docker socket mount is required because Uni-Agent creates task Sandbox containers through the host Docker daemon. The training container needs a Docker CLI, but it must not start a second Docker daemon.

Inside the training container, verify the environment:

~~~bash
npu-smi info
python -V
docker version

python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU count:", torch.npu.device_count())
assert torch.npu.device_count() >= 8
PY
~~~

If the Docker CLI is missing inside the training container, install only a compatible Docker CLI package. Do not launch `dockerd` inside it.

## 4. Host: clone Recipe, VERL and Uni-Agent

Run Git commands on the host in the mounted workspace. The paths then appear under `/workspace` inside the training container.

~~~bash
export HOST_WORK=/ssd_3/jyb/project/uniagent_linear

# Work on the separate verification branch. Do not update the PR head branch here.
git clone https://github.com/1903943257/verl-recipe.git \
  "$HOST_WORK/verl-recipe-pr165-fix-verify"
git -C "$HOST_WORK/verl-recipe-pr165-fix-verify" \
  checkout fix/tpr-pr165-precommit-20261029

# Clean upstream VERL checkout at the exact pinned revision.
git clone https://github.com/verl-project/verl.git \
  "$HOST_WORK/verl-tpr-recipe-smoke"
export RECIPE_ROOT="$HOST_WORK/verl-recipe-pr165-fix-verify"
export VERL_ROOT="$HOST_WORK/verl-tpr-recipe-smoke"
PINNED_SHA=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
test -n "$PINNED_SHA"
git -C "$VERL_ROOT" checkout --detach "$PINNED_SHA"
test "$(git -C "$VERL_ROOT" rev-parse HEAD)" = "$PINNED_SHA"

# Uni-Agent is a separate checkout; it is not included automatically by the Recipe.
# The local directory name is arbitrary. This guide uses un-iagent for a fresh clone.
git clone https://github.com/verl-project/uni-agent.git \
  "$HOST_WORK/un-iagent"
~~~

The successful run used `UNIAGENT_ROOT=/workspace/uni-agent`. A fresh clone at `/workspace/un-iagent` is equally valid as long as the environment variable points to that checkout. Python imports use the package name `uni_agent`.

## 5. Host: apply the VERL patch

The patch must be applied to the clean upstream checkout at the revision in `tpr/REQUIRED_VERL.txt`:

~~~bash
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" diff --check
git -C "$VERL_ROOT" diff --name-only
~~~

Expected changed VERL files:

~~~text
verl/workers/config/engine.py
verl/workers/engine/megatron/transformer_impl.py
~~~

Do not apply the patch twice. If the checkout is already patched or has unrelated changes, create a fresh checkout rather than resetting a working tree.

## 6. Host: obtain the model and custom task data

### 6.1 Qwen3-4B-128K

`Qwen3-4B-128K` is the locally prepared long-context model artifact used in the experiment. Do **not** download the public `Qwen/Qwen3-4B` model and rename the directory: that is not proof of the same 128K configuration or YaRN settings.

The exact public model repository/source URL for this locally prepared artifact is not recorded in this Recipe. Copy it from the approved internal artifact store or a machine that already has the validated checkpoint:

~~~bash
export MODEL_SOURCE=/path/to/verified/Qwen3-4B-128K
export MODEL_DEST="$HOST_WORK/hf_models/Qwen3-4B-128K"
mkdir -p "$MODEL_DEST"
rsync -a --info=progress2 "$MODEL_SOURCE/" "$MODEL_DEST/"
test -f "$MODEL_DEST/config.json"
~~~

For a remote source host, use the same command with a source such as `user@source-host:/absolute/path/Qwen3-4B-128K/`. Keep the trailing slash so the checkpoint contents are copied into `MODEL_DEST`.

Inside the training container, inspect the config:

~~~bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/workspace/hf_models/Qwen3-4B-128K")
cfg = json.loads((path / "config.json").read_text())
print("model_type:", cfg.get("model_type"))
print("max_position_embeddings:", cfg.get("max_position_embeddings"))
print("rope_scaling:", cfg.get("rope_scaling"))
assert cfg.get("max_position_embeddings") == 131072
PY
~~~

The validated artifact uses a 131,072-token position limit and YaRN scaling. Record its source revision and a checkpoint manifest when transferring it.

### 6.2 Custom Pelita Parquet

`pelita875_hint_h2.parquet` is a custom hint-injected experiment artifact, not a public SWE-ReBench download. Copy the existing artifact from the internal/shared experiment storage:

~~~bash
export DATA_SOURCE=/path/to/pelita875_hint_h2.parquet
mkdir -p "$HOST_WORK/tpr_experiments/shortfix"
cp "$DATA_SOURCE" "$HOST_WORK/tpr_experiments/shortfix/pelita875_hint_h2.parquet"
~~~

The original row refers to an x86_64 task image. On the ARM64 host, create a separate Parquet whose `extra_info.tools_kwargs.task.sandbox.image` points to the ARM64 image built in the next section. Do not overwrite the original artifact.

## 7. Host: build and verify the ARM64 Pelita task image

The task image is separate from the Ascend training image. The original SWE-ReBench image reference `swerebench/sweb.eval.x86_64.aspp_1776_pelita-875` resolved to an `amd64` image in the earlier check. It must not be used on this `aarch64` host.

The validated local image is:

- Tag: `uniagent-swe-pelita-875:latest`
- Architecture: `linux/arm64`
- Image ID at validation time: `sha256:5eb899d263e3bc355836ad1497f470f03f383d5bd755087729dd0197125f5ac9`
- Pelita base commit used by the task: `68af15d8d4199882d32bb4ede363195e2c5b5a99`

Build it on the ARM64 host. The helper reads the Docker daemon proxy, passes it to the build, and does not print proxy credentials:

~~~bash
bash "$HOST_WORK/verl-recipe-pr165-fix-verify/tpr/scripts/rebuild_pelita_arm64.sh"
~~~

The helper builds an ARM64 image with Node 22, Miniconda/Python 3.9.25, Claude Code 2.1.161, the Pelita repository and its test dependencies preinstalled. It verifies the image architecture and checks that the required base commit object exists.

Verify the local image before running PPO:

~~~bash
docker image inspect uniagent-swe-pelita-875:latest \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'

docker run --rm --pull never --entrypoint bash \
  uniagent-swe-pelita-875:latest -lc \
  'uname -m; /opt/conda/envs/testbed/bin/python --version; claude --version; git -C /testbed cat-file -t 68af15d8d4199882d32bb4ede363195e2c5b5a99'
~~~

Expected architecture is `linux/arm64`; `uname -m` inside the image should report `aarch64`.

### 7.1 Create the ARM64-specific Parquet

Run inside the training container after the original Parquet has been copied into `/workspace`:

~~~bash
python - <<'PY'
import copy
import pandas as pd

src = "/workspace/tpr_experiments/shortfix/pelita875_hint_h2.parquet"
dst = "/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet"
image = "uniagent-swe-pelita-875:latest"

df = pd.read_parquet(src)
assert len(df) == 1, f"expected the one-row Pelita hint artifact, got {len(df)}"

for index in df.index:
    extra = copy.deepcopy(df.at[index, "extra_info"])
    task = extra["tools_kwargs"]["task"]
    assert task["name"] == "swe_rebench"
    old_image = task["sandbox"]["image"]
    task["sandbox"]["image"] = image
    df.at[index, "extra_info"] = extra
    print(f"row={index}: {old_image} -> {image}")

df.to_parquet(dst, index=False)
print("saved:", dst)
PY
~~~

Verify the final row points to the local ARM64 image:

~~~bash
python - <<'PY'
import pandas as pd

path = "/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet"
df = pd.read_parquet(path)
assert len(df) == 1
task = df.iloc[0]["extra_info"]["tools_kwargs"]["task"]
print("task:", task["name"])
print("image:", task["sandbox"]["image"])
assert task["sandbox"]["image"] == "uniagent-swe-pelita-875:latest"
PY
~~~

Because the training container mounts `/var/run/docker.sock`, the host Docker daemon can see the local image tag when Uni-Agent creates the task Sandbox. The task config uses `pull_policy: missing`, so the local image is reused when present.

## 8. Inside the training container: install Python packages and verify imports

Enter the container and set paths. This block uses the directory names from the validated run; adjust `UNIAGENT_ROOT` if you cloned to `/workspace/un-iagent`.

~~~bash
export RECIPE_ROOT=/workspace/verl-recipe-pr165-fix-verify
export VERL_ROOT=/workspace/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export TRAIN_FILE=/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"

export PYTHONPATH="$RECIPE_ROOT/tpr:$VERL_ROOT:$UNIAGENT_ROOT:$PYTHONPATH"

python -m pip install --no-deps -e "$VERL_ROOT"
python -m pip install --no-deps -e "$UNIAGENT_ROOT"
python -m pip install --no-deps -e "$RECIPE_ROOT/tpr"

# Install only if these Runner-side packages are missing; do not upgrade torch/torch-npu.
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
~~~

The Recipe package intentionally does not install VERL, MindSpeed, Megatron-LM, PyTorch or Ascend dependencies. Those must come from the compatible training image. Avoid an unrestricted `pip install -U` in the active Ascend environment.

Verify paths and imports:

~~~bash
test -f "$VERL_ROOT/verl/trainer/main_ppo.py"
test -f "$UNIAGENT_ROOT/uni_agent/framework/entry.py"
test -f "$MODEL_PATH/config.json"
test -f "$TRAIN_FILE"
test -f "$TASK_CONFIG"
bash -n "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"

python - <<'PY'
from dataclasses import fields
from pathlib import Path
import verl
import verl_tpr
import uni_agent
from verl.workers.config.engine import McoreEngineConfig
from verl_tpr.adapter import run_tpr_forward_backward_batch
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter

for package in (verl, verl_tpr, uni_agent):
    print(package.__name__, Path(package.__file__).resolve())

assert "tpr_enabled" in {field.name for field in fields(McoreEngineConfig)}
print("PASS: VERL, TPR and Uni-Agent imports")
PY

npu-smi info
docker version
~~~

## 9. Run the validated one-step NPU PPO smoke

Run only when all eight NPUs and the local Ray resources are dedicated to this job. Do not launch this beside another job using the same Ray instance or devices.

~~~bash
export RECIPE_ROOT=/workspace/verl-recipe-pr165-fix-verify
export VERL_ROOT=/workspace/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export TRAIN_FILE=/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT="/workspace/tpr_experiments/recipe_smoke/tpr_arm64_$(date +%Y%m%d_%H%M%S)"

set -o pipefail
env \
  NNODES=1 \
  N_GPUS_PER_NODE=8 \
  TRAIN_TP=2 \
  TRAIN_CP=4 \
  TRAIN_PP=1 \
  TRAIN_SP=False \
  TPR_ENABLED=True \
  ACTOR_RECOMPUTE=True \
  N=8 \
  CONCURRENCY=8 \
  TOTAL_TRAINING_STEPS=1 \
  TOTAL_EPOCHS=1 \
  MAX_PROMPT_LENGTH=18432 \
  MAX_RESPONSE_LENGTH=2048 \
  bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh" \
  2>&1 | tee "$RUN_ROOT.console.log"
~~~

The launcher writes `launcher.log`, `agent/` and `ckpts/` beneath `RUN_ROOT`. The console log above is an additional copy of the terminal output. The launcher does not stop Ray, install the task image, patch VERL, or alter Uni-Agent.

### 9.1 Check the run

~~~bash
grep -E \
  'TPR runtime import OK|TPR_ENABLED=|Topology:|training/global_step|actor/tpr/|Traceback|ERROR|Exception' \
  "$RUN_ROOT/launcher.log" | tail -120
~~~

For this smoke test, acceptance means:

- the launcher imports `verl_tpr` from the selected Recipe checkout;
- the run uses `TPR_ENABLED=True` and TP2×CP4×DP1;
- rollout and Reward return data to VERL;
- `training/global_step=1` is reached;
- the Actor update completes without an exception.

A successful import or Ray startup alone is not E2E success.

## 10. Optional runtime experiments (not part of the validated baseline)

### 10.1 Disable Claude Code auto-compaction

The current task config is `tpr/task_config_swe_rebench_docker.yaml`. To test Claude Code's automatic compaction behavior without modifying Uni-Agent source, add the following under the existing `agent:` block:

~~~yaml
agent:
  extra_env:
    DISABLE_AUTO_COMPACT: "1"
~~~

Preserve existing `agent` keys. This must be passed to the Claude Code process inside the Sandbox; exporting it only in the outer training container is not sufficient unless the task runner explicitly forwards it.

This setting does **not** prevent the Uni-Agent Gateway from materializing a full Chain and accepting a later request that creates another Chain. It tests CC auto-compaction only. The result must be checked from per-session logs and the number of materialized trajectories.

### 10.2 Sandbox proxy and Reward install

The Docker daemon proxy used by the ARM64 image build is a **build-time** setting. It does not prove that a running task Sandbox has `HTTP_PROXY`/`HTTPS_PROXY`, nor that the Sandbox can reach the configured proxy.

The current `task_config_swe_rebench_docker.yaml` does not inject runtime proxy variables. Diagnose connectivity inside an actual task Sandbox before adding `sandbox_kwargs.run_args` with `--env HTTP_PROXY=...` and `--env HTTPS_PROXY=...`. The proxy hostname/address must be reachable from the Sandbox network; `127.0.0.1` inside the Sandbox is the Sandbox itself.

SWE-ReBench has two distinct dependency environments:

- Runner-side Reward packages (for example `swebench`) belong in the training image/Python environment.
- The per-instance `metadata.install` command runs inside the task Sandbox/testbed environment.

The ARM64 Pelita image preinstalls Pelita's project/test dependencies at image-build time. However, the current Reward implementation may still execute `metadata.install` at runtime. Do not claim that image preinstallation skips the runtime install until that behavior is verified. For other SWE-ReBench tasks, retain their instance-specific install step unless the corresponding image/dependency manifest has been validated.

## 11. Reproducibility manifest

After a successful run, save the versions and hashes alongside the output:

~~~bash
mkdir -p "$RUN_ROOT/repro"
python -m pip freeze > "$RUN_ROOT/repro/pip-freeze.txt"
git -C "$VERL_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/verl-sha.txt"
git -C "$UNIAGENT_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/uni-agent-sha.txt"
git -C "$RECIPE_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/recipe-sha.txt"
sha256sum "$TRAIN_FILE" "$MODEL_PATH/config.json" \
  > "$RUN_ROOT/repro/inputs.sha256"
npu-smi info > "$RUN_ROOT/repro/npu-smi.txt"
docker image inspect uniagent-swe-pelita-875:latest \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}' \
  > "$RUN_ROOT/repro/task-image.txt"
~~~

Also record the host driver/firmware, CANN version, training image ID, model artifact source/revision and task image digest. Do not put proxy credentials or other secrets into the manifest.

## 12. Troubleshooting

| Symptom | Likely check |
|---|---|
| Task container exits immediately with code 1 | Verify task image architecture; the original x86_64 Pelita image is `amd64`, while this host requires the ARM64 image |
| `No such image` in Sandbox | Confirm the task Parquet references `uniagent-swe-pelita-875:latest` and the host Docker daemon has that exact tag |
| `Container ... is not running` during git-history sanitization | First check whether the task container was created from the wrong architecture/image or exited; completed/aborted Sandboxes are destroyed by Uni-Agent |
| `tpr_enabled` rejected by Hydra | Confirm the pinned VERL patch was applied to the exact VERL checkout imported by Python |
| TransferQueue divides `None` by `num_data_storage_units` | Use the Recipe launcher, which supplies `transfer_queue.backend.SimpleStorage.total_storage_size=100000`; custom commands must provide a numeric capacity |
| `wrong verl_tpr import` | Check `PYTHONPATH` and the path printed by `verl_tpr.__file__` |
| Reward install takes several minutes | Inspect the Sandbox proxy environment, pip index configuration and the exact `metadata.install` command; do not infer network failure from the elapsed time alone |
| More than one trajectory appears in a Session | Inspect CC compaction events and Gateway `max_trajectory_length` materializations separately; `DISABLE_AUTO_COMPACT` does not itself disable Gateway Chain recreation |

