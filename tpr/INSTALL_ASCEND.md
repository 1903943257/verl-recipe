# Fresh Ascend NPU environment: TPR Recipe + Uni-Agent + SWE-reBench

This is the **new-machine** guide for an 8-NPU (TP2 x CP4) PPO smoke test.
The documented public Docker image is a candidate, NOT a validated match for
the experiment's custom image. First try a previously validated NPU image.

The remote repository is https://github.com/verl-project/uni-agent; its **local
checkout directory is named un-iagent**. Python imports use uni_agent and the
launcher still expects the UNIAGENT_ROOT environment variable.

## 0. Installation inventory

Required: a host with eight Ascend NPUs and supported driver/firmware, Docker
Engine/daemon, access to the SWE-reBench Docker task images, enough disk for
Qwen3-4B, and permission to access public GitHub/Hugging Face or offline files.
Run Git commands on the HOST; run pip imports and PPO inside the training
container. All paths in the launcher must be container-visible.

For repeatability, record: host driver, CANN, torch/torch-npu,
Megatron-LM/MindSpeed, mbridge, vLLM/vLLM-Ascend, VERL SHA, Uni-Agent SHA,
Recipe SHA, model SHA, dataset SHA, and training/task image digests.

## 1. Host: verify NPUs and install Docker if necessary

On a supported Ubuntu host (adjust package manager for openEuler/EulerOS):

~~~bash
npu-smi info
# Only on Ubuntu without Docker Engine:
sudo apt-get update
sudo apt-get install -y docker.io git curl
sudo systemctl enable --now docker
sudo docker version
~~~

Ascend driver/firmware installation is hardware- and OS-dependent; provision
it via the cluster administrator/vendor before continuing. Do not try to
install kernel drivers inside the training container.

## 2. Host: obtain an Ascend training image

**Preferred:** use your already-tested image from the earlier TPR environment:

~~~bash
docker inspect EXISTING_CONTAINER --format '{{.Image}}'
# For a disconnected destination host:
docker save WORKING_IMAGE:TAG -o /path/to/working-ascend.tar
docker load -i /path/to/working-ascend.tar
export TPR_IMAGE=WORKING_IMAGE:TAG
~~~

**Public candidate** for A2/910B (not guaranteed compatible with the
Recipe's pinned VERL/MindSpeed requirements):

~~~bash
export TPR_IMAGE=quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-vllm
docker pull "$TPR_IMAGE"
~~~

For A3 choose the official A3 tag instead of the A2 tag. To build from
source, follow the official VERL Ascend guide; an A2 example:

~~~bash
cd /path/to/upstream/verl/docker/ascend
docker build -f Dockerfile.ascend_9.1.0_a2 -t verl-ascend:9.1.0-a2 .
export TPR_IMAGE=verl-ascend:9.1.0-a2
~~~

Official references:
- https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/dockerfile_build_guidance.html
- https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/install_guidance.html

## 3. Host: shared directory and training container

~~~bash
export HOST_WORK=/data/tpr-work
mkdir -p "$HOST_WORK"/src "$HOST_WORK"/hf_models "$HOST_WORK"/data "$HOST_WORK"/outputs
: "$TPR_IMAGE"

docker run -dit --name tpr_recipe_8npu \
  --ipc=host --network=host --shm-size=64g --privileged \
  -v "$HOST_WORK:/workspace" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  "$TPR_IMAGE" sleep infinity
docker exec -it tpr_recipe_8npu bash
~~~

SECURITY: privileged mode and the host Docker socket grant extensive host
access. Use this only on a trusted, dedicated host. For shared production
hosts replace them with approved device mappings and sandbox isolation.

The example assumes the host provides /usr/local/bin/npu-smi; adjust that
mount if your distribution installs the binary elsewhere. Additional
Ascend firmware/device mounts may be required by your driver version.

The host Docker daemon also runs **task sandboxes**. If the training
container lacks the docker CLI, install the CLI only, not another daemon.
The task image must be available to that host daemon.

## 4. Host: clone the three source checkouts

These commands run outside Docker in the HOST_WORK directory.

~~~bash
export HOST_WORK=/data/tpr-work
git clone https://github.com/1903943257/verl-recipe.git "$HOST_WORK/src/verl-recipe"
git -C "$HOST_WORK/src/verl-recipe" checkout fix/tpr-pr165-precommit-20261029

git clone https://github.com/verl-project/verl.git "$HOST_WORK/src/verl-tpr-recipe-smoke"
export RECIPE_ROOT="$HOST_WORK/src/verl-recipe"
export VERL_ROOT="$HOST_WORK/src/verl-tpr-recipe-smoke"
PINNED_SHA=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout --detach "$PINNED_SHA"
test "$(git -C "$VERL_ROOT" rev-parse HEAD)" = "$PINNED_SHA"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" diff --check
git -C "$VERL_ROOT" diff --name-only

git clone https://github.com/verl-project/uni-agent.git "$HOST_WORK/src/un-iagent"
git -C "$HOST_WORK/src/un-iagent" rev-parse HEAD
~~~

Patch expected: only verl/workers/config/engine.py and
verl/workers/engine/megatron/transformer_impl.py. Do not install another
version of VERL through Uni-Agent's optional git submodule; this Recipe
requires the separately patched, pinned upstream checkout above.

For long-term reproducibility, replace the sample Uni-Agent HEAD with the
SHA actually validated by the run.

## 5. Host: download Qwen3-4B

Use a separate download virtualenv so pip cannot upgrade torch-npu in the
training container:

~~~bash
export HOST_WORK=/data/tpr-work
python3 -m venv "$HOST_WORK/.hf-venv"
"$HOST_WORK/.hf-venv/bin/python" -m pip install -U huggingface_hub
"$HOST_WORK/.hf-venv/bin/hf" download Qwen/Qwen3-4B \
  --local-dir "$HOST_WORK/hf_models/Qwen3-4B"
~~~

The public source is https://huggingface.co/Qwen/Qwen3-4B.
Qwen3-4B-128K in our older experiment is a **local directory/artifact name**;
downloading official Qwen3-4B to that name is NOT proof of the same config
or long-context extension. The official public config may have a different
max_position_embeddings and RoPE configuration. For exact reproduction,
copy the complete original checkpoint/config/tokenizer into HOST_WORK/hf_models
and set MODEL_PATH to that path. For the smoke below use the public 4B
weights with a total configured context of 20,480 tokens.

## 6. Inside Docker: Python runtime and Uni-Agent installation

~~~bash
export RECIPE_ROOT=/workspace/src/verl-recipe
export VERL_ROOT=/workspace/src/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/src/un-iagent
export PYTHONPATH="$RECIPE_ROOT/tpr:$VERL_ROOT:$UNIAGENT_ROOT:$PYTHONPATH"

npu-smi info
python -V
python - <<'PY'
import torch
import torch_npu
print("Torch:", torch.__version__, "NPU count:", torch.npu.device_count())
assert torch.npu.device_count() >= 8
PY
docker version

python -m pip install --no-deps -e "$VERL_ROOT"
python -m pip install --no-deps -e "$UNIAGENT_ROOT"
python -m pip install --no-deps -e "$RECIPE_ROOT/tpr"
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow

python - <<'PY'
from dataclasses import fields
from pathlib import Path
import verl
import verl_tpr
import uni_agent
from verl.workers.config.engine import McoreEngineConfig
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from verl_tpr.adapter import run_tpr_forward_backward_batch
for package in (verl, uni_agent, verl_tpr):
    print(package.__name__, Path(package.__file__).resolve())
assert {"tpr_enabled", "tpr_cp_backend"} <= {x.name for x in fields(McoreEngineConfig)}
print("PASS: patched VERL and TPR imports")
PY
~~~

The container may already contain all dependencies. Do not blindly replace
Torch, torch-npu, CANN, MindSpeed or Megatron: check the first import error
and resolve with the matched Ascend stack. Uni-Agent's upstream guide is:
https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html

## 7. Inside Docker: create one public SWE-reBench sample

Upstream Uni-Agent includes the official full-dataset preprocessing command:

~~~bash
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir /workspace/data/swe_agent
~~~

The Recipe also includes a deterministic one-instance converter matching
the upstream row *schema*. It streams the filtered public split:

~~~bash
export RECIPE_ROOT=/workspace/src/verl-recipe
export TRAIN_FILE=/workspace/data/swe_agent/pelita875_public_smoke.parquet
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py" \
  --instance-id ASPP__pelita-875 --output "$TRAIN_FILE"
~~~

For offline data use --input-parquet /workspace/data/raw-rebench.parquet.
If the instance is absent the script will fail clearly.

**Exact old experiment:** pelita875_hint_h2.parquet was created with a
custom hint-injection step and is not a public downloadable file. Copy your
actual previous artifact into the mounted data directory if you need that
same input. The public smoke is NOT byte-identical or prompt-equivalent.

~~~bash
python - <<'PY'
import os
import pandas as pd
df = pd.read_parquet(os.environ["TRAIN_FILE"])
assert len(df) == 1 and {"prompt", "extra_info"} <= set(df.columns)
print("data schema OK, rows:", len(df))
PY
~~~

The SWE-reBench task Docker images are **separate** from the training image;
for pelita-875 the canonical name is
swerebench/sweb.eval.x86_64.aspp_1776_pelita-875. The host Docker daemon
must be able to pull the task image or already have it locally.
For offline use transfer its image tar with docker save/load.

## 8. Inside Docker: task sandbox and one real training step

This Recipe ships task_config_swe_rebench_docker.yaml using the local
Docker Sandbox. Unlike Uni-Agent's public Modal/OpenYuanrong examples,
it needs access to the host Docker daemon via the mounted socket.
Claude Code is installed *inside each task sandbox* when absent and
network access is available; on air-gapped hosts provide a task image
with Claude Code preinstalled. The training container itself does not
serve as that sandbox.

~~~bash
export RECIPE_ROOT=/workspace/src/verl-recipe
export VERL_ROOT=/workspace/src/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/src/un-iagent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B
export TRAIN_FILE=/workspace/data/swe_agent/pelita875_public_smoke.parquet
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
test -f "$VERL_ROOT/verl/trainer/main_ppo.py"
test -f "$UNIAGENT_ROOT/uni_agent/framework/entry.py"
test -f "$MODEL_PATH/config.json"
test -f "$TRAIN_FILE"
test -f "$TASK_CONFIG"
bash -n "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT="/workspace/outputs/recipe-smoke-$(date +%Y%m%d_%H%M%S)"
NNODES=1 N_GPUS_PER_NODE=8 TRAIN_TP=2 TRAIN_CP=4 \
TRAIN_PP=1 TRAIN_SP=False TPR_ENABLED=True ACTOR_RECOMPUTE=True \
N=8 CONCURRENCY=4 TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 \
MAX_PROMPT_LENGTH=18432 MAX_RESPONSE_LENGTH=2048 \
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
~~~

The launcher refuses to reuse RUN_ROOT/launcher.log. One completed PPO
Actor update and reward generation are required for E2E acceptance; a
successful import or startup does not mean training passed.

~~~bash
tail -120 "$RUN_ROOT/launcher.log"
grep -E 'ERROR|Traceback|Exception|TPR runtime import OK|Topology:' \
  "$RUN_ROOT/launcher.log" | tail -80
~~~

To smoke-test Full, use TPR_ENABLED=False and a fresh RUN_ROOT. Independent
online rollouts are stochastic and not a controlled speedup comparison.

## 9. After success: lock everything

~~~bash
mkdir -p "$RUN_ROOT/repro"
python -m pip freeze > "$RUN_ROOT/repro/pip-freeze.txt"
git -C "$VERL_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/verl-sha.txt"
git -C "$UNIAGENT_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/uni-agent-sha.txt"
git -C "$RECIPE_ROOT" rev-parse HEAD > "$RUN_ROOT/repro/recipe-sha.txt"
sha256sum "$TRAIN_FILE" "$MODEL_PATH/config.json" \
  > "$RUN_ROOT/repro/inputs.sha256"
npu-smi info > "$RUN_ROOT/repro/npu-smi.txt"
~~~

Also record host driver/CANN versions, the training image digest, the
SWE-reBench task image digest and the model checkpoint revision.
This document is a **candidate installation procedure**, not a declaration
that the public image and upstream Uni-Agent HEAD have passed TPR E2E.
