# TPR Recipe — Development and E2E Runbook

Use the original public SWE-ReBench data with a native Docker sandbox. **AMD64 is the default**. ARM64 uses the same Parquet with one additional task YAML and a locally supplied verified ARM64 image. All commands below, except managing the outer training container, run **inside the Ascend training container**.

## 1. Start or reproduce the training environment

Requires 8 Ascend NPUs, an NPU-compatible Python 3.11 training container with VERL/Megatron/MindSpeed, PyTorch/torch_npu, vLLM-Ascend, Ray, Docker CLI and access to the **host Docker daemon**. Mount the workspace and model directory. A clone plus editable installs alone do not provision the complete NPU stack.

On the existing host, identify the exact validated training image before migrating:

```bash
docker inspect uniagent_16npu --format '{{.Config.Image}}'
docker info --format 'Docker daemon={{.OSType}}/{{.Architecture}}'
```

Export/import that training image as needed using Docker save/load; workspace and model files in bind mounts must be copied separately. Avoid assuming that a similarly named image tag has the same dependencies.

## 2. Checkouts and dependencies

Use the fixed upstream VERL revision recorded in [REQUIRED_VERL.txt](REQUIRED_VERL.txt), and clone the public Uni-Agent repository. These are the paths used in the validated development environment; on another host choose equivalent mounted paths.

```bash
export RECIPE_ROOT=/workspace/verl-recipe-pr165-fix-verify
export VERL_ROOT=/workspace/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K

# On a new machine:
git clone https://github.com/verl-project/verl.git "$VERL_ROOT"
git -C "$VERL_ROOT" checkout 2ab35292039a51175a4b3d3b3b2d76e951a0d3e4
git clone https://github.com/verl-project/uni-agent.git "$UNIAGENT_ROOT"
git clone -b fix/tpr-pr165-precommit-20261029 https://github.com/1903943257/verl-recipe.git "$RECIPE_ROOT"

test -f "$UNIAGENT_ROOT/uni_agent/tasks/swe_rebench/task.py"
PYTHONPATH="$UNIAGENT_ROOT:$PYTHONPATH" python -c   'import uni_agent.tasks.swe_rebench.task; print("Uni-Agent import OK")'

git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
python -m pip install -e "$RECIPE_ROOT/tpr" --no-deps
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
```

The two editable installs assume the compatible training dependencies are already present. On a previously patched VERL checkout, do not re-apply the patch; inspect `git -C "$VERL_ROOT" status --short` first. Do not point `UNIAGENT_ROOT` to `/workspace/uni-agent-cc`: that is a different checkout and led to `No module named uni_agent.tasks`.

## 3. Create one **original**, unmodified task

Dataset: [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench), `filtered` split; instance `ASPP__pelita-875`. The Recipe preprocessing script uses the dataset's problem statement and metadata without a custom hint.

```bash
mkdir -p /workspace/tpr_data
python "$RECIPE_ROOT/tpr/prepare_swe_rebench_smoke.py"   --instance-id ASPP__pelita-875   --output /workspace/tpr_data/swe_rebench_pelita875.parquet

export TRAIN_FILE=/workspace/tpr_data/swe_rebench_pelita875.parquet
```

If Hugging Face is blocked, run this preprocessing on a network-enabled machine and copy the generated Parquet over. **Do not** use `pelita875_hint_h2.parquet` or `pelita875_hint_h2_arm64.parquet`. The Parquet contains a canonical public `swerebench/...x86_64...` image reference on **both** architectures.

## 4. Docker Sandbox: choose one native platform

**AMD64 Docker host (default)**

```bash
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"
docker pull --platform linux/amd64   swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
```

The original official image does not include Claude Code. Uni-Agent will install it if absent: the Sandbox needs working access to the installer. For offline sites, preinstall the matching AMD64 Claude Code in a derivative image (the public dataset remains unchanged; image_map can target that derivative image if configured).

**ARM64 Docker host (optional)**

Use the previously tested native ARM64 Pelita-875 image `uniagent-swe-pelita-875:latest` (image ID previously observed: `sha256:5eb899d263e3bc355836ad1497f470f03f383d5bd755087729dd0197125f5ac9`). It contains Claude Code 2.1.308. It is **a local image, not a public registry tag**.

```bash
# On the machine that has the validated ARM64 image:
docker save -o /workspace/uniagent-swe-pelita-875-arm64.tar   uniagent-swe-pelita-875:latest

# Copy the archive to the new host / mounted workspace, then:
docker load -i /workspace/uniagent-swe-pelita-875-arm64.tar
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker_arm64.yaml"

docker info --format 'Docker daemon={{.OSType}}/{{.Architecture}}'
docker image inspect uniagent-swe-pelita-875:latest   --format 'Image={{.Id}} {{.Os}}/{{.Architecture}}'
docker run --rm --entrypoint bash uniagent-swe-pelita-875:latest -lc   'uname -m; claude --version; git -C /testbed rev-parse HEAD'
```

Expected: `linux/arm64`, `aarch64`, Claude Code 2.1.308 and Pelita base commit `68af15d8d4199882d32bb4ede363195e2c5b5a99`.

The ARM64 YAML uses Uni-Agent's `sandbox.image_map` to resolve the canonical sample image into the local ARM64 image; **no dataset rewrite**. The Recipe's preflight image check verifies the resolved image matches the Docker daemon architecture. QEMU is not the E2E path.

Both task YAMLs have `agent.extra_env.DISABLE_AUTO_COMPACT: "1"`. This passes the variable to the Claude Code process, but its effect on session compaction is an **experiment to verify**, not an assumed guarantee.

## 5. Run N=8, concurrency=8, one PPO step

Use `TASK_CONFIG` from section 4 and `TRAIN_FILE` from section 3. Do not start a second Ray/VERL job on the same 8 NPUs.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT="/workspace/tpr_experiments/recipe_smoke/native_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

set -o pipefail
env NNODES=1 N_GPUS_PER_NODE=8   TRAIN_TP=2 TRAIN_CP=4 TRAIN_PP=1 TRAIN_SP=False   TPR_ENABLED=True ACTOR_RECOMPUTE=True   N=8 CONCURRENCY=8   TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1   MAX_PROMPT_LENGTH=18432 MAX_RESPONSE_LENGTH=2048   bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"   2>&1 | tee "$RUN_ROOT/train.log"
echo "TRAIN_STATUS=$?"
```

The launcher writes `$RUN_ROOT/launcher.log` and per-session logs under `$RUN_ROOT/agent/`.

## 6. Validate rollout, capacity, compaction, reward and TPR

```bash
grep -E 'training/global_step:1|actor/tpr/|timing_s/update_actor'   "$RUN_ROOT/launcher.log" | tail -10

grep -RhiE 'trajectory\(ies\)|materialization_reason=|max_trajectory_length|compact|compaction|rollback_count'   "$RUN_ROOT/agent" --include='*.log' | tail -150

grep -RhiE 'eval finished|reward for|task done|run_task done|resolved='   "$RUN_ROOT/agent" --include='*.log' | tail -60
```

Confirm **one training step completed**, TPR actor metrics were produced, **all eight rollout session outcomes** are accounted for, and task/reward results are logged. For each session, record trajectory count, `materialization_reason`, `finished`, and any compaction events. Multiple trajectories caused by `max_trajectory_length` do **not** alone prove Claude auto-compaction; inspect the session logs before concluding whether `DISABLE_AUTO_COMPACT` works.

## 7. Check the Recipe checkout

```bash
cd "$RECIPE_ROOT"
pre-commit run --all-files
python -m compileall -q tpr/verl_tpr tpr/check_task_images.py tpr/prepare_swe_rebench_smoke.py
bash -n tpr/run_uniagent_rebench_ppo.sh
git diff --check
```

The above describes how to rerun tests; a result is not marked passed until the actual run completes. Work on the separate fix branch until ARM64 revalidation and the later native AMD64-host run are both complete.
