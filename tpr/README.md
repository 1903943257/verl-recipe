# Tree Prefix Reuse (TPR) — VERL Megatron Actor

TPR is an optional forward/backward execution path **inside VERL's Megatron
Actor**. It reuses shared token prefixes across RL trajectories in a DP-local
mini-batch. It is not a standalone replacement for Megatron and does not
require Uni-Agent: Uni-Agent supplies our validated Agentic RL trajectories.

See [RFC #7561](https://github.com/verl-project/verl/issues/7561) and
[REQUIRED_VERL.txt](REQUIRED_VERL.txt) for pinned VERL/Uni-Agent commits.

## 1. Install into an existing VERL environment

TPR's Python package is separate from the **VERL Engine integration**:
installing `verl-tpr` alone does not change VERL's execution behavior.

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

# Use the pinned, clean VERL checkout; apply this Engine patch only once.
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

# Install the runtime into the actual Actor worker Python environment.
python -m pip install --no-deps --no-build-isolation -e "$RECIPE_ROOT/tpr"
python -c 'from verl_tpr.adapter import run_tpr_forward_backward_batch; print("TPR runtime import OK")'
```

The editable install is suitable only when **all Ray workers can access the
same installed source path**. For isolated Ray runtime environments or
multi-node clusters, install the same package/wheel into each worker
environment; the driver-only installation does not guarantee worker imports.

For a production-ready wheel, build from this exact Recipe revision:
`python -m pip wheel --no-deps --no-build-isolation "$RECIPE_ROOT/tpr" -w /shared/wheels`.

## 2. Enable in your own VERL training script

Keep the existing Trainer, rollout engine, dataset, reward and Agent Framework.
In the Megatron Actor Hydra arguments add:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

Set to `False` to compare against native Full training. The short
`TPR_ENABLED=True` environment setting works only for scripts that explicitly
translate it to this Hydra key (including our examples).

**Supported/validated scope:** VERL V1 Megatron Actor at the pinned revision,
dense causal self-attention (Qwen3-4B family tested), PP=EP=VPP=1, compatible
TP/CP/DP/SP, vanilla token-mean PPO Actor loss, zero dropout, and optionally
TPR-aware full/uniform/1 recompute. MoE, MTP, router replay, PP>1,
sliding-window attention, FP8/FP4, Dynamic CP and activation offload are not
validated/supported. Config compatibility is not automatic.

**Actor mini-batch contract:** existing VERL tensors `input_ids` and
`response_mask`, and per-row `uid` or `tpr_trajectory_keys`. VERL Engine
supplies `batch_num_tokens` and `dp_size`; TPR forms the token-LCP Forest
locally, without a Uni-Agent-specific tree API. Cross-UID reuse occurs only
within the same DP-local Actor mini-batch.

## 3. Reproduce with Uni-Agent's *original* launch structure

The reference [examples/uniagent/run_train.sh](examples/uniagent/run_train.sh)
is copied from the pinned upstream
[Uni-Agent Claude Code SWE trainer](https://github.com/verl-project/uni-agent/blob/d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f/examples/claude_code_swe_task/run_train.sh).

The upstream source is **not modified**. The copied example keeps the official
`ray job submit --runtime-env`, V1 Trainer, Gateway, Agent Framework,
Task Runner, PPO and Reward wiring. Its differences are deliberately limited
to TPR enablement, supported PP=1, SP=True, remove-padding=True and an
environment-variable override for `FILTER_GROUPS_ENABLE`. Model/dataset/task
paths and the remaining training options are inherited from Uni-Agent and
should be supplied for your environment.

For a one-step 4-NPU smoke with a **known-good local dataset and task config**:

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export DATA_DIR=/path/to/shared/data
export RUNTIME_DIR=/path/to/shared/runtime
export RUNTIME_ENV=/path/to/your/working_runtime_env.yaml
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=/path/to/your/two_task.parquet
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG=/path/to/your/known_good_task_config.yaml

export NNODES=1 NGPUS_PER_NODE=4
export TRAIN_TP=2 TRAIN_CP=2 TRAIN_PP=1
export TRAIN_PROMPT_BSZ=2 PPO_MINI_BATCH_SIZE=2 N_RESP_PER_PROMPT=4
export TRAJECTORY_SELECTION=all FILTER_GROUPS_ENABLE=False
export GATEWAY_COUNT=1 NUM_AGENT_WORKERS=1 CONCURRENCY=8
export TOOL_PARSER=hermes
export MAX_PROMPT_LENGTH=16384 MAX_RESPONSE_LENGTH=16384
export TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1
export SAVE_FREQ=-1 TEST_FREQ=-1
export TPR_ENABLED=True

# Important: do not export the historically problematic Ray visibility override.
unset RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES

# Uni-Agent's original script requires execution from the Uni-Agent repo root.
cd "$UNIAGENT_ROOT"
bash "$RECIPE_ROOT/tpr/examples/uniagent/run_train.sh"
```

Because the official launcher uses `ray job submit --no-wait`, success of the
submission command does **not** mean training succeeded. Follow the Ray Job
logs until the actual Actor update finishes. Worker imports, Ray cluster
configuration and sandbox image availability must be valid; this official-flow
example is **not yet confirmed E2E**. Using the known-good local two-task
Docker workflow tests the official Ray submission path but **does not validate
the separate public networked Docker/Claude Code download path**.

The original [run_uniagent_rebench_ppo.sh](run_uniagent_rebench_ppo.sh)
remains unchanged as a historical direct-launch baseline. It contains a
Ray NPU visibility override that caused a device-ID error in one local
environment; do not use that script for this official-flow verification.

## 4. Validation status

- **Observed locally:** TP2×DP2 CP1/SP E2E, actual CP1 SP padding
  (logical 6481 → physical 6482), and TP2×CP2 DP1 E2E with eight
  trajectories/two UIDs in one Tree (nine segments, reuse factor ≈2.826).
- **Pending:** official `ray job submit` + TPR wheel/worker import path;
  public remote Docker task images and Claude Code installation; a matched
  Full-vs-TPR run using the copied upstream launcher.

After validating, the only TPR-specific integration API for a supported VERL
Megatron Actor remains the `tpr_enabled` flag.
