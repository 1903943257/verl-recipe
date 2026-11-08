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

# This patch is validated against the pinned VERL commit in REQUIRED_VERL.txt.
# Check your existing checkout first; do not force-apply to a different revision.
git -C "$VERL_ROOT" rev-parse HEAD
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
translate it to this Hydra key (including our examples). **If you already
have a working Uni-Agent launcher, keep it:** patch VERL, install the TPR
package in the Actor worker environment, then add this Hydra flag. Do not
replace your training/rollout pipeline just to enable TPR. Your existing
Megatron model and loss configuration must still satisfy the supported scope
below.

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
follows the pinned upstream
[Uni-Agent Claude Code SWE trainer](https://github.com/verl-project/uni-agent/blob/d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f/examples/claude_code_swe_task/run_train.sh).
Upstream Uni-Agent is unchanged. The copied launcher preserves the official
`ray job submit --runtime-env`, V1 colocate-async Trainer, Gateway, Agent
Framework, Task Runner, PPO, and Reward wiring. TPR adds the Megatron Actor
`tpr_enabled` switch and the verified Ascend Actor settings (PP1, SP=True,
remove-padding, recompute). The pinned VERL-specific TQ capacity is explicit.

**Default example (proposed eight-NPU run):** one node, TP2 × CP2 × DP2,
PP1, SP=True, Qwen3-4B, two prompts × eight sessions each, ten training steps
and ten epochs (one step per epoch for this two-prompt dataset).
The launcher uses conservative Ascend vLLM settings: eager execution, no
forced `mamba_cache_mode=align` or Graph capture. Group filtering and warmup
generation are disabled, since async ReplayBuffer refills failed prompt groups
even with filtering off. The example defaults to no periodic checkpoint
saving or validation (SAVE_FREQ=-1, TEST_FREQ=-1). These defaults remain overridable
with the corresponding environment variables; they do **not** establish
eight-NPU E2E validation.

Supply an existing working Ray runtime environment, two-prompt training data,
a model directory and a Docker sandbox TaskConfig:

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export DATA_DIR=/path/to/shared/data
export RUNTIME_DIR=/path/to/shared/runtime
export RUNTIME_ENV=/path/to/your/working_runtime_env.yaml
export MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_FILE=/path/to/your/two_task.parquet
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG=/path/to/your/known_good_task_config.yaml

# Pin these explicitly if reusing a shell previously configured for four NPUs.
export NNODES=1 NGPUS_PER_NODE=8
export TRAIN_TP=2 TRAIN_CP=2 TRAIN_PP=1
export TRAIN_PROMPT_BSZ=2 PPO_MINI_BATCH_SIZE=2 N_RESP_PER_PROMPT=8
export CONCURRENCY=8 NUM_AGENT_WORKERS=1 GATEWAY_COUNT=1
export ROLLOUT_MAX_NUM_SEQS=20 ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
export TRAJECTORY_SELECTION=longest TOOL_PARSER=hermes
export TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1
export NUM_WARMUP_BATCHES=0 FILTER_GROUPS_ENABLE=False
export TOTAL_TRAINING_STEPS=10 TOTAL_EPOCHS=10
export TPR_ENABLED=True

# Keep each run's session logs separate, even if AGENT_LOG_DIR was exported earlier.
export EXP_NAME="tpr-8npu-$(date +%Y%m%d_%H%M%S)"
export AGENT_LOG_DIR="$RUNTIME_DIR/logs/tpr-uniagent-rebench/$EXP_NAME"

# Check that Ray actually advertises eight NPUs before submitting.
ray status

# Must launch from the Uni-Agent repository root.
cd "$UNIAGENT_ROOT"
bash "$RECIPE_ROOT/tpr/examples/uniagent/run_train.sh"
```

The script submits with `--no-wait`; a successful submission is not proof of
successful training. Follow the Job logs for `training/global_step=10` and
`actor/tpr/` metrics. If all sessions for a prompt repeatedly fail, the
asynchronous ReplayBuffer may refill indefinitely; stop the failed Job and
investigate the first model-server error rather than waiting for a step limit. For two prompts and DP2, each DP-local Actor mini-batch
may contain trajectories from only one prompt; TPR never shares prefixes
across DP ranks. The example tests TP/CP/DP integration, not cross-DP reuse.

The original [run_uniagent_rebench_ppo.sh](run_uniagent_rebench_ppo.sh)
remains an unchanged historical direct-launch baseline. The public remote
Docker/Claude Code installation flow is a separate validation target.

## 4. Validation status

- **E2E confirmed (2026-11-08, four Ascend NPUs):** official Uni-Agent
  `ray job submit` flow, Qwen3-4B, TP2 × CP2 × DP1 / SP=True;
  `training/global_step=1`, Actor update, and Ray Job `succeeded`.
  Actor metrics: `forest_trees=1`, `physical_segments=9`,
  `logical_input_tokens=142190`, `unique_tree_tokens=33928`,
  `topology_token_reuse_ratio=4.19093`. The ratio describes token reuse,
  **not** training speedup.
- **Previously validated:** TP2 × DP2 / CP1-SP E2E, and CP/SP physical
  padding correctness in dedicated tests.
- **Pending:** official-flow eight-NPU TP2 × CP2 × DP2 10-step E2E; public
  remote-sandbox Docker/Claude Code setup; matched Full-vs-TPR performance
  and correctness under the copied upstream launcher.

After validation, the only TPR-specific integration API for a supported VERL
Megatron Actor remains `tpr_enabled`.
