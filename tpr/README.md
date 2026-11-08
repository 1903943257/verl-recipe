# Tree Prefix Reuse (TPR) for VERL Megatron Actor

TPR reuses common token prefixes across trajectories **within a DP-local
Actor mini-batch**, using the existing VERL Megatron training path. It is
optional and **does not depend on Uni-Agent**. See [RFC #7561](https://github.com/verl-project/verl/issues/7561)
and [REQUIRED_VERL.txt](REQUIRED_VERL.txt) for the reference revisions.

## Quick start: enable TPR in an existing VERL training job

**Prerequisite:** you already have a working VERL V1 Megatron Actor training
environment. Keep your existing Trainer, rollout, dataset, reward and launcher.
Only the VERL Engine integration and the TPR Python package need installing.

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

# Apply once to the compatible VERL source checkout (see REQUIRED_VERL.txt).
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

# Install in the Python environment used by the Megatron Actor workers.
python -m pip install --no-deps --no-build-isolation -e "$RECIPE_ROOT/tpr"
python -c 'from verl_tpr.adapter import run_tpr_forward_backward_batch; print("TPR import OK")'
```

In **your existing training script**, add this one Hydra argument to the
Megatron Actor configuration:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

Use `False` for a native Full baseline. No Uni-Agent-specific changes, extra
TPR tree dataset, or separate trajectory-building service are required.
Installing `verl-tpr` **without** applying the Engine patch does not enable
TPR; the patch makes VERL recognize and route the new Actor setting.

**Supported scope:** dense causal self-attention with Qwen3-4B family
validated; PP=EP=VPP=1, compatible TP/CP/DP/SP, vanilla token-mean PPO,
zero attention dropout, and optional `full/uniform/1` recompute.
MoE, MTP, router replay, PP>1, sliding-window attention, FP8/FP4, Dynamic
CP and activation offload are outside the validated scope. Your existing
Megatron model/attention/loss settings must satisfy these constraints.

**Data contract:** the existing VERL Actor batch must carry `input_ids`,
`response_mask`, and a row-wise `uid` (or `tpr_trajectory_keys`). The
Engine supplies global `batch_num_tokens` and `dp_size`. TPR builds a
token-LCP Forest internally, allowing cross-UID prefix reuse **within** a
DP-local mini-batch; it never shares state across DP replicas.

**Ray environments:** an import in the driver is not sufficient if Actor
workers run in other containers, virtual environments or hosts. Install the
same TPR package and patched VERL code in the **worker execution environment**
too. A single shared environment needs only one installation.

## Example: Uni-Agent SWE-ReBench

[run_train.sh](run_train.sh) follows the pinned
[Uni-Agent Claude Code SWE launcher](https://github.com/verl-project/uni-agent/blob/d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f/examples/claude_code_swe_task/run_train.sh),
retaining Ray Jobs, Agent Framework, Gateway, reward and V1 Trainer. This
is **one example**, not a required TPR runtime component.

Example defaults: **8 Ascend NPUs** (TP2 × CP2 × DP2, PP1, SP=True),
Qwen3-4B, two prompts × **8 sessions per prompt**, concurrency 8, **10
steps / 10 epochs**, no Group Filtering or warmup. vLLM uses eager mode,
`max_num_seqs=20`, memory utilization 0.4 and does not force Mamba
cache alignment. For this two-prompt dataset and batch size 2 there is
one training step per epoch; adjust these values for your dataset.

If your Uni-Agent training is already working, you can keep its original
launcher and follow the **Quick start** above. To run this exact example,
supply your own working model, dataset, TaskConfig and Ray runtime YAML:

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export DATA_DIR=/path/to/data
export RUNTIME_DIR=/path/to/runtime
export RUNTIME_ENV=/path/to/working_ray_runtime_env.yaml
export MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_FILE=/path/to/two_prompt_train.parquet
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG=/path/to/working_swe_rebench_task.yaml
export EXP_NAME="tpr-$(date +%Y%m%d_%H%M%S)"
export AGENT_LOG_DIR="$RUNTIME_DIR/logs/tpr-uniagent-rebench/$EXP_NAME"

cd "$UNIAGENT_ROOT"
bash "$RECIPE_ROOT/tpr/run_train.sh"
```

The example submits a Ray Job with `--no-wait`; verify the actual Job
completion, `training/global_step` and `actor/tpr/` metrics. If inference
fails repeatedly, the asynchronous ReplayBuffer can continually refill
failed prompt groups: stop that Job and inspect the original vLLM error.

## Validation

- **E2E passed (2026-11-08):** official Uni-Agent Ray Job, four Ascend
  NPUs (TP2 × CP2 × DP1, SP=True), Qwen3-4B, one Actor update,
  `training/global_step=1`. `forest_trees=1`,
  `physical_segments=9`, `logical_input_tokens=142190`,
  `unique_tree_tokens=33928`, `topology_token_reuse_ratio=4.19093`.
  This is **token reuse**, not a measured speedup.
- Other dedicated tests have exercised TP/CP/DP/SP and padding paths.
  The new eight-NPU, ten-step example has **not yet been run E2E**.
  Public Docker sandbox setup and matched Full-vs-TPR E2E performance
  remain separate validation tasks.
