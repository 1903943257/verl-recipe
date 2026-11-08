# Integrate TPR into an existing VERL training project

TPR is a **Megatron Actor forward/backward execution path within VERL**.
It reuses common token prefixes across trajectories in one Actor mini-batch.
It is **not** a replacement for the VERL Trainer, a rollout engine, an
Agent Framework, or a standalone peer of the Megatron backend.

Uni-Agent is our demonstrated Agentic RL producer, **not a runtime dependency
of `verl_tpr`**. Existing projects can retain their dataset, rollout, reward,
task/sandbox providers, and training launcher if they meet the compatibility
contract below.

## 1. Install once

Use the pinned VERL revision in [REQUIRED_VERL.txt](REQUIRED_VERL.txt).
A clean source checkout is required because the current integration uses an
Engine patch (there is not yet a stable upstream Megatron Engine plugin API).

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

# Development machine / single-node process environment:
python -m pip install --no-deps -e "$RECIPE_ROOT/tpr"
python -c 'from verl_tpr.adapter import run_tpr_forward_backward_batch; print("TPR import OK")'
```

Apply the patch only once. For a frozen or multi-node deployment, use a wheel
built from the *same Recipe revision* instead of an editable path:

```bash
python -m pip wheel --no-deps "$RECIPE_ROOT/tpr" -w /shared/wheels
python -m pip install --no-deps /shared/wheels/verl_tpr-0.1.0.dev0-py3-none-any.whl
```

Install that wheel in **the Python environment executing each Ray worker**.
If the Ray job's `runtime_env` creates an isolated Python environment, install
the wheel there as well; installing it only in the submitting shell is not
sufficient. VERL, Megatron-Core, the chosen hardware stack, and their compatible
versions are provisioned separately.

## 2. Validate import *through Ray*, before training

After preparing your normal Ray runtime environment and starting the cluster:

```bash
export RUNTIME_ENV=/path/to/your/uniagent-or-verl-runtime-env.yaml

ray job submit --runtime-env "$RUNTIME_ENV" -- \
  python -m verl_tpr.check_ray_runtime
```

This probes the import and the Megatron adapter on every Ray node advertising
GPU/NPU resources (falling back to alive nodes when none advertise accelerators).
Look for `TPR_RAY_IMPORT_OK` records with the expected package path/version.
This is a packaging/dependency check, **not** a training-correctness test.

## 3. Enable TPR in an existing Megatron Actor configuration

In a launcher invoking `python -m verl.trainer.main_ppo`, add:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

Or in a complete resolved config:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      tpr_enabled: true
```

Turn off with `tpr_enabled=False`. Only the TPR-enabled Megatron Actor
forward/backward route is replaced; the original route remains the Full
baseline. `TPR_ENABLED` is an environment-variable convenience used by
[our standalone reference launcher](run_uniagent_rebench_ppo.sh); it is **not**
a universal VERL environment variable.

### Required configuration / validated scope

| Area | Contract |
| --- | --- |
| Engine | VERL V1 Megatron Actor; pinned VERL commit |
| Model | Compatible dense causal SelfAttention (Qwen3-4B family validated) |
| Parallelism | PP=EP=VPP=1; TP/CP/DP and SP supported in tested combinations |
| Attention | Full causal attention; dropout=0, no sliding window |
| Objective | Native vanilla PPO token-mean loss; other objectives require validation |
| Recompute | Off or supported full/uniform/1; advanced offload/FP8/FP4 unvalidated |
| Exclusions | MoE, MTP, router replay, FP8, CUDA graph, native CPU activation offload |

**The one-line switch does not waive these constraints.** For example, a
standard Uni-Agent run with PP=2 must set `TRAIN_PP=1` before enabling TPR.

## 4. Required Actor mini-batch data contract

TPR consumes the tensors already handed to the VERL Megatron Engine, **not**
Uni-Agent Session objects or a prebuilt TPR Tree. The relevant fields are:

| Field | Purpose |
| --- | --- |
| `input_ids` | Each trajectory's logical causal token sequence |
| `response_mask` | Supervised response positions |
| `uid` or `tpr_trajectory_keys` | Stable per-row identity; preserves trajectory grouping |
| `responses` (when present) | Must match the `input_ids` response suffix |
| `loss_mask` (when checked) | Must agree with supervised response positions |
| `batch_num_tokens`, `dp_size` | Attached by VERL Engine for global normalization |

Constructing a Forest or task-specific dataset converter in the agent/rollout
layer is **not required**. The Actor adapter builds a token-LCP Forest from
the actual mini-batch. Different UIDs may share a prefix **within one
DP-local mini-batch**; no reuse is attempted across DP replicas or batches.
If a custom pipeline drops both `uid` and `tpr_trajectory_keys`, TPR
fails clearly instead of silently assigning incompatible identities.

## 5. Uni-Agent: retain the official workflow

The pinned Uni-Agent source example is
[examples/claude_code_swe_task/run_train.sh](https://github.com/verl-project/uni-agent/blob/d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f/examples/claude_code_swe_task/run_train.sh).
It runs the official **`ray job submit --runtime-env`** path, with the
same Uni-Agent Gateway, Task Runner, Rollout, Reward and Sandbox components.

The upstream shell script **does not forward arbitrary extra CLI arguments**
to the remote Hydra process. To demonstrate the smallest possible change, we
ship a one-line, reversible patch to that pinned script:

```bash
export UNIAGENT_ROOT=/path/to/uni-agent
git -C "$UNIAGENT_ROOT" apply --check \
  "$RECIPE_ROOT/tpr/examples/uniagent/enable_tpr.patch"
git -C "$UNIAGENT_ROOT" apply \
  "$RECIPE_ROOT/tpr/examples/uniagent/enable_tpr.patch"
```

It adds **only** the `+actor_rollout_ref.actor.megatron.tpr_enabled=True`
Hydra argument. It does not change the agent, dataset, task, gateway, reward,
scheduler, or sandbox logic. To revert, use `git -C "$UNIAGENT_ROOT"
apply -R "$RECIPE_ROOT/tpr/examples/uniagent/enable_tpr.patch"`.

A proposed **one-step public integration smoke** using the *same* Uni-Agent
launcher (not a separate TPR training process) is:

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export DATA_DIR=/path/to/shared/data
export RUNTIME_DIR=/path/to/shared/runtime
export RUNTIME_ENV=/path/to/your/working_ray_runtime_env.yaml
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE="$DATA_DIR/swe_rebench_filtered.parquet"
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"

export NNODES=1 NGPUS_PER_NODE=8
export TRAIN_TP=2 TRAIN_CP=4 TRAIN_PP=1
export TRAIN_PROMPT_BSZ=2 PPO_MINI_BATCH_SIZE=2 N_RESP_PER_PROMPT=4
export TRAJECTORY_SELECTION=all
export MAX_PROMPT_LENGTH=16384 MAX_RESPONSE_LENGTH=16384
export TOOL_PARSER=hermes
export TOTAL_TRAINING_STEPS=1

# Upstream script submits through Ray Jobs and returns without waiting.
(cd "$UNIAGENT_ROOT" && bash examples/claude_code_swe_task/run_train.sh)
```

Before submitting, ensure the original Uni-Agent runtime env contains
`verl_tpr` on the workers and the chosen Docker sandbox has access to the
required public image/Claude Code downloads. Use the Ray import check above.
The command is **proposed for validation, not a claimed passing E2E run**.

The official example's defaults are *not* all TPR-compatible:
it defaults to PP2 and Qwen3.5-9B, which is not the model covered by our
Qwen3-4B validation. Provide a compatible dense model and set `TRAIN_PP=1`;
adjust TP×CP×DP, batch sizes, and the platform-specific `TASK_CONFIG`
for the available hardware. Use an official Uni-Agent dataset and a task
sandbox provider that works in your environment. No second TPR data pipeline
is required.

The self-contained [TPR SWE-ReBench example](run_uniagent_rebench_ppo.sh)
is retained for reproducibility, but its custom Docker task config and launch
defaults are an **optional reference setup**, not a required installation step.

## 6. Honest public validation boundary

- **Validated:** local offline Uni-Agent SWE-ReBench → VERL TPR Megatron
  E2E; TP2×DP2 (CP1/SP), TP2×CP2 (DP1) and cross-UID Tree diagnostics;
  CP1 SP physical padding observed with an odd-length segment.
- **Not yet validated end-to-end:** clean networked public Docker images and
  Claude Code installation; official `ray job submit` plus packaged TPR wheel.
- **Release gate:** run the Ray import probe, an official-flow Full step
  (`tpr_enabled=False`), then a matched TPR step
  (`tpr_enabled=True`) with the same data/model/rollout/batches. Capture the
  exact VERL and Uni-Agent commits and the Wheel version.

Do not describe the public flow as verified until that gate passes.
