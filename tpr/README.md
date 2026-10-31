# Tree Prefix Reuse (TPR) for VERL

TPR is a training-side prefix-reuse runtime for VERL's Megatron engine. The Recipe packages the runtime separately and uses a small, version-specific patch to connect it to VERL.

## Validation status

**A one-step NPU PPO E2E run has completed successfully** with the Recipe patch, runtime and launcher:

- Hardware: one ARM64 host with 8 Ascend NPUs.
- Topology: TP=2, CP=4 (Ring), PP=1, DP=1, SP=False.
- Model: the locally prepared `Qwen3-4B-128K` checkpoint.
- Task: `ASPP__pelita-875`, using the ARM64 task image.
- Rollout: N=8, concurrency=8, prompt limit=18,432, response limit=2,048.
- Training: `TPR_ENABLED=True`, one PPO training step completed.
- Observed `actor/tpr/topology_token_reuse_ratio`: approximately 3.28x (logical input tokens / unique tree tokens).

This is an end-to-end smoke test, **not** a controlled Full-vs-TPR performance claim.

Known runtime follow-ups are documented in [INSTALL_ASCEND.md](INSTALL_ASCEND.md): Claude Code may compact or continue into another Gateway Chain after the per-trajectory capacity is reached, and the Sandbox Reward install step may be slow when it needs network access. Neither issue is hidden by this validation status.

## Contents

- `verl_tpr/`: TPR runtime implementation.
- `verl-tpr.patch`: version-specific changes to VERL's engine configuration and Megatron execution path.
- `run_uniagent_rebench_ppo.sh`: Uni-Agent + SWE-ReBench PPO example launcher.
- `task_config_swe_rebench_docker.yaml`: local Docker Sandbox task configuration.
- `REQUIRED_VERL.txt`: pinned VERL revision for the patch.
- `INSTALL_ASCEND.md`: fresh-server installation, image preparation, and verified one-step run.
- `ARCHITECTURE.md`: Recipe layout, VERL integration points, execution flow and support boundaries.
- `scripts/rebuild_pelita_arm64.sh`: builds the ARM64 Pelita task image used by the validated run.

## Fresh-server installation

Follow [INSTALL_ASCEND.md](INSTALL_ASCEND.md). It distinguishes the Ascend **training container image** from the per-task **SWE-ReBench Sandbox image**, and records the actual paths and command used for the successful run.

## Required VERL revision

Use the exact commit recorded in `REQUIRED_VERL.txt`. The patch targets the upstream VERL source tree, not the private `verl-tpr` development fork.

## Quick start in the validated container

The following assumes the patched VERL checkout, Recipe, Uni-Agent, model, ARM64 task image and ARM64-adjusted Parquet are already present inside the same container:

~~~bash
export RECIPE_ROOT=/workspace/verl-recipe-pr165-fix-verify
export VERL_ROOT=/workspace/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/uni-agent
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export TRAIN_FILE=/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64.parquet
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT="/workspace/tpr_experiments/recipe_smoke/tpr_$(date +%Y%m%d_%H%M%S)"

env NNODES=1 N_GPUS_PER_NODE=8 \
  TRAIN_TP=2 TRAIN_CP=4 TRAIN_PP=1 TRAIN_SP=False \
  TPR_ENABLED=True ACTOR_RECOMPUTE=True \
  N=8 CONCURRENCY=8 \
  TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 \
  MAX_PROMPT_LENGTH=18432 MAX_RESPONSE_LENGTH=2048 \
  bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
~~~

`TPR_ENABLED` is the only TPR enable/disable switch. CP uses the existing VERL `context_parallel_size`; CP>1 automatically uses the Recipe's Ring path. There is no separate TPR CP-backend selector.

The launcher does not install the Ascend stack, patch VERL, create the task image, prepare the Parquet, stop Ray, or allocate devices. Use a dedicated 8-NPU allocation. It writes `launcher.log`, `agent/` and `ckpts/` under `RUN_ROOT`; use a new `RUN_ROOT` for each run.

## Prerequisites and scope

This Recipe is an integration layer, **not** a complete Ascend environment installer. It requires a compatible Python 3.11+ Ascend environment with PyTorch/torch-npu, VERL/Megatron/MindSpeed, vLLM Ascend, Ray, Docker CLI access to the host Docker daemon, and Uni-Agent runtime dependencies.

TPR is an optional Megatron Actor-training execution path. VERL continues to own PPO orchestration, rollout/reward integration, optimizer/scheduler and distributed process groups. The stable user-facing TPR switch is `tpr_enabled`; CP>1 uses Ring automatically.