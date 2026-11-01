# TPR Recipe — Development and Validation

This page is for contributors. Start with [README.md](README.md) for installation and usage.

## Prerequisites

Use the VERL commit in [REQUIRED_VERL.txt](REQUIRED_VERL.txt), the recipe's `verl-tpr.patch`, and a compatible Ascend training environment. Uni-Agent, the model, Parquet dataset, and Docker sandbox image are external inputs. The launcher does not install dependencies or modify checkouts.

The patch changes only the VERL engine config and Megatron integration: it installs `TPRSelfAttention` during model construction and routes Actor forward/backward **before** native micro-batch splitting. The remaining implementation is in `verl_tpr/`.

## One-step E2E smoke

After completing the README setup, use a native Docker image matching the Docker daemon architecture. On AMD64, use `configs/task_config_swe_rebench_docker.yaml`; on ARM64, use `configs/task_config_swe_rebench_docker_arm64.yaml` and provide the local ARM64 image referenced there. Both configs pass `DISABLE_AUTO_COMPACT=1` to Claude Code; verify actual compaction behavior in the agent logs.

```bash
export TASK_CONFIG="$RECIPE_ROOT/tpr/configs/task_config_swe_rebench_docker.yaml"
export RUN_ROOT=/path/to/output/tpr-smoke

env NNODES=1 N_GPUS_PER_NODE=8 \
  TRAIN_TP=2 TRAIN_CP=4 TRAIN_PP=1 TRAIN_SP=False \
  TPR_ENABLED=True ACTOR_RECOMPUTE=True \
  N=8 CONCURRENCY=8 TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 \
  MAX_PROMPT_LENGTH=18432 MAX_RESPONSE_LENGTH=2048 \
  bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

Use the ARM64 config instead of the first export when running on ARM64. Do not run a second Ray/VERL job on the same NPUs. The launcher checks Docker image resolution and architecture before training.

## Validation checklist

- `git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"` passes on a **clean, pinned** checkout (before patching).
- `pre-commit run --all-files` passes from the Recipe repository root.
- `python -m compileall -q tpr/verl_tpr tpr/scripts` and `bash -n tpr/run_uniagent_rebench_ppo.sh` pass.
- One PPO step completes with Actor TPR metrics; all N=8 rollout sessions, rewards, and sandbox outcomes are accounted for.
- Inspect `$RUN_ROOT/launcher.log` and `$RUN_ROOT/agent/` for errors and compaction. `DISABLE_AUTO_COMPACT=1` alone is not proof that compaction was disabled.

Only mark a validation as passed after the corresponding run finishes. Full/TPR comparison should use the same data, model, topology, and training parameters.
