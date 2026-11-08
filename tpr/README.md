# Tree Prefix Reuse (TPR) for VERL Megatron Actor

TPR is a **pluggable training execution path within VERL's Megatron Actor**.
It builds a token-LCP Forest from a logical RL mini-batch and reuses shared
prefix computation across trajectories during Actor forward/backward.

It is **not** an Agent Framework, a rollout backend, or an independent
replacement for Megatron. Uni-Agent is our demonstrated Agentic RL producer,
but the TPR runtime does not depend on Uni-Agent.

See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Quick Start: use TPR in your existing VERL training

Prerequisites: the pinned VERL checkout in
[REQUIRED_VERL.txt](REQUIRED_VERL.txt), compatible Megatron/Ascend dependencies,
and a supported dense model with PP=EP=VPP=1.

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

# Current VERL integration is a small source patch; apply exactly once.
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

# Make the runtime importable where the Actor workers execute.
python -m pip install --no-deps -e "$RECIPE_ROOT/tpr"
```

Then add **one** argument to the existing VERL Megatron training command:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

For Ray jobs, ensure the runtime is importable in the workers' Python
environment (including any isolated `runtime_env`). Installation,
one-off import validation, data contract and compatibility details are in
[INTEGRATION.md](INTEGRATION.md).

## Use with Uni-Agent (optional Agentic RL example)

Reuse Uni-Agent's own training workflow: dataset preprocessing, Ray Job
submission, Gateway, Task Runner, Reward and Sandbox stay unchanged.
For an existing `run_train.sh`, add the single `tpr_enabled=True`
Hydra argument to its Megatron Actor configuration; **no Uni-Agent patch
or replacement launcher is required**. Read
[integration instructions](INTEGRATION.md#5-uni-agent-retain-the-official-workflow)
first: the upstream example defaults to unsupported PP2 and uses a model
not included in our validation set.

We also retain a **self-contained SWE-ReBench reference run**:

```bash
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_FILE=$HOME/data/tpr/swe_rebench_filtered.parquet

# Standard Uni-Agent dataset preprocessing (run in a networked environment):
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir "$HOME/data/tpr"

# Optional reference launcher; unlike the official Uni-Agent entry point,
# this script launches the VERL Python module directly:
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

This reference uses
[task_config_swe_rebench_docker.yaml](task_config_swe_rebench_docker.yaml),
which relies on network access to pull public `linux/amd64` task images
and install Claude Code if missing, plus Docker CLI/daemon access.
The original Full route can be selected with `TPR_ENABLED=False`.
Its launch defaults use eight NPUs and 10 steps and are adjustable via
the script's environment variables.

## Scope and evidence

| Dimension | Current support |
| --- | --- |
| Integration | VERL V1 Megatron Actor (pinned commit + Engine patch) |
| Model | Dense causal SelfAttention; Qwen3-4B family tested |
| Parallelism | TP/CP/DP; SP including CP1 physical padding; PP=EP=VPP=1 |
| Training | Vanilla token-mean PPO Actor; Tree built per DP-local mini-batch |
| Recompute | Off or TPR-aware full/uniform/1 |
| Not supported / validated | MoE, MTP, router replay, PP>1, EP>1, nonzero dropout, sliding window, FP8/FP4, dynamic CP, activation offload |

**Observed in local E2E:** TP2×DP2 CP1/SP run completed;
CP1 SP physical padding occurred (logical 6481 → physical 6482);
TP2×CP2 DP1 ran with 8 trajectories from 2 UIDs in one Tree,
9 physical segments and 2.826× topology token reuse.

**Not yet claimed as verified:** clean public installation via a Ray
`runtime_env` and public Docker/Claude Code image path. The complete
official-flow release gate is in [INTEGRATION.md](INTEGRATION.md#6-honest-public-validation-boundary).
