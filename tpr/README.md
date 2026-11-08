# Tree Prefix Reuse (TPR)

Design discussion: [VERL RFC #7561](https://github.com/verl-project/verl/issues/7561).

TPR reuses common token prefixes across trajectories within a DP-local VERL V1 Megatron Actor mini-batch, reducing redundant attention computation without changing the existing rollout, reward, or dataset pipeline.

## Quick Start

Requires a working VERL V1 Megatron Actor environment. Use the compatible VERL revision specified in [REQUIRED_VERL.txt](REQUIRED_VERL.txt).

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
python -m pip install --no-deps --no-build-isolation -e "$RECIPE_ROOT/tpr"
```

Enable TPR in your existing Megatron Actor training launcher:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

Set the option to `False` to use the native Full training path.

## Example: Uni-Agent SWE

For a complete Uni-Agent setup, see the [Uni-Agent project](https://github.com/verl-project/uni-agent), its [installation guide](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html), and the [Agent RL training guide](https://uni-agent.readthedocs.io/en/latest/quickstart/rl-training.html). The [Claude Code SWE example](https://github.com/verl-project/uni-agent/tree/d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f/examples/claude_code_swe_task) covers the corresponding agent setup.

With the model, datasets, Sandbox and Ray environment prepared as in Uni-Agent, run:

```bash
bash tpr/run_train.sh
```

The launcher follows Uni-Agent's model, data and runtime directory conventions. This TPR example defaults to Qwen3-4B on 8 Ascend NPUs (TP2 × CP2 × DP2, SP enabled).

## Scope

**Validated:** Qwen3-4B, dense causal softmax attention, supported TP/CP/DP/SP combinations, PP=EP=VPP=1, vanilla token-mean PPO, zero attention dropout, and full recomputation (`uniform/full/1`).

**Not validated:** MoE, MTP, router replay, PP>1, sliding-window attention, FP8/FP4, dynamic CP, linear attention models, and hybrid linear-attention architectures.
