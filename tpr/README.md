# Tree Prefix Reuse (TPR)

TPR reuses identical prefixes across trajectories in a logical Actor mini-batch during VERL Megatron training. VERL still owns rollout, reward, PPO objectives, and optimization. This recipe adds an optional Actor forward/backward path, controlled by `tpr_enabled`.

See [RFC #7561](https://github.com/verl-project/verl/issues/7561) for motivation and design.

## Layout

- `verl_tpr/`: independent TPR runtime (tree planning, attention, execution, Ring CP, recompute).
- `verl-tpr.patch`: minimal integration into the pinned VERL Megatron engine.
- `REQUIRED_VERL.txt`: compatible VERL commit.
- `pyproject.toml`: installable Python package.
- `run_uniagent_rebench_ppo.sh`: Uni-Agent + SWE-ReBench example.
- `scripts/`, `configs/`: example data/image utilities and task configuration.

## Setup

Use an existing Python 3.11 Ascend NPU environment with compatible VERL, Megatron/MindSpeed, torch_npu, vLLM-Ascend, and Ray. The example also needs Uni-Agent, Docker CLI, access to a Docker daemon, a Qwen3 model, and 8 NPUs. An editable install does **not** provision the NPU stack.

```bash
export RECIPE_ROOT=/path/to/verl-recipe
export VERL_ROOT=/path/to/verl
export UNIAGENT_ROOT=/path/to/uni-agent

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
python -m pip install -e "$RECIPE_ROOT/tpr" --no-deps
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
```

Use a clean VERL checkout; do not reapply the patch if it is already present.

## SWE-ReBench example

Prepare one **original** public SWE-ReBench task (no task-specific prompt modification):

```bash
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=/path/to/swe_rebench_pelita875.parquet

python "$RECIPE_ROOT/tpr/scripts/prepare_swe_rebench_smoke.py" \
  --instance-id ASPP__pelita-875 --output "$TRAIN_FILE"
```

On an **AMD64 Docker host**, the default task config uses the original SWE-ReBench image:

```bash
docker pull --platform linux/amd64 swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

On an **ARM64 Docker host**, supply a compatible native sandbox image locally and use:

```bash
export TASK_CONFIG="$RECIPE_ROOT/tpr/configs/task_config_swe_rebench_docker_arm64.yaml"
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The ARM64 config maps the original dataset image to the local `uniagent-swe-pelita-875:latest`; it does not change the Parquet. The sandbox must have a usable Claude Code installation or installer.

Defaults: **8 NPUs, TP2 × CP4 × DP1, N=8, concurrency=8, one PPO step**. Set `TPR_ENABLED=False` to run the native Full baseline. `TRAIN_TP`, `TRAIN_CP`, `TRAIN_SP`, `ACTOR_RECOMPUTE`, `TOTAL_TRAINING_STEPS`, `MAX_PROMPT_LENGTH`, and `MAX_RESPONSE_LENGTH` can be overridden. CP>1 uses the runtime's Ring path automatically.

## Scope

The current recipe targets supported dense Megatron Actor training configurations. PP/VPP/EP and some SP/CP combinations are not supported; in particular, set `TRAIN_SP=False` when TP>1 and CP>1. TPR groups trajectories within each logical Actor mini-batch, not across mini-batches or DP replicas. The unpatched VERL path remains available when `tpr_enabled=False`.

For reproducibility checks and contributing, see [DEVELOPMENT.md](DEVELOPMENT.md).
