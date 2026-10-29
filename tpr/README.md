# Tree Prefix Reuse (TPR) for VERL

TPR is a training-side prefix-reuse runtime for VERL's Megatron engine. This recipe packages the runtime separately and uses a small, version-specific patch to connect it to VERL.

## Status

Ruff lint/format, Python compilation and clean pinned VERL Patch application have been checked during the current manual acceptance run. Container imports, Hydra composition and a complete NPU PPO step remain pending.

## Contents

- `verl_tpr/`: TPR runtime implementation.
- `verl-tpr.patch`: version-specific changes to VERL's engine configuration and Megatron execution path.
- `run_uniagent_rebench_ppo.sh`: one PPO example using UniAgent and SWE-ReBench.
- `REQUIRED_VERL.txt`: pinned VERL revision for the patch.
- `pyproject.toml`: standalone Python package metadata.

## Fresh-server installation

Start with [INSTALL_ASCEND.md](INSTALL_ASCEND.md) for host Docker/image setup, the container environment, the local `un-iagent` directory (remote `verl-project/uni-agent`), Qwen model download, public SWE-reBench preparation and one-step acceptance. The shorter steps below assume those dependencies are already available.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the Recipe file layout, the exact VERL integration points, the training execution flow, and the current support boundaries.

## Required VERL revision

Use the exact commit recorded in `REQUIRED_VERL.txt`. The patch is based on the upstream VERL source tree, not on the private `verl-tpr` development fork.

## Prerequisites

This recipe is an integration layer, **not** a complete environment installer. It requires an existing, compatible Ascend NPU runtime with Python 3.11+, PyTorch/torch-npu, VERL/Megatron/MindSpeed, vLLM Ascend, Ray and UniAgent's runtime dependencies. Install these using the upstream/project environment guide first; do not overwrite packages in a working training environment.

You also need:

- A local Qwen3 model directory containing `config.json` and its checkpoints.
- A prepared SWE-ReBench task Parquet with a `prompt` column and data compatible with UniAgent.
- A Uni-Agent checkout (locally named `un-iagent`) containing `uni_agent/framework/entry.py`, its task YAML and required task/reward containers or credentials.
- A dedicated allocation of **8 Ascend NPUs** for the example defaults. The launcher does not stop other Ray jobs or manage devices.
- The **Recipe checkout, patched VERL checkout, UniAgent checkout, model and data visible in the same running environment**. If using Docker, make sure all host paths are mounted and use their container paths below.

### Step 1 — Create a clean, pinned VERL checkout

Do not patch your existing TPR development tree or switch an active training checkout. From a directory where GitHub access works (host or container), set the paths you will use:

```bash
export RECIPE_ROOT=/absolute/path/to/verl-recipe
export VERL_ROOT=/absolute/path/to/new/verl-tpr-recipe-smoke
PINNED_SHA=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
test -n "$PINNED_SHA" || { echo 'Missing VERL_COMMIT'; exit 1; }

# Option A: clone upstream into a fresh directory (requires GitHub access)
git clone https://github.com/verl-project/verl.git "$VERL_ROOT"
git -C "$VERL_ROOT" checkout --detach "$PINNED_SHA"

# Option B instead of clone: from a separate existing upstream VERL Git repo,
# use: git -C /path/to/existing/verl worktree add --detach "$VERL_ROOT" "$PINNED_SHA"
# Fetch the pinned SHA into that repo first if it is not available locally.

test "$(git -C "$VERL_ROOT" rev-parse HEAD)" = "$PINNED_SHA"
git -C "$VERL_ROOT" status --short
```

If Git operations run on the host but training runs inside a container, export `RECIPE_ROOT` and `VERL_ROOT` again using the **container-visible** paths before continuing.

### Step 2 — Apply the version-specific patch

Run inside the environment that can access both directories:

```bash
test -f "$VERL_ROOT/verl/trainer/main_ppo.py"
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" diff --check
git -C "$VERL_ROOT" diff --name-only
```

Expected changed VERL files: `verl/workers/config/engine.py` and `verl/workers/engine/megatron/transformer_impl.py`. The patch must apply cleanly to the pinned upstream revision. An already-patched or unclean checkout is **not** a valid clean-install acceptance test.

### Step 3 — Install and verify the runtime

In the **same Python environment that launches VERL on Ascend**:

```bash
python -m pip install -e "$RECIPE_ROOT/tpr" --no-deps
bash -n "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
export PYTHONPATH="$RECIPE_ROOT/tpr:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from pathlib import Path
from dataclasses import fields
from verl.workers.config.engine import McoreEngineConfig
import verl_tpr
from verl_tpr.adapter import install_tpr_module_spec, run_tpr_forward_backward_batch
assert {'tpr_enabled', 'tpr_cp_backend'} <= {f.name for f in fields(McoreEngineConfig)}
print('TPR import:', Path(verl_tpr.__file__).resolve())
print('VERL Engine TPR config: OK')
PY
```

The runtime's `pyproject.toml` intentionally does **not** install VERL, MindSpeed, or the Ascend framework dependencies. If the import fails, check that the correct training environment is active and that the pinned VERL checkout is first on `PYTHONPATH`.

### Step 4 — Point the launcher at the external inputs

Set all five required paths to real files inside the container; placeholders below must be replaced:

```bash
export VERL_ROOT=/absolute/path/to/patched/verl-tpr-recipe-smoke
export UNIAGENT_ROOT=/workspace/src/un-iagent
export MODEL_PATH=/absolute/path/to/Qwen3-4B-128K
export TRAIN_FILE=/absolute/path/to/prepared-swe-rebench-task.parquet
export TASK_CONFIG=/absolute/path/to/swe-rebench-task-config.yaml

test -f "$VERL_ROOT/verl/trainer/main_ppo.py"
test -f "$UNIAGENT_ROOT/uni_agent/framework/entry.py"
test -f "$MODEL_PATH/config.json"
test -f "$TRAIN_FILE"
test -f "$TASK_CONFIG"
```

`TEST_FILE` defaults to `TRAIN_FILE`. A one-step smoke needs at least one usable task and its reward environment. The configured `MAX_PROMPT_LENGTH` must be long enough for the Parquet prompts after chat templating; filtering can otherwise leave no training examples. Do not assume the example downloads model weights, prepares tasks, or starts the task containers for you.

### Step 5 — Launch one real TPR PPO training step

From the NPU container, with a dedicated 8-NPU allocation and the paths above set:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RUN_ROOT=/absolute/path/to/outputs/tpr-recipe-smoke-1

NNODES=1 N_GPUS_PER_NODE=8 \
TRAIN_TP=2 TRAIN_CP=4 TRAIN_PP=1 TRAIN_SP=False \
TPR_ENABLED=True ACTOR_RECOMPUTE=True \
N=8 CONCURRENCY=4 TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 \
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

By default, the script allows up to `MAX_PROMPT_LENGTH=16384` input tokens and `MAX_RESPONSE_LENGTH=8192` output tokens. Increase or decrease those limits only after checking your selected dataset and the available NPU memory. `TRAIN_CP=4` selects the Ring CP path internally, but successful startup alone does not demonstrate numerical equivalence with the Full baseline. `SESSION_HARD_CAP=True` should be enabled only on a UniAgent checkout that supports that feature.

The script writes `launcher.log`, `agent/` and `ckpts/` beneath `RUN_ROOT` and refuses to overwrite an existing `launcher.log`. Set a **new** `RUN_ROOT` for every run. It does not install dependencies, patch VERL, clean Ray state or allocate devices.

### Step 6 — Inspect the result and, optionally, run Full

```bash
grep -E 'TPR runtime import OK|TPR_ENABLED=|Topology:|Traceback|ERROR|Error|Exception' "$RUN_ROOT/launcher.log" | tail -80
tail -100 "$RUN_ROOT/launcher.log"
```

A successful acceptance run should complete one actor update without an exception, with `TPR_ENABLED=True` and `CP=4` shown in `launcher.log`. Check the actual training metrics and task/reward output rather than treating package import or launcher startup as E2E success.

To smoke-test the native Full path, use the same external inputs and topology, but a **different** `RUN_ROOT` and `TPR_ENABLED=False`. Independent online rollouts are stochastic; their wall-clock times are not a controlled Full/TPR performance comparison. For numerical/performance claims, replay captured identical Actor mini-batches.

## Common first-run failures

| Symptom | Check |
| --- | --- |
| `git apply --check` fails | Upstream VERL SHA mismatch, pre-existing edits or patch already applied; start from a clean pinned checkout |
| `wrong verl_tpr import` / import error | Editable install and `PYTHONPATH` point to a different runtime; ensure Recipe and VERL directories are visible inside Docker |
| Hydra `tpr_enabled`/`tpr_cp_backend` unknown | The patch was not applied to the VERL checkout Python is loading |
| Model or task path validation error | `MODEL_PATH`, `TRAIN_FILE`, `TASK_CONFIG` or `UNIAGENT_ROOT` missing/not mounted inside the container |
| No examples after prompt filtering | `MAX_PROMPT_LENGTH` too small for the prepared Parquet; inspect actual input lengths |
| Ring / MindSpeed import or kernel error | Ascend container dependencies/topology not compatible; inspect the first stack trace, not just the last Ray error |
| Existing `launcher.log` | Use a new `RUN_ROOT`; the launcher intentionally avoids overwrites |

## Validation status

Ruff lint/format, Python compilation and clean pinned VERL Patch application have passed in the current manual acceptance run. **Container imports, Hydra composition and NPU E2E are not yet certified**. See [INSTALL_ASCEND.md](INSTALL_ASCEND.md) for the full installation path. The instructions above are the intended first-run acceptance procedure; they are not evidence that the example has already succeeded.

## Scope

TPR is an optional Megatron actor-training execution path. VERL continues to own PPO orchestration, rollout/reward integration, optimizer/scheduler, and distributed process groups. Experimental placement planners and unrelated profiling/test utilities are intentionally excluded from this recipe.
