# TPR Recipe 运行手册

## 1. 环境要求

在 ARM64 Ascend 训练容器中执行以下步骤。容器需要具备：

- Python 3.11。
- Ascend CANN、PyTorch、torch-npu。
- VERL、Megatron-LM、MindSpeed、vLLM-Ascend、Ray。
- Docker CLI，并挂载宿主机 Docker Socket：`/var/run/docker.sock`。
- 8 张可用 NPU。

先检查环境：

~~~bash
python -c "import torch, torch_npu; print(torch.npu.device_count())"
docker info
~~~

## 2. 获取代码

~~~bash
cd /workspace

git clone https://github.com/verl-project/verl-recipe.git
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/uni-agent.git

export RECIPE_ROOT=/workspace/verl-recipe
export VERL_ROOT=/workspace/verl
export UNIAGENT_ROOT=/workspace/uni-agent
~~~

## 3. 安装 TPR 并应用 VERL Patch

~~~bash
cd "$VERL_ROOT"

VERL_COMMIT=$(sed -n 's/^VERL_COMMIT=//p' "$RECIPE_ROOT/tpr/REQUIRED_VERL.txt")
git checkout "$VERL_COMMIT"

git apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git apply "$RECIPE_ROOT/tpr/verl-tpr.patch"

python -m pip install -e "$RECIPE_ROOT/tpr"
python -m pip install -e "$UNIAGENT_ROOT" --no-deps
python -m pip install 'swebench==4.1.0' datasets pandas pyarrow
~~~

## 4. 准备 SWE-ReBench Sandbox

构建 ARM64 Pelita 任务镜像：

~~~bash
bash "$RECIPE_ROOT/tpr/scripts/rebuild_pelita_arm64.sh"
~~~

确认镜像：

~~~bash
docker image inspect uniagent-swe-pelita-875:latest \
  --format '{{.Os}}/{{.Architecture}}'
~~~

输出应为 `linux/arm64`。

准备 SWE-ReBench Parquet 文件，并将任务镜像配置为 `uniagent-swe-pelita-875:latest`。以下示例基于已有的 Pelita Parquet：

~~~bash
export TRAIN_FILE=/workspace/data/pelita875_hint_h2_arm64.parquet

mkdir -p /workspace/data
python - <<'PY'
import copy
import pandas as pd

src = "/path/to/pelita875_hint_h2.parquet"
dst = "/workspace/data/pelita875_hint_h2_arm64.parquet"
image = "uniagent-swe-pelita-875:latest"

df = pd.read_parquet(src)
for index in df.index:
    extra = copy.deepcopy(df.at[index, "extra_info"])
    extra["tools_kwargs"]["task"]["sandbox"]["image"] = image
    df.at[index, "extra_info"] = extra

df.to_parquet(dst, index=False)
print(dst)
PY
~~~

将 Python 脚本中的 src 替换为实际 Parquet 路径。

## 5. 配置运行参数

~~~bash
export MODEL_PATH=/workspace/models/Qwen3-4B-128K
export TRAIN_FILE=/workspace/data/pelita875_hint_h2_arm64.parquet
export TASK_CONFIG="$RECIPE_ROOT/tpr/task_config_swe_rebench_docker.yaml"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export RUN_NAME=tpr-swe-rebench
export RUN_ROOT=/workspace/outputs/$RUN_NAME
~~~

MODEL_PATH 指向本地 Qwen3 模型目录，TRAIN_FILE 指向准备好的 Parquet 文件。

## 6. 启动训练

以下为已验证的 TP2 × CP4 × DP1、N=8 单步运行配置：

~~~bash
env \
  NNODES=1 \
  N_GPUS_PER_NODE=8 \
  TRAIN_TP=2 \
  TRAIN_CP=4 \
  TRAIN_PP=1 \
  TRAIN_SP=False \
  TPR_ENABLED=True \
  ACTOR_RECOMPUTE=True \
  N=8 \
  CONCURRENCY=8 \
  TOTAL_TRAINING_STEPS=1 \
  TOTAL_EPOCHS=1 \
  MAX_PROMPT_LENGTH=18432 \
  MAX_RESPONSE_LENGTH=2048 \
  RUN_NAME="$RUN_NAME" \
  RUN_ROOT="$RUN_ROOT" \
  bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
~~~

## 7. 检查运行结果

~~~bash
grep -E \
  'TPR runtime import OK|Topology:|training/global_step|actor/tpr/|Traceback|ERROR' \
  "$RUN_ROOT/launcher.log"
~~~

确认日志中出现 `training/global_step=1`，且 Actor update 正常结束。
