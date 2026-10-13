#!/usr/bin/env bash

# Evaluation
# ++reward.*: main_eval looks the reward function up under `reward`, which its config
# (evaluation.yaml) does not define, so the key is added here.
python3 -m verl.trainer.main_eval \
    data.path=$HOME/data/gen/qwen_8b_gen_test.parquet \
    ++reward.custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    ++reward.custom_reward_function.name=compute_score_data_source \
    "$@"
