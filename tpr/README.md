# Tree Prefix Reuse (TPR) Recipe

TPR is a training-side prefix-reuse runtime for VERL's Megatron engine.

## Run

Follow the [TPR Recipe running guide](INSTALL_ASCEND.md).

## Files

- `verl_tpr/`: TPR runtime.
- `verl-tpr.patch`: VERL integration patch.
- `run_uniagent_rebench_ppo.sh`: PPO training launcher.
- `task_config_swe_rebench_docker.yaml`: Docker Sandbox task configuration.
- `REQUIRED_VERL.txt`: pinned VERL revision.
- `INSTALL_ASCEND.md`: installation and running guide.
- `ARCHITECTURE.md`: implementation structure.
