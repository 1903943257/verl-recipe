"""Check that every training-capable Ray node imports the same TPR runtime.

Usage (from a configured Ray client)::
    ray job submit --runtime-env "$RUNTIME_ENV" -- \
      python -m verl_tpr.check_ray_runtime
"""

from __future__ import annotations

import json
import socket

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
def _probe() -> dict[str, str]:
    import importlib.metadata

    import verl_tpr
    from verl_tpr.adapter import run_tpr_forward_backward_batch

    return {
        "hostname": socket.gethostname(),
        "module": str(verl_tpr.__file__),
        "version": importlib.metadata.version("verl-tpr"),
        "engine_hook": run_tpr_forward_backward_batch.__name__,
    }


def main() -> None:
    ray.init(address="auto")
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if not nodes:
        raise RuntimeError("Ray has no alive nodes")

    # Probe accelerator nodes rather than a driver-only/head node when possible.
    target_nodes = [
        node
        for node in nodes
        if any(
            ("gpu" in name.lower() or "npu" in name.lower()) and amount > 0
            for name, amount in node.get("Resources", {}).items()
        )
    ] or nodes

    futures = [
        _probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote()
        for node in target_nodes
    ]
    for node, result in zip(target_nodes, ray.get(futures, timeout=120), strict=True):
        print("TPR_RAY_IMPORT_OK " + json.dumps({"node_id": node["NodeID"], **result}, sort_keys=True))


if __name__ == "__main__":
    main()
