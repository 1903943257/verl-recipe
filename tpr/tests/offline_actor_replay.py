# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Training-side TP2 x Ring-CP4 x SP replay against the Recipe TPR runtime.

This is an offline recorded-Actor test. It does not start Ray, vLLM, Uni-Agent,
Claude Code, or rollout. Both arms use verl_tpr.adapter:
- unshared: each captured trajectory owns an independent tree;
- tpr: captured trajectory keys are preserved so shared prefixes can be reused.

The unshared arm is a controlled TPR-runtime reference, not native VERL Full.
No optimizer step is applied. The test checks that the Recipe adapter completes
real Qwen3-4B forward/backward, Ring communication, and gradient finalization
with native Megatron sequence parallelism enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


class _PPO:
    policy_loss = {"loss_mode": "vanilla"}
    loss_agg_mode = "token-mean"
    clip_ratio = 0.2
    clip_ratio_low = None
    clip_ratio_high = None
    entropy_coeff = 0.0
    use_kl_loss = False
    loss_scale_factor = None

    def __init__(self):
        self.global_batch_info = {}

    def get(self, key, default=None):
        return getattr(self, key, default)


class _RingCPOnlyCoreAttention:
    """Construction sentinel; all active attention must go through Recipe Ring."""

    def __new__(cls, *args, **kwargs):
        from megatron.core.transformer.module import MegatronModule

        class _Impl(MegatronModule):
            def __init__(
                self,
                config,
                layer_number,
                attn_mask_type,
                attention_type,
                attention_dropout=None,
                softmax_scale=None,
                cp_comm_type=None,
                pg_collection=None,
            ):
                super().__init__(config=config)
                if getattr(config, "apply_query_key_layer_scaling", False):
                    raise NotImplementedError("QK layer scaling is outside this replay")
                self.layer_number = max(1, layer_number)
                self.attn_mask_type = attn_mask_type
                self.attention_type = attention_type
                self.softmax_scale = (
                    1.0 / math.sqrt(config.kv_channels)
                    if softmax_scale is None
                    else float(softmax_scale)
                )

            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "TPR replay reached the square core attention; expected Recipe Ring CP"
                )

        return _Impl(*args, **kwargs)


def _sanitize_mindspeed_args(args: Any) -> None:
    values = vars(args)
    if "" in values:
        value = values[""]
        expected = (
            value is None
            or isinstance(value, (bool, str))
            or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        )
        if not expected:
            raise RuntimeError(
                f"Unexpected MindSpeed empty argument value type: {type(value).__name__}"
            )
        del values[""]
    malformed = [
        repr(key) for key in values if not isinstance(key, str) or not key.isidentifier()
    ]
    if malformed:
        raise RuntimeError(f"Invalid MindSpeed argument names after repatch: {malformed}")


def _initialize_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("This replay requires exactly eight NPU processes (TP2 x CP4 x DP1)")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    saved_argv = sys.argv[:]
    try:
        # MindSpeed parses sys.argv while patching TransformerConfig.
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.args_utils import get_full_args
        from mindspeed.megatron_adaptor import repatch

        repatch(
            {
                "tensor_model_parallel_size": 2,
                "context_parallel_size": 4,
                "pipeline_model_parallel_size": 1,
                "context_parallel_algo": "megatron_cp_algo",
                "sequence_parallel": True,
                "use_flash_attn": True,
            }
        )
        args = get_full_args()
        _sanitize_mindspeed_args(args)
        if not bool(args.sequence_parallel):
            raise AssertionError("MindSpeed did not enable sequence_parallel")
        if args.context_parallel_algo != "megatron_cp_algo":
            raise AssertionError(
                f"Unexpected context_parallel_algo={args.context_parallel_algo!r}"
            )
    finally:
        sys.argv[:] = saved_argv

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=4,
            expert_model_parallel_size=1,
        )

    groups = {
        "tp": parallel_state.get_tensor_model_parallel_group(),
        "cp": parallel_state.get_context_parallel_group(),
        "dp": parallel_state.get_data_parallel_group(),
        "pp": parallel_state.get_pipeline_model_parallel_group(),
    }
    actual = {
        "tp": dist.get_world_size(groups["tp"]),
        "cp": dist.get_world_size(groups["cp"]),
        "dp": dist.get_world_size(groups["dp"]),
        "pp": dist.get_world_size(groups["pp"]),
    }
    if actual != {"tp": 2, "cp": 4, "dp": 1, "pp": 1}:
        raise AssertionError(f"Megatron process groups mismatch: {actual}")

    model_parallel_cuda_manual_seed(261013)
    return SimpleNamespace(
        **{f"{name}_group": group for name, group in groups.items()},
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(),
        dp_rank=parallel_state.get_data_parallel_rank(),
        global_rank=dist.get_rank(),
        device=torch.device("npu", local_rank),
        tp_size=2,
        cp_size=4,
        dp_size=1,
    )


def _load_qwen_config(model_path: Path):
    from transformers import AutoConfig

    config_path = model_path / "config.json"
    if not config_path.is_file() or not tuple(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Qwen3-4B config or safetensors weights missing under {model_path}")
    hf = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    for name, value in expected.items():
        if getattr(hf, name, None) != value:
            raise AssertionError(
                f"Expected Qwen3-4B {name}={value!r}, got {getattr(hf, name, None)!r}"
            )
    if getattr(hf, "attention_dropout", 0.0) != 0:
        raise AssertionError("TPR replay requires attention_dropout=0")
    return hf


def _load_hf_state_dict(model_path: Path):
    from safetensors.torch import load_file

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = [item.name for item in sorted(model_path.glob("*.safetensors"))]
    state = {}
    for name in shard_names:
        state.update(load_file(str(model_path / name), device="cpu"))
    return state


@torch.no_grad()
def _load_qwen3_tp2_weights(model, hf, model_path: Path, tp_rank: int):
    state = _load_hf_state_dict(model_path)
    loaded = set()

    def put(target, value, source):
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f"Qwen3 TP shard shape mismatch for {source}: "
                f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        loaded.add(id(target))

    def source(name):
        if name not in state:
            raise KeyError(f"Qwen3 checkpoint missing {name}")
        return state[name]

    def row_shard(value):
        if value.shape[0] % 2:
            raise ValueError("Qwen3 row-parallel weight is not divisible by TP2")
        return value.chunk(2, dim=0)[tp_rank].contiguous()

    def col_shard(value):
        if value.shape[1] % 2:
            raise ValueError("Qwen3 column-parallel weight is not divisible by TP2")
        return value.chunk(2, dim=1)[tp_rank].contiguous()

    put(
        model.embedding.word_embeddings.weight,
        row_shard(source("model.embed_tokens.weight")),
        "model.embed_tokens.weight",
    )
    head_dim = hf.head_dim
    groups = hf.num_key_value_heads
    queries_per_group = hf.num_attention_heads // groups
    groups_per_rank = groups // 2

    for index, layer in enumerate(model.decoder.layers):
        prefix = f"model.layers.{index}"
        attention = f"{prefix}.self_attn"
        put(layer.input_layernorm.weight, source(f"{prefix}.input_layernorm.weight"), prefix)
        q = source(f"{attention}.q_proj.weight").reshape(
            groups, queries_per_group * head_dim, hf.hidden_size
        )
        k = source(f"{attention}.k_proj.weight").reshape(
            groups, head_dim, hf.hidden_size
        )
        v = source(f"{attention}.v_proj.weight").reshape(
            groups, head_dim, hf.hidden_size
        )
        start = tp_rank * groups_per_rank
        stop = start + groups_per_rank
        qkv = torch.cat(
            (q[start:stop], k[start:stop], v[start:stop]), dim=1
        ).reshape(-1, hf.hidden_size).contiguous()
        put(layer.self_attention.linear_qkv.weight, qkv, f"{attention}.qkv")
        put(
            layer.self_attention.q_layernorm.weight,
            source(f"{attention}.q_norm.weight"),
            f"{attention}.q_norm.weight",
        )
        put(
            layer.self_attention.k_layernorm.weight,
            source(f"{attention}.k_norm.weight"),
            f"{attention}.k_norm.weight",
        )
        put(
            layer.self_attention.linear_proj.weight,
            col_shard(source(f"{attention}.o_proj.weight")),
            f"{attention}.o_proj.weight",
        )
        put(
            layer.pre_mlp_layernorm.weight,
            source(f"{prefix}.post_attention_layernorm.weight"),
            f"{prefix}.post_attention_layernorm.weight",
        )
        fc1 = torch.cat(
            (
                row_shard(source(f"{prefix}.mlp.gate_proj.weight")),
                row_shard(source(f"{prefix}.mlp.up_proj.weight")),
            ),
            dim=0,
        )
        put(layer.mlp.linear_fc1.weight, fc1, f"{prefix}.mlp.gate/up")
        put(
            layer.mlp.linear_fc2.weight,
            col_shard(source(f"{prefix}.mlp.down_proj.weight")),
            f"{prefix}.mlp.down_proj.weight",
        )

    put(model.decoder.final_layernorm.weight, source("model.norm.weight"), "model.norm.weight")
    missing = [name for name, param in model.named_parameters() if id(param) not in loaded]
    if missing:
        raise RuntimeError(f"Qwen3 TP2 checkpoint loader missed parameters: {missing}")
    if model.output_layer.weight is not None:
        raise AssertionError("Qwen3-4B tied embeddings must not create a second LM-head weight")
    del state


def _build_model(runtime, model_path: Path):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from verl.models.mcore.config_converter import (
        get_hf_rope_theta,
        hf_to_mcore_config_dense,
    )
    from verl_tpr.adapter import _get_layer_specs, replace_self_attention_with_tpr

    hf = _load_qwen_config(model_path)
    config = hf_to_mcore_config_dense(
        hf,
        torch.bfloat16,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        expert_model_parallel_size=1,
        sequence_parallel=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
    )
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    spec = replace_self_attention_with_tpr(spec)
    for layer_spec in _get_layer_specs(spec):
        layer_spec.submodules.self_attention.submodules.core_attention = _RingCPOnlyCoreAttention

    pg = SimpleNamespace(
        tp=runtime.tp_group,
        cp=runtime.cp_group,
        pp=runtime.pp_group,
        embd=None,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=hf.vocab_size,
        max_sequence_length=hf.max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_base=get_hf_rope_theta(hf),
        pg_collection=pg,
    ).to(device=runtime.device, dtype=torch.bfloat16)
    if hasattr(model.rotary_pos_emb, "inv_freq"):
        model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if getattr(module, "tp_group", "missing") is None:
            module.tp_group = runtime.tp_group

    _load_qwen3_tp2_weights(model, hf, model_path, runtime.tp_rank)
    if len(model.decoder.layers) != 36:
        raise AssertionError(f"Expected 36 decoder layers, got {len(model.decoder.layers)}")
    if config.sequence_parallel is not True:
        raise AssertionError("The constructed Qwen3 model did not retain sequence_parallel=True")
    from verl_tpr.attention import TPRSelfAttention

    if not all(isinstance(layer.self_attention, TPRSelfAttention) for layer in model.decoder.layers):
        raise AssertionError("Not every decoder layer uses the Recipe TPRSelfAttention")
    model.train()
    return model


def _load_capture(path: Path, rows: tuple[int, ...], mode: str):
    from verl.utils import tensordict_utils as tu

    if not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected a recorded actor_update_mini_batch capture")
    source = raw["tensordict"]
    required = (
        "input_ids",
        "prompts",
        "responses",
        "response_mask",
        "loss_mask",
        "advantages",
        "old_log_probs",
        "temperature",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise AssertionError(f"Actor capture is missing PPO fields: {missing}")
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= len(raw["keys"]):
        raise ValueError(f"Invalid selected capture rows {rows}; capture has {len(raw['keys'])} rows")

    data = tu.index_select_tensor_dict(source, list(rows))
    original_keys = tuple(raw["keys"][index] for index in rows)
    if mode == "unshared":
        keys = tuple(f"offline_full_{index}_tpr_0" for index in range(len(rows)))
    else:
        keys = original_keys

    token_count = sum(int(mask.to(torch.bool).sum().item()) for mask in data["response_mask"].unbind(0))
    if token_count <= 0:
        raise AssertionError("Selected actor rows contain no PPO response tokens")
    for ids, prompt, response, mask, advantage, old_log_prob in zip(
        data["input_ids"].unbind(0),
        data["prompts"].unbind(0),
        data["responses"].unbind(0),
        data["response_mask"].unbind(0),
        data["advantages"].unbind(0),
        data["old_log_probs"].unbind(0),
        strict=True,
    ):
        if not torch.equal(ids, torch.cat((prompt, response))):
            raise AssertionError("Captured input_ids must equal prompt + response")
        if mask.numel() != response.numel():
            raise AssertionError("response_mask length differs from response")
        if advantage.numel() != response.numel() or old_log_prob.numel() != response.numel():
            raise AssertionError("Captured advantages / old_log_probs do not align with response")

    tu.assign_non_tensor(
        data,
        tpr_trajectory_keys=keys,
        dp_size=1,
        batch_num_tokens=token_count,
        global_batch_size=len(rows),
        tpr_capture_log_probs=True,
    )
    digest = hashlib.sha256()
    for name in ("input_ids", "response_mask", "old_log_probs", "advantages"):
        for tensor in data[name].unbind(0):
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return data, original_keys, token_count, digest.hexdigest()


def _sample_tensor(value: torch.Tensor):
    flat = value.detach().reshape(-1)
    count = min(32, flat.numel())
    indices = torch.linspace(
        0, max(flat.numel() - 1, 0), steps=count, device=flat.device
    ).long()
    return {
        "shape": tuple(value.shape),
        "indices": indices.cpu(),
        "values": flat.index_select(0, indices).float().cpu(),
    }


def _gradient_witness(model):
    result = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        result[name] = _sample_tensor(param.grad)
    return result


def _global_gradient_norm(model, runtime):
    local = torch.zeros((), dtype=torch.float32, device=runtime.device)
    if runtime.cp_rank == 0:
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Missing gradient for global norm: {name}")
            tp_sharded = bool(getattr(param, "tensor_model_parallel", False))
            if tp_sharded or runtime.tp_rank == 0:
                grad = param.grad.detach().float()
                local += (grad * grad).sum()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(local).item())


def _metric_value(metrics, name):
    value = metrics.get(name)
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    if value is not None:
        return float(value)
    return None


def _run_mode(mode: str, args):
    runtime = _initialize_runtime()
    from verl.workers.utils.losses import ppo_loss
    from verl_tpr.adapter import run_tpr_forward_backward_batch
    from verl_tpr.tree import build_tree_execution_plans

    model_path = Path(args.model).resolve()
    capture_path = Path(args.capture).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = tuple(int(value) for value in args.rows.split(","))
    batch, original_keys, token_count, data_sha = _load_capture(capture_path, rows, mode)
    forest = build_tree_execution_plans(
        tuple(batch["tpr_trajectory_keys"].tolist())
        if hasattr(batch["tpr_trajectory_keys"], "tolist")
        else tuple(batch["tpr_trajectory_keys"]),
        batch,
        tree_policy="qwen_turn",
        turn_min_segment_tokens=args.turn_min_segment_tokens,
    )
    if mode == "unshared" and (
        len(forest.trees) != len(rows)
        or any(len(tree.member_rows) != 1 for tree in forest.trees)
    ):
        raise AssertionError("Unshared control unexpectedly merged distinct actor rows")

    model = _build_model(runtime, model_path)
    config = model.config
    config.no_sync_func = None
    config.grad_scale_func = None
    config.calculate_per_token_loss = False
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None

    def finalize(chunks, num_tokens, *, force_all_reduce=False, **kwargs):
        if chunks != [model] or num_tokens is not None or not force_all_reduce:
            raise AssertionError("Unexpected native Megatron gradient-finalization arguments")
        torch.npu.synchronize()
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"No gradient before CP finalization: {name}")
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
            param.grad.div_(runtime.cp_size)
        torch.npu.synchronize()

    config.finalize_model_grads_func = finalize
    engine = SimpleNamespace(
        module=[model],
        engine_config=SimpleNamespace(
            tpr_enabled=True,
            tensor_model_parallel_size=2,
            context_parallel_size=4,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            dynamic_context_parallel=False,
            use_fused_kernels=False,
        ),
        model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False)),
        enable_routing_replay=False,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_group=lambda: runtime.dp_group,
    )

    model.zero_grad(set_to_none=True)
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated_before = int(torch.npu.memory_allocated())
    start = time.perf_counter()
    result = run_tpr_forward_backward_batch(
        engine,
        batch,
        partial(ppo_loss, config=_PPO()),
        forward_only=False,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    loss = float(result["loss"][0])
    if not math.isfinite(loss):
        raise FloatingPointError(f"Non-finite Recipe TPR loss: {loss}")
    grad_norm = _global_gradient_norm(model, runtime)
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError(f"Invalid finalized gradient norm: {grad_norm}")
    gradients = _gradient_witness(model)
    log_probs = getattr(engine, "_tpr_captured_log_probs", None)
    if not isinstance(log_probs, dict) or not log_probs:
        raise AssertionError("Recipe objective did not return the requested logical log-prob witness")

    logical_input_tokens = sum(int(row.numel()) for row in batch["input_ids"].unbind(0))
    unique_tree_tokens = sum(
        segment.length for tree in forest.trees for segment in tree.segment_plan.segments.values()
    )
    metrics = result.get("metrics", {})
    payload = {
        "mode": mode,
        "rank": runtime.global_rank,
        "tp_rank": runtime.tp_rank,
        "cp_rank": runtime.cp_rank,
        "dp_rank": runtime.dp_rank,
        "tp": runtime.tp_size,
        "cp": runtime.cp_size,
        "dp": runtime.dp_size,
        "sp": True,
        "model": "Qwen3-4B",
        "checkpoint": str(model_path),
        "capture": str(capture_path),
        "source_rows": rows,
        "source_keys": original_keys,
        "data_sha256": data_sha,
        "objective": "GRPO advantages + vanilla PPO token-mean",
        "logical_ppo_tokens": token_count,
        "forest_trees": len(forest.trees),
        "forest_segments": forest.segment_count,
        "logical_input_tokens": logical_input_tokens,
        "unique_tree_tokens": unique_tree_tokens,
        "topology_reuse_ratio": logical_input_tokens / max(unique_tree_tokens, 1),
        "loss": loss,
        "global_bf16_grad_norm": grad_norm,
        "gradient_samples": gradients,
        "logprobs": log_probs,
        "metrics": {
            name: _metric_value(metrics, name)
            for name in (
                "tpr/forest_trees",
                "tpr/physical_segments",
                "tpr/logical_loss_tokens",
                "tpr/logical_input_tokens",
                "tpr/unique_tree_tokens",
                "tpr/topology_token_reuse_ratio",
                "tpr/active_tree_tokens",
                "tpr/active_physical_segments",
            )
        },
        "forward_backward_seconds": elapsed,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
    }
    target = output_dir / f"{mode}_rank{runtime.global_rank}.pt"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite replay result: {target}")
    torch.save(payload, target)
    print(
        "TPR_RECIPE_REPLAY "
        f"mode={mode} rank={runtime.global_rank} TP=2 CP=4 DP=1 SP=1 "
        f"loss={loss:.9g} grad_norm={grad_norm:.9g} "
        f"trees={len(forest.trees)} segments={forest.segment_count} "
        f"reuse={payload['topology_reuse_ratio']:.4f} "
        f"fwd_bwd_s={elapsed:.3f} "
        f"peak_alloc_gib={payload['peak_allocated_bytes'] / 2**30:.3f} "
        f"saved={target}",
        flush=True,
    )

    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _collect_log_probs(items):
    values = {}
    for item in items:
        if item["tp_rank"] != 0:
            continue
        for key, value in item["logprobs"].items():
            key = tuple(key)
            if key in values:
                raise AssertionError(f"Duplicate logical log-prob owner {key}")
            values[key] = float(value)
    return values


def _compare(output_dir: Path):
    snapshots = {}
    for mode in ("unshared", "tpr"):
        paths = [output_dir / f"{mode}_rank{rank}.pt" for rank in range(8)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {mode} replay ranks: {missing}")
        snapshots[mode] = [
            torch.load(path, map_location="cpu", weights_only=False) for path in paths
        ]

    unshared, tpr = snapshots["unshared"], snapshots["tpr"]
    for rank, (left, right) in enumerate(zip(unshared, tpr, strict=True)):
        for key in ("rank", "tp", "cp", "dp", "sp", "source_rows", "data_sha256", "checkpoint"):
            if left[key] != right[key]:
                raise AssertionError(f"Replay inputs differ on rank {rank}: {key}")
        if left["rank"] != rank or not math.isfinite(left["loss"]) or not math.isfinite(right["loss"]):
            raise AssertionError(f"Invalid rank/loss snapshot at rank {rank}")

    left_lp = _collect_log_probs(unshared)
    right_lp = _collect_log_probs(tpr)
    if left_lp.keys() != right_lp.keys():
        raise AssertionError(
            f"Logical log-prob ownership differs: unshared={len(left_lp)} TPR={len(right_lp)}"
        )
    keys = sorted(left_lp)
    left_values = torch.tensor([left_lp[key] for key in keys], dtype=torch.float64)
    right_values = torch.tensor([right_lp[key] for key in keys], dtype=torch.float64)
    logprob_rel = float(
        torch.linalg.vector_norm(left_values - right_values)
        / torch.linalg.vector_norm(left_values).clamp_min(1e-30)
    )
    logprob_max = float((left_values - right_values).abs().max())

    numerator = denominator = 0.0
    for left, right in zip(unshared, tpr, strict=True):
        if left["gradient_samples"].keys() != right["gradient_samples"].keys():
            raise AssertionError(f"Gradient parameter names differ on rank {left['rank']}")
        for name, a in left["gradient_samples"].items():
            b = right["gradient_samples"][name]
            if a["shape"] != b["shape"] or not torch.equal(a["indices"], b["indices"]):
                raise AssertionError(f"Gradient witness mismatch: rank={left['rank']} {name}")
            x = a["values"].double()
            y = b["values"].double()
            numerator += float(((x - y) ** 2).sum())
            denominator += float((x * x).sum())
    grad_rel = math.sqrt(numerator / max(denominator, 1e-30))
    loss_gap = abs(unshared[0]["loss"] - tpr[0]["loss"])
    if not all(math.isfinite(value) for value in (logprob_rel, logprob_max, grad_rel, loss_gap)):
        raise FloatingPointError("Non-finite Full/TPR comparison metric")

    print(
        "TPR_RECIPE_REPLAY_COMPARISON "
        f"topology=TP2xCP4xDP1_SP1 rows={unshared[0]['source_rows']} "
        f"ppo_tokens={len(keys)} "
        f"loss_unshared={unshared[0]['loss']:.9g} loss_tpr={tpr[0]['loss']:.9g} "
        f"loss_gap={loss_gap:.9g} logprob_rel_l2={logprob_rel:.9g} "
        f"logprob_max_abs={logprob_max:.9g} sampled_grad_rel_l2={grad_rel:.9g}"
    )
    print(
        "NOTE: unshared is a no-cross-trajectory-sharing control using the same "
        "Recipe SegmentExecutor/Ring path; this is not native VERL Full parity. "
        "Numerical differences are reported, not judged against an unapproved threshold."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("unshared", "tpr"))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--capture")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--rows", default=os.getenv("TPR_OFFLINE_ROWS", "0,1"))
    parser.add_argument(
        "--turn-min-segment-tokens",
        type=int,
        default=int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS", "128")),
    )
    args = parser.parse_args()
    if args.compare is not None:
        _compare(args.compare.resolve())
        return
    for name in ("mode", "capture", "model", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required for a replay run")
    _run_mode(args.mode, args)


if __name__ == "__main__":
    main()
