from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..algorithms import (
    AlgorithmBuildResult,
    DeepEPBuilder,
    DeepEPConfig,
    MoonEPBuilder,
    MoonEPConfig,
    NCCLBuilder,
    NCCLConfig,
)
from ..cost import H100CostModel
from ..graph import TaskGraph
from ..schema import ModelSpec, MoEInvocation, Placement, ValidationError, make_uniform_assignments


AlgorithmName = Literal["nccl", "deepep", "moonep"]


@dataclass(frozen=True)
class TransformerWorkloadConfig:
    model: ModelSpec
    placement: Placement
    tokens_per_rank: int
    algorithm: AlgorithmName
    chunk_tokens: int = 128
    replicas_per_rank: int = 0
    token_padding: int = 128
    dispatch_dtype: str = "fp8"
    combine_dtype: str = "bf16"
    weight_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.tokens_per_rank <= 0:
            raise ValidationError("tokens_per_rank must be positive")
        if self.model.num_experts != self.placement.num_experts:
            raise ValidationError("model and placement expert counts differ")
        if self.algorithm not in {"nccl", "deepep", "moonep"}:
            raise ValidationError(f"unsupported algorithm: {self.algorithm}")


@dataclass(frozen=True)
class WorkloadBuildResult:
    graph: TaskGraph
    metadata: dict[str, object]
    algorithm_results: tuple[AlgorithmBuildResult, ...]


def _attention_flops(tokens: int, model: ModelSpec) -> int:
    projection_flops = 8 * tokens * model.hidden * model.hidden
    attention_flops = 4 * tokens * model.sequence_length * model.hidden
    return projection_flops + attention_flops


def _router_flops(tokens: int, model: ModelSpec) -> int:
    return 2 * tokens * model.hidden * model.num_experts


def build_transformer_workload(
    config: TransformerWorkloadConfig,
    *,
    cost_model: H100CostModel | None = None,
) -> WorkloadBuildResult:
    cost = cost_model or H100CostModel()
    graph = TaskGraph(config.model.name, config.placement.num_ranks)
    tokens = (config.tokens_per_rank,) * config.placement.num_ranks

    if config.algorithm == "nccl":
        algorithm_builder = NCCLBuilder(
            cost,
            NCCLConfig(chunk_routes=config.chunk_tokens),
        )
    elif config.algorithm == "moonep":
        algorithm_builder = MoonEPBuilder(
            cost,
            MoonEPConfig(
                replicas_per_rank=config.replicas_per_rank,
                token_padding=config.token_padding,
                chunk_tokens=config.chunk_tokens,
            ),
        )
    else:
        algorithm_builder = DeepEPBuilder(
            cost,
            DeepEPConfig(chunk_tokens=config.chunk_tokens),
        )

    previous_attention: dict[int, str] = {}
    algorithm_results: list[AlgorithmBuildResult] = []
    for micro_batch in range(config.model.micro_batches):
        attention_keys: dict[int, str] = {}
        for rank in range(config.placement.num_ranks):
            key = f"mb{micro_batch}.attention.rank{rank}"
            predecessors = (
                {previous_attention[rank]} if rank in previous_attention else set()
            )
            graph.add_compute(
                key,
                rank,
                cost.estimate(
                    _attention_flops(config.tokens_per_rank, config.model),
                    overlaps_communication=micro_batch > 0,
                ),
                predecessors=predecessors,
                metadata={
                    "operation": "attention",
                    "micro_batch": micro_batch,
                    "cost_status": "theoretical_flops",
                },
            )
            attention_keys[rank] = key
            previous_attention[rank] = key

        router_keys: set[str] = set()
        for rank in range(config.placement.num_ranks):
            key = f"mb{micro_batch}.router.rank{rank}"
            graph.add_compute(
                key,
                rank,
                cost.estimate(
                    _router_flops(config.tokens_per_rank, config.model),
                    overlaps_communication=micro_batch > 0,
                ),
                predecessors={attention_keys[rank]},
                metadata={
                    "operation": "router_projection",
                    "micro_batch": micro_batch,
                    "cost_status": "theoretical_flops",
                },
            )
            router_keys.add(key)

        invocation = MoEInvocation(
            invocation_id=f"mb{micro_batch}.moe",
            placement=config.placement,
            tokens_per_source_rank=tokens,
            hidden=config.model.hidden,
            ffn_hidden=config.model.ffn_hidden,
            topk=config.model.topk,
            dispatch_dtype=config.dispatch_dtype,
            combine_dtype=config.combine_dtype,
            weight_dtype=config.weight_dtype,
            assignments=make_uniform_assignments(
                tokens,
                config.model.topk,
                config.model.num_experts,
            ),
        )
        algorithm_results.append(
            algorithm_builder.build(graph, invocation, entry_keys=router_keys)
        )

    graph.validate()
    metadata: dict[str, object] = {
        "generator": "moe_dag_v1",
        "algorithm": config.algorithm,
        "model": {
            "name": config.model.name,
            "hidden": config.model.hidden,
            "ffn_hidden": config.model.ffn_hidden,
            "num_attention_heads": config.model.num_attention_heads,
            "num_kv_heads": config.model.num_kv_heads,
            "head_dim": config.model.head_dim,
            "num_experts": config.model.num_experts,
            "topk": config.model.topk,
            "sequence_length": config.model.sequence_length,
            "micro_batches": config.model.micro_batches,
            "batch_size": config.model.batch_size,
            "dtype": config.model.dtype,
        },
        "placement": {
            "num_ranks": config.placement.num_ranks,
            "gpus_per_server": config.placement.gpus_per_server,
            "expert_to_rank": list(config.placement.expert_to_rank),
        },
        "tokens_per_rank": config.tokens_per_rank,
        "chunk_tokens": config.chunk_tokens,
        "replicas_per_rank": config.replicas_per_rank,
        "token_padding": config.token_padding,
        "dtypes": {
            "dispatch": config.dispatch_dtype,
            "combine": config.combine_dtype,
            "weight": config.weight_dtype,
        },
        "routing_provider": {
            "name": "uniform_deterministic",
            "random_seed": None,
        },
        "compute_cost": cost.manifest(),
        "scope": {
            "network": "packet_simulated_flow_completion_dependencies",
            "chunk_pipeline": "explicit_flow_tasks",
            "compute": "fixed_theoretical_duration",
            "communication_sms_per_rank_phase": cost.communication_sms,
            "single_flow_packet_progress_events": False,
            "dynamic_gpu_resource_scheduling": False,
        },
        "micro_batch_algorithms": [result.metadata for result in algorithm_results],
    }
    return WorkloadBuildResult(graph, metadata, tuple(algorithm_results))
