from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..algorithms import (
    AlgorithmBuildResult,
    DeepEPBuilder,
    DeepEPConfig,
    EPLBBuilder,
    EPLBConfig,
    MoonEPBuilder,
    MoonEPConfig,
    NCCLBuilder,
    NCCLConfig,
)
from ..cost import ComputeCostModel, H100CostModel
from ..gate import BalancedPermutedGateProvider, GateProvider
from ..graph import TaskGraph
from ..schema import ModelSpec, MoEInvocation, Placement, ValidationError
from .streams import apply_double_buffered_two_stream_schedule


AlgorithmName = Literal["nccl", "deepep", "eplb", "moonep"]


@dataclass(frozen=True)
class TransformerWorkloadConfig:
    model: ModelSpec
    placement: Placement
    tokens_per_rank: int
    algorithm: AlgorithmName
    chunk_tokens: int = 128
    replicas_per_rank: int = 0
    token_padding: int = 128
    eplb_num_physical_experts: int = 0
    eplb_num_groups: int = 0
    eplb_estimated_loads: tuple[float, ...] | None = None
    eplb_load_source: str = "current_invocation_proxy"
    gate_provider: GateProvider = field(
        default_factory=BalancedPermutedGateProvider
    )
    dispatch_dtype: str = "fp8"
    combine_dtype: str = "bf16"
    weight_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.tokens_per_rank <= 0:
            raise ValidationError("tokens_per_rank must be positive")
        if self.model.num_experts != self.placement.num_experts:
            raise ValidationError("model and placement expert counts differ")
        if self.algorithm not in {"nccl", "deepep", "eplb", "moonep"}:
            raise ValidationError(f"unsupported algorithm: {self.algorithm}")
        if self.eplb_num_physical_experts < 0:
            raise ValidationError("eplb_num_physical_experts must be non-negative")
        if self.eplb_num_groups < 0:
            raise ValidationError("eplb_num_groups must be non-negative")
        if (
            self.eplb_estimated_loads is not None
            and len(self.eplb_estimated_loads) != self.model.num_experts
        ):
            raise ValidationError(
                "eplb_estimated_loads length must equal model num_experts"
            )


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
    cost_model: ComputeCostModel | None = None,
) -> WorkloadBuildResult:
    cost = cost_model or H100CostModel()
    graph = TaskGraph(config.model.name, config.placement.num_ranks)
    tokens = (config.tokens_per_rank,) * config.placement.num_ranks

    if config.algorithm == "nccl":
        algorithm_builder = NCCLBuilder(
            cost,
            NCCLConfig(chunk_routes=config.chunk_tokens),
        )
    elif config.algorithm == "eplb":
        physical_experts = config.eplb_num_physical_experts
        if physical_experts == 0:
            baseline_slots = (
                (config.model.num_experts + config.placement.num_ranks - 1)
                // config.placement.num_ranks
            ) * config.placement.num_ranks
            physical_experts = baseline_slots + config.placement.num_ranks
        algorithm_builder = EPLBBuilder(
            cost,
            EPLBConfig(
                num_physical_experts=physical_experts,
                num_groups=(
                    config.eplb_num_groups or config.placement.num_servers
                ),
                chunk_tokens=config.chunk_tokens,
                estimated_loads=config.eplb_estimated_loads,
                load_source=config.eplb_load_source,
            ),
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

    algorithm_results: list[AlgorithmBuildResult] = []
    algorithm_metadata: list[dict[str, object]] = []
    microbatch_task_keys: list[tuple[str, ...]] = []
    for micro_batch in range(config.model.micro_batches):
        before_microbatch = len(graph.tasks)
        previous_rank_terminals: dict[int, frozenset[str]] = {}
        for layer in range(config.model.num_layers):
            before_layer = len(graph.tasks)
            scope = f"mb{micro_batch}"
            if config.model.num_layers > 1:
                scope += f".layer{layer}"

            attention_keys: dict[int, str] = {}
            for rank in range(config.placement.num_ranks):
                key = f"{scope}.attention.rank{rank}"
                graph.add_compute(
                    key,
                    rank,
                    cost.estimate(
                        _attention_flops(config.tokens_per_rank, config.model),
                        operation="attention",
                        overlaps_communication=micro_batch % 2 == 1,
                        token_count=config.tokens_per_rank,
                    ),
                    predecessors=set(previous_rank_terminals.get(rank, ())),
                    metadata={
                        "operation": "attention",
                        "micro_batch": micro_batch,
                        "layer": layer,
                        "operation_flops_status": "theoretical_formula",
                    },
                )
                attention_keys[rank] = key

            router_keys: set[str] = set()
            for rank in range(config.placement.num_ranks):
                key = f"{scope}.router.rank{rank}"
                graph.add_compute(
                    key,
                    rank,
                    cost.estimate(
                        _router_flops(config.tokens_per_rank, config.model),
                        operation="router_projection",
                        overlaps_communication=micro_batch % 2 == 1,
                        token_count=config.tokens_per_rank,
                    ),
                    predecessors={attention_keys[rank]},
                    metadata={
                        "operation": "router_projection",
                        "micro_batch": micro_batch,
                        "layer": layer,
                        "operation_flops_status": "theoretical_formula",
                    },
                )
                router_keys.add(key)

            gate_sample = config.gate_provider.sample(
                layer_id=layer,
                microbatch_id=micro_batch,
                tokens_per_source_rank=tokens,
                placement=config.placement,
                topk=config.model.topk,
            )
            invocation = MoEInvocation(
                invocation_id=f"{scope}.moe",
                placement=config.placement,
                tokens_per_source_rank=tokens,
                hidden=config.model.hidden,
                ffn_hidden=config.model.ffn_hidden,
                topk=config.model.topk,
                dispatch_dtype=config.dispatch_dtype,
                combine_dtype=config.combine_dtype,
                weight_dtype=config.weight_dtype,
                assignments=gate_sample.assignments,
            )
            algorithm_result = algorithm_builder.build(
                graph, invocation, entry_keys=router_keys
            )
            for task in graph.tasks[before_layer:]:
                task.metadata["micro_batch"] = micro_batch
                task.metadata["layer"] = layer
            algorithm_results.append(algorithm_result)
            algorithm_metadata.append(
                {
                    "micro_batch": micro_batch,
                    "layer": layer,
                    "gate": gate_sample.metadata,
                    **algorithm_result.metadata,
                }
            )
            previous_rank_terminals = algorithm_result.rank_terminal_keys
        microbatch_task_keys.append(
            tuple(task.key for task in graph.tasks[before_microbatch:])
        )

    stream_schedule = apply_double_buffered_two_stream_schedule(
        graph, tuple(microbatch_task_keys)
    )
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
            "num_layers": config.model.num_layers,
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
        "eplb": {
            "num_physical_experts_requested": config.eplb_num_physical_experts,
            "num_groups_requested": config.eplb_num_groups,
            "load_source": config.eplb_load_source,
            "estimated_loads": config.eplb_estimated_loads,
            "effective_num_physical_experts": (
                algorithm_results[0].metadata["num_physical_experts"]
                if config.algorithm == "eplb"
                else None
            ),
            "effective_num_groups": (
                algorithm_results[0].metadata["num_groups"]
                if config.algorithm == "eplb"
                else None
            ),
        },
        "dtypes": {
            "dispatch": config.dispatch_dtype,
            "combine": config.combine_dtype,
            "weight": config.weight_dtype,
        },
        "routing_provider": {
            "name": algorithm_metadata[0]["gate"]["name"],
            "random_seed": algorithm_metadata[0]["gate"]["seed"],
            "routing_fidelity": algorithm_metadata[0]["gate"][
                "routing_fidelity"
            ],
            "parameters": algorithm_metadata[0]["gate"]["parameters"],
        },
        "compute_cost": cost.manifest(),
        "stream_schedule": stream_schedule.manifest(),
        "scope": {
            "network": "packet_simulated_flow_completion_dependencies",
            "chunk_pipeline": "explicit_flow_tasks",
            "compute": "fixed_theoretical_duration",
            "communication_sms_per_rank_phase": cost.communication_sms,
            "single_flow_packet_progress_events": False,
            "dynamic_gpu_resource_scheduling": False,
        },
        "micro_batch_algorithms": algorithm_metadata,
    }
    return WorkloadBuildResult(graph, metadata, tuple(algorithm_results))
