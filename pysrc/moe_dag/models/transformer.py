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
    ProbeEPBuilder,
    ProbeEPConfig,
    ProbeNICControllerConfig,
)
from ..cost import ComputeCostModel, H100CostModel
from ..gate import BalancedPermutedGateProvider, GateProvider
from ..graph import TaskGraph
from ..schema import ModelSpec, MoEInvocation, Placement, ValidationError
from .streams import apply_double_buffered_two_stream_schedule


AlgorithmName = Literal["nccl", "deepep", "eplb", "moonep", "probeep"]


@dataclass(frozen=True)
class TransformerWorkloadConfig:
    model: ModelSpec
    placement: Placement
    tokens_per_rank: int
    algorithm: AlgorithmName
    chunk_tokens: int = 128
    replicas_per_rank: int = 0
    token_padding: int = 128
    probeep_route_chunk_tokens: int = 0
    probeep_weight_chunk_bytes: int = 4 * 1024 * 1024
    probeep_initial_nic_budget_bytes: int = 16 * 1024 * 1024
    probeep_nic_line_rate_gbps: float = 400.0
    probeep_target_overlap_ratio: float = 0.90
    probeep_nic_budget_by_dispatch: dict[
        tuple[int, int], tuple[int, ...]
    ] | None = None
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
        if self.algorithm not in {"nccl", "deepep", "eplb", "moonep", "probeep"}:
            raise ValidationError(f"unsupported algorithm: {self.algorithm}")
        if self.probeep_route_chunk_tokens < 0:
            raise ValidationError(
                "probeep_route_chunk_tokens must be non-negative"
            )
        if self.probeep_weight_chunk_bytes <= 0:
            raise ValidationError("probeep_weight_chunk_bytes must be positive")
        if self.probeep_nic_budget_by_dispatch is not None:
            for scope, budgets in self.probeep_nic_budget_by_dispatch.items():
                if (
                    len(scope) != 2
                    or scope[0] < 0
                    or scope[0] >= self.model.micro_batches
                    or scope[1] < 0
                    or scope[1] >= self.model.num_layers
                ):
                    raise ValidationError(
                        "ProbeEP Dispatch budget override has an invalid scope"
                    )
                if len(budgets) != self.placement.num_ranks:
                    raise ValidationError(
                        "ProbeEP budget override length must equal num_ranks"
                    )
                if any(value < 0 for value in budgets):
                    raise ValidationError(
                        "ProbeEP budget overrides must be non-negative"
                    )
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
    elif config.algorithm == "probeep":
        algorithm_builder = ProbeEPBuilder(
            cost,
            ProbeEPConfig(
                token_padding=config.token_padding,
                chunk_tokens=config.chunk_tokens,
                route_chunk_tokens=(
                    config.probeep_route_chunk_tokens or config.chunk_tokens
                ),
                weight_chunk_bytes=config.probeep_weight_chunk_bytes,
                nic_controller=ProbeNICControllerConfig(
                    initial_budget_bytes=(
                        config.probeep_initial_nic_budget_bytes
                    ),
                    nic_line_rate_gbps=(
                        config.probeep_nic_line_rate_gbps
                    ),
                    target_overlap_ratio=(
                        config.probeep_target_overlap_ratio
                    ),
                ),
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
    probeep_attention_reference = tuple(
        cost.estimate(
            _attention_flops(config.tokens_per_rank, config.model),
            operation="attention",
            overlaps_communication=True,
            token_count=config.tokens_per_rank,
        ).duration_us
        for _ in range(config.placement.num_ranks)
    )
    probeep_moe_reference: dict[tuple[int, int], tuple[float, ...]] = {}
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
            if isinstance(algorithm_builder, ProbeEPBuilder):
                pair_start = micro_batch - micro_batch % 2
                # MB0 Weight+Dispatch overlaps MB1 Attention; MB1
                # Weight+Dispatch overlaps MB0 Expert FFN. Combine is
                # telemetry only and never enters ProbeEP admission control.
                dispatch_overlap_kind = (
                    "attention" if micro_batch % 2 == 0 else "moe"
                )
                moe_reference = probeep_moe_reference.get(
                    (pair_start, layer)
                )
                algorithm_result = algorithm_builder.build(
                    graph,
                    invocation,
                    entry_keys=router_keys,
                    dispatch_overlap_compute_kind=dispatch_overlap_kind,
                    attention_compute_us_by_rank=(
                        probeep_attention_reference
                    ),
                    moe_compute_us_by_rank=moe_reference,
                    nic_budget_override=(
                        config.probeep_nic_budget_by_dispatch.get(
                            (micro_batch, layer)
                        )
                        if config.probeep_nic_budget_by_dispatch is not None
                        else None
                    ),
                )
                if micro_batch % 2 == 0:
                    final_compute = algorithm_result.metadata[
                        "final_compute_us_by_rank"
                    ]
                    probeep_moe_reference[(pair_start, layer)] = tuple(
                        float(final_compute[rank])
                        for rank in range(config.placement.num_ranks)
                    )
            else:
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
        "token_padding": config.token_padding,
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
    if config.algorithm == "moonep":
        metadata["replicas_per_rank"] = config.replicas_per_rank
    if config.algorithm == "probeep":
        metadata["probeep"] = {
            "route_chunk_tokens": (
                config.probeep_route_chunk_tokens or config.chunk_tokens
            ),
            "weight_chunk_bytes": config.probeep_weight_chunk_bytes,
            "initial_nic_budget_bytes": (
                config.probeep_initial_nic_budget_bytes
            ),
            "nic_line_rate_gbps": config.probeep_nic_line_rate_gbps,
            "target_overlap_ratio": (
                config.probeep_target_overlap_ratio
            ),
            "budget_override_dispatches": (
                sorted(
                    f"mb{scope[0]}.layer{scope[1]}"
                    for scope in config.probeep_nic_budget_by_dispatch
                )
                if config.probeep_nic_budget_by_dispatch is not None
                else []
            ),
        }
    if config.algorithm == "eplb":
        metadata["eplb"] = {
            "num_physical_experts_requested": config.eplb_num_physical_experts,
            "num_groups_requested": config.eplb_num_groups,
            "load_source": config.eplb_load_source,
            "estimated_loads": config.eplb_estimated_loads,
            "effective_num_physical_experts": algorithm_results[0].metadata[
                "num_physical_experts"
            ],
            "effective_num_groups": algorithm_results[0].metadata["num_groups"],
        }
    return WorkloadBuildResult(graph, metadata, tuple(algorithm_results))
