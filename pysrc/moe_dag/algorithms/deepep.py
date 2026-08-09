from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..load_profile import build_expert_load_profile
from ..schema import MoEInvocation, ValidationError
from .common import (
    AlgorithmBuildResult,
    HierarchicalTransferSummary,
    TokenPayloadPolicy,
    build_hierarchical_combine,
    build_hierarchical_dispatch,
    hierarchical_token_server_pair_profile,
    plan_hierarchical_token_payloads,
)


@dataclass(frozen=True)
class DeepEPConfig:
    chunk_tokens: int = 128
    overlap_expert_compute: bool = True
    payload_metadata_sample_limit: int = 8
    token_payload_policy: TokenPayloadPolicy = field(
        default_factory=TokenPayloadPolicy
    )

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")
        if self.payload_metadata_sample_limit < 0:
            raise ValidationError(
                "payload_metadata_sample_limit must be non-negative"
            )
        if not self.token_payload_policy.deduplicate:
            raise ValidationError("DeepEP requires token payload deduplication")


class DeepEPBuilder:
    def __init__(self, cost_model: ComputeCostModel, config: DeepEPConfig) -> None:
        self.cost_model = cost_model
        self.config = config

    def build(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        *,
        entry_keys: set[str] | None = None,
    ) -> AlgorithmBuildResult:
        if graph.num_ranks != invocation.placement.num_ranks:
            raise ValidationError("graph and placement rank counts differ")
        roots = set(entry_keys or ())
        before = len(graph.tasks)
        placement = invocation.placement

        payload_plan = plan_hierarchical_token_payloads(
            invocation.sorted_assignments(),
            lambda assignment: placement.expert_rank(assignment.expert_id),
            placement,
        )
        route_count_by_rank: dict[int, int] = defaultdict(int)
        route_count_by_origin: dict[int, int] = defaultdict(int)
        for assignment in invocation.sorted_assignments():
            route_count_by_rank[placement.expert_rank(assignment.expert_id)] += 1
            route_count_by_origin[assignment.src_rank] += 1

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        transfer_summary = HierarchicalTransferSummary()
        build_hierarchical_dispatch(
            graph,
            invocation,
            payload_plan,
            algorithm="deepep",
            chunk_tokens=self.config.chunk_tokens,
            predecessors_for_server=lambda _server: set(roots),
            metadata_sample_limit=self.config.payload_metadata_sample_limit,
            arrivals=dispatch_arrivals,
            summary=transfer_summary,
        )

        expert_keys: dict[int, str] = {}
        for rank, route_count in sorted(route_count_by_rank.items()):
            key = f"{invocation.invocation_id}.expert.rank{rank}"
            flops = route_count * 6 * invocation.hidden * invocation.ffn_hidden
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    flops,
                    operation="expert_ffn",
                    overlaps_communication=self.config.overlap_expert_compute,
                    token_count=route_count,
                ),
                predecessors=roots | dispatch_arrivals[rank],
                metadata={
                    "algorithm": "deepep",
                    "operation": "expert_ffn",
                    "real_token_routes": route_count,
                },
            )
            expert_keys[rank] = key

        combine_arrivals: dict[int, set[str]] = defaultdict(set)
        local_expert_origins: set[int] = set()
        build_hierarchical_combine(
            graph,
            invocation,
            payload_plan,
            expert_keys,
            algorithm="deepep",
            chunk_tokens=self.config.chunk_tokens,
            metadata_sample_limit=self.config.payload_metadata_sample_limit,
            arrivals=combine_arrivals,
            local_expert_origins=local_expert_origins,
            summary=transfer_summary,
        )

        rank_terminals: dict[int, frozenset[str]] = {}
        terminal_keys: set[str] = set()
        for origin, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            predecessors = set(combine_arrivals[origin])
            if origin in local_expert_origins and origin in expert_keys:
                predecessors.add(expert_keys[origin])
            key = f"{invocation.invocation_id}.combine_reduce.rank{origin}"
            graph.add_compute(
                key,
                origin,
                self.cost_model.estimate(
                    max(1, route_count_by_origin[origin] * invocation.hidden * 2),
                    operation="combine_final_reduce",
                    token_count=token_count,
                ),
                predecessors=predecessors,
                metadata={
                    "algorithm": "deepep",
                    "operation": "combine_final_reduce",
                    "token_count": token_count,
                    "route_partials": route_count_by_origin[origin],
                },
            )
            rank_terminals[origin] = frozenset({key})
            terminal_keys.add(key)

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind != "transfer":
                continue
            transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
        route_count = payload_plan.route_count
        expert_load_profile = build_expert_load_profile(invocation)
        return AlgorithmBuildResult(
            algorithm="deepep_hierarchical",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "workload_scope": "training_prefill_forward",
                "chunk_tokens": self.config.chunk_tokens,
                "token_payload_policy": {
                    "deduplicate": True,
                    "scope": "destination_rank_then_server",
                },
                "forwarding": {
                    "mode": "hierarchical_scaleout_scaleup",
                    "relay_coordinate": "source_local_index",
                    "dispatch": "fabric_then_local_fanout",
                    "combine": "local_reduce_then_fabric",
                },
                "route_count": route_count,
                "unique_token_payload_count": payload_plan.rank_payload_count,
                "unique_server_payload_count": payload_plan.server_payload_count,
                "deduplicated_route_count": (
                    route_count - payload_plan.rank_payload_count
                ),
                "scaleout_deduplicated_route_count": (
                    route_count - payload_plan.server_payload_count
                ),
                "logical_transfer_task_count": sum(
                    task.kind == "transfer" for task in created
                ),
                "server_forward_task_count": 0,
                "hierarchical_transfer": transfer_summary.manifest(),
                "token_server_pair_transport": (
                    hierarchical_token_server_pair_profile(
                        invocation, payload_plan
                    )
                ),
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
                "expert_load_profile": expert_load_profile,
            },
        )
