from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..load_profile import build_expert_load_profile
from ..schema import MoEInvocation, ValidationError
from .common import (
    AlgorithmBuildResult,
    TokenPayload,
    TokenPayloadPolicy,
    chunked,
    plan_token_payloads,
)


@dataclass(frozen=True)
class NCCLConfig:
    chunk_routes: int = 128
    overlap_expert_compute: bool = True
    payload_metadata_sample_limit: int = 8
    token_payload_policy: TokenPayloadPolicy = field(
        default_factory=lambda: TokenPayloadPolicy(
            deduplicate=False, scope="none"
        )
    )

    def __post_init__(self) -> None:
        if self.chunk_routes <= 0:
            raise ValidationError("chunk_routes must be positive")
        if self.payload_metadata_sample_limit < 0:
            raise ValidationError(
                "payload_metadata_sample_limit must be non-negative"
            )
        if self.token_payload_policy.deduplicate:
            raise ValidationError("NCCL requires non-deduplicated token payloads")


class NCCLBuilder:
    def __init__(self, cost_model: ComputeCostModel, config: NCCLConfig) -> None:
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

        payloads_by_pair = plan_token_payloads(
            invocation.sorted_assignments(),
            lambda assignment: placement.expert_rank(assignment.expert_id),
            self.config.token_payload_policy,
        )
        route_count_by_rank: dict[int, int] = defaultdict(int)
        route_count_by_origin: dict[int, int] = defaultdict(int)
        for assignment in invocation.sorted_assignments():
            route_count_by_rank[placement.expert_rank(assignment.expert_id)] += 1
            route_count_by_origin[assignment.src_rank] += 1

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        self._build_dispatch(
            graph,
            invocation,
            roots,
            payloads_by_pair,
            dispatch_arrivals,
        )

        expert_keys: dict[int, str] = {}
        for rank, route_count in sorted(route_count_by_rank.items()):
            key = f"{invocation.invocation_id}.expert.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    route_count * 6 * invocation.hidden * invocation.ffn_hidden,
                    operation="expert_ffn",
                    overlaps_communication=self.config.overlap_expert_compute,
                    token_count=route_count,
                ),
                predecessors=roots | dispatch_arrivals[rank],
                metadata={
                    "algorithm": "nccl",
                    "operation": "expert_ffn",
                    "real_token_routes": route_count,
                },
            )
            expert_keys[rank] = key

        combine_arrivals: dict[int, set[str]] = defaultdict(set)
        self._build_combine(
            graph,
            invocation,
            payloads_by_pair,
            expert_keys,
            combine_arrivals,
        )

        terminal_keys: set[str] = set()
        rank_terminals: dict[int, frozenset[str]] = {}
        for origin, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            predecessors = set(combine_arrivals[origin])
            if (origin, origin) in payloads_by_pair and origin in expert_keys:
                predecessors.add(expert_keys[origin])
            key = f"{invocation.invocation_id}.combine_reduce.rank{origin}"
            graph.add_compute(
                key,
                origin,
                self.cost_model.estimate(
                    max(1, route_count_by_origin[origin] * invocation.hidden * 2),
                    operation="combine_reduce",
                    token_count=token_count,
                ),
                predecessors=predecessors,
                metadata={
                    "algorithm": "nccl",
                    "operation": "combine_reduce",
                    "token_count": token_count,
                    "route_partials": route_count_by_origin[origin],
                },
            )
            terminal_keys.add(key)
            rank_terminals[origin] = frozenset({key})

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind == "transfer":
                transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
        route_count = len(invocation.assignments)
        payload_count = sum(len(items) for items in payloads_by_pair.values())
        expert_load_profile = build_expert_load_profile(invocation)
        return AlgorithmBuildResult(
            algorithm="nccl_alltoall_forward",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "collective_semantics": "alltoallv",
                "chunk_routes": self.config.chunk_routes,
                "token_payload_policy": {
                    "deduplicate": False,
                    "scope": "none",
                },
                "hierarchical_forwarding": False,
                "route_count": route_count,
                "token_payload_count": payload_count,
                "logical_transfer_task_count": sum(
                    task.kind == "transfer" for task in created
                ),
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
                "expert_load_profile": expert_load_profile,
            },
        )

    def _build_dispatch(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        roots: set[str],
        payloads_by_pair: dict[tuple[int, int], tuple[TokenPayload, ...]],
        arrivals: dict[int, set[str]],
    ) -> None:
        for (src, dst), payloads in payloads_by_pair.items():
            if src == dst:
                continue
            for chunk_id, payload_chunk in enumerate(
                chunked(payloads, self.config.chunk_routes)
            ):
                key = (
                    f"{invocation.invocation_id}.dispatch."
                    f"src{src}.dst{dst}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    src,
                    dst,
                    len(payload_chunk) * invocation.dispatch_token_bytes,
                    "dispatch_hidden",
                    f"{invocation.invocation_id}:dispatch:rank:{src}",
                    predecessors=roots,
                    chunk_id=chunk_id,
                    metadata=self._payload_metadata(payload_chunk),
                )
                arrivals[dst].add(key)

    def _build_combine(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        payloads_by_pair: dict[tuple[int, int], tuple[TokenPayload, ...]],
        expert_keys: dict[int, str],
        arrivals: dict[int, set[str]],
    ) -> None:
        for (origin, execution_rank), payloads in payloads_by_pair.items():
            if origin == execution_rank:
                continue
            for chunk_id, payload_chunk in enumerate(
                chunked(payloads, self.config.chunk_routes)
            ):
                key = (
                    f"{invocation.invocation_id}.combine."
                    f"src{execution_rank}.dst{origin}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    execution_rank,
                    origin,
                    len(payload_chunk) * invocation.combine_token_bytes,
                    "combine_partial",
                    f"{invocation.invocation_id}:combine:rank:{origin}",
                    predecessors={expert_keys[execution_rank]},
                    chunk_id=chunk_id,
                    metadata=self._payload_metadata(payload_chunk),
                )
                arrivals[origin].add(key)

    def _payload_metadata(
        self,
        payloads: tuple[TokenPayload, ...],
    ) -> dict[str, object]:
        sample = payloads[: self.config.payload_metadata_sample_limit]
        return {
            "payloads": [
                {
                    "src_rank": payload.src_rank,
                    "token_id": payload.token_id,
                    "topk_slot": payload.routes[0].topk_slot,
                    "expert_id": payload.routes[0].expert_id,
                }
                for payload in sample
            ],
            "payload_count": len(payloads),
            "payload_sample_count": len(sample),
            "payload_metadata_truncated": len(sample) < len(payloads),
            "deduplicated": False,
            "forwarding": "rank_direct",
        }
