from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..schema import MoEInvocation, ValidationError
from .common import (
    AlgorithmBuildResult,
    TokenPayload,
    TokenPayloadPolicy,
    chunked,
    destination_forward_route,
    plan_token_payloads,
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
            flops = route_count * 6 * invocation.hidden * invocation.ffn_hidden
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    flops,
                    operation="expert_ffn",
                    overlaps_communication=self.config.overlap_expert_compute,
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
        self._build_combine(
            graph,
            invocation,
            payloads_by_pair,
            expert_keys,
            combine_arrivals,
        )

        rank_terminals: dict[int, frozenset[str]] = {}
        terminal_keys: set[str] = set()
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
                    operation="combine_final_reduce",
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
        server_forward_tasks = 0
        for task in created:
            if task.kind != "transfer":
                continue
            transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
            if task.route_spec and task.route_spec.startswith("server_forward "):
                server_forward_tasks += 1
        payload_count = sum(len(items) for items in payloads_by_pair.values())
        route_count = len(invocation.assignments)
        return AlgorithmBuildResult(
            algorithm="deepep_forward",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "workload_scope": "training_prefill_forward",
                "chunk_tokens": self.config.chunk_tokens,
                "token_payload_policy": {
                    "deduplicate": True,
                    "scope": "destination_rank",
                },
                "forwarding": {
                    "mode": "destination",
                    "relay_coordinate": "source_local_index",
                    "completion": "full_message",
                },
                "route_count": route_count,
                "unique_token_payload_count": payload_count,
                "deduplicated_route_count": route_count - payload_count,
                "logical_transfer_task_count": sum(
                    task.kind == "transfer" for task in created
                ),
                "server_forward_task_count": server_forward_tasks,
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
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
                chunked(payloads, self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.dispatch."
                    f"src{src}.dst{dst}.chunk{chunk_id}"
                )
                route_spec, relay = destination_forward_route(
                    invocation.placement, src, dst
                )
                graph.add_transfer(
                    key,
                    src,
                    dst,
                    len(payload_chunk) * invocation.dispatch_token_bytes,
                    "dispatch_hidden",
                    f"{invocation.invocation_id}:dispatch:rank:{src}",
                    predecessors=roots,
                    route_spec=route_spec,
                    chunk_id=chunk_id,
                    metadata=self._transfer_metadata(
                        payload_chunk, relay, route_spec
                    ),
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
                chunked(payloads, self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.combine."
                    f"src{execution_rank}.dst{origin}.chunk{chunk_id}"
                )
                route_spec, relay = destination_forward_route(
                    invocation.placement, execution_rank, origin
                )
                graph.add_transfer(
                    key,
                    execution_rank,
                    origin,
                    len(payload_chunk) * invocation.combine_token_bytes,
                    "combine_partial",
                    f"{invocation.invocation_id}:combine:rank:{origin}",
                    predecessors={expert_keys[execution_rank]},
                    route_spec=route_spec,
                    chunk_id=chunk_id,
                    metadata=self._transfer_metadata(
                        payload_chunk, relay, route_spec
                    ),
                )
                arrivals[origin].add(key)

    def _transfer_metadata(
        self,
        payloads: tuple[TokenPayload, ...],
        relay: int | None,
        route_spec: str | None,
    ) -> dict[str, object]:
        sample = payloads[: self.config.payload_metadata_sample_limit]
        return {
            "payloads": [
                {
                    "src_rank": payload.src_rank,
                    "token_id": payload.token_id,
                    "route_count": len(payload.routes),
                    "topk_slots": [route.topk_slot for route in payload.routes],
                    "expert_ids": [route.expert_id for route in payload.routes],
                }
                for payload in sample
            ],
            "payload_count": len(payloads),
            "payload_sample_count": len(sample),
            "payload_metadata_truncated": len(sample) < len(payloads),
            "deduplicated_by_destination_rank": True,
            "forwarding": "destination" if route_spec else "local",
            "dst_relay": relay,
        }
