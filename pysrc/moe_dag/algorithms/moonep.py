from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from ..cost import H100CostModel
from ..graph import TaskGraph
from ..schema import MoEInvocation, RoutingAssignment, ValidationError
from .common import AlgorithmBuildResult, chunked


RouteKey = tuple[int, int, int]
Token = tuple[int, int]


@dataclass(frozen=True)
class MoonEPConfig:
    replicas_per_rank: int
    token_padding: int = 128
    chunk_tokens: int = 128
    overlap_expert_compute: bool = True

    def __post_init__(self) -> None:
        if self.replicas_per_rank < 0:
            raise ValidationError("replicas_per_rank must be non-negative")
        if self.token_padding <= 0:
            raise ValidationError("token_padding must be positive")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")


@dataclass(frozen=True)
class MoonEPPlan:
    execution_rank: dict[RouteKey, int]
    replicas_by_rank: dict[int, tuple[int, ...]]
    real_routes_by_rank: dict[int, int]


class MoonEPBuilder:
    def __init__(self, cost_model: H100CostModel, config: MoonEPConfig) -> None:
        self.cost_model = cost_model
        self.config = config

    def plan(self, invocation: MoEInvocation) -> MoonEPPlan:
        placement = invocation.placement
        if placement.num_servers != 1:
            raise ValidationError("MoonEP v1 model requires a single server")
        total_routes = len(invocation.assignments)
        if total_routes % placement.num_ranks != 0:
            raise ValidationError(
                "MoonEP balanced plan requires total routes divisible by num_ranks"
            )
        target = total_routes // placement.num_ranks
        remaining = {rank: target for rank in range(placement.num_ranks)}
        replicas: dict[int, set[int]] = {
            rank: set() for rank in range(placement.num_ranks)
        }
        execution: dict[RouteKey, int] = {}

        routes_by_expert: dict[int, list[RoutingAssignment]] = defaultdict(list)
        for assignment in invocation.sorted_assignments():
            routes_by_expert[assignment.expert_id].append(assignment)
        ordered_experts = sorted(
            routes_by_expert,
            key=lambda expert: (-len(routes_by_expert[expert]), expert),
        )

        for expert in ordered_experts:
            routes = routes_by_expert[expert]
            home = placement.expert_rank(expert)
            candidates = {
                home,
                *(
                    rank
                    for rank, rank_replicas in replicas.items()
                    if expert in rank_replicas
                ),
            }
            while sum(remaining[rank] for rank in candidates) < len(routes):
                available = [
                    rank
                    for rank in range(placement.num_ranks)
                    if rank not in candidates
                    and rank != home
                    and remaining[rank] > 0
                    and len(replicas[rank]) < self.config.replicas_per_rank
                ]
                if not available:
                    raise ValidationError(
                        "MoonEP deterministic planner cannot balance routes with "
                        f"replicas_per_rank={self.config.replicas_per_rank}"
                    )
                replica_rank = min(available, key=lambda rank: (-remaining[rank], rank))
                replicas[replica_rank].add(expert)
                candidates.add(replica_rank)

            for assignment in routes:
                rank = min(
                    (item for item in candidates if remaining[item] > 0),
                    key=lambda item: (-remaining[item], item),
                )
                execution[
                    (
                        assignment.src_rank,
                        assignment.token_id,
                        assignment.topk_slot,
                    )
                ] = rank
                remaining[rank] -= 1

        if any(remaining.values()):
            raise ValidationError(
                "MoonEP deterministic planner left unused rank capacity: "
                f"{remaining}"
            )
        return MoonEPPlan(
            execution_rank=execution,
            replicas_by_rank={
                rank: tuple(sorted(experts)) for rank, experts in replicas.items()
            },
            real_routes_by_rank={rank: target for rank in range(placement.num_ranks)},
        )

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
        plan = self.plan(invocation)
        placement = invocation.placement

        planning_keys: set[str] = set()
        planning_flops = max(
            1,
            len(invocation.assignments)
            * max(1, placement.num_experts.bit_length())
            * 2,
        )
        for rank in range(placement.num_ranks):
            key = f"{invocation.invocation_id}.plan.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(planning_flops),
                predecessors=roots,
                barrier_group=f"{invocation.invocation_id}.planning_join",
                metadata={
                    "algorithm": "moonep",
                    "operation": "planning_proxy",
                    "cost_status": "theoretical_placeholder",
                },
            )
            planning_keys.add(key)

        prefetch_by_rank: dict[int, set[str]] = defaultdict(set)
        replica_records: list[dict[str, int]] = []
        for dst_rank, experts in sorted(plan.replicas_by_rank.items()):
            for slot, expert in enumerate(experts):
                home = placement.expert_rank(expert)
                if home == dst_rank:
                    continue
                key = (
                    f"{invocation.invocation_id}.prefetch.expert{expert}."
                    f"src{home}.dst{dst_rank}.slot{slot}"
                )
                graph.add_transfer(
                    key,
                    home,
                    dst_rank,
                    invocation.expert_weight_bytes,
                    "expert_weight_prefetch",
                    f"{invocation.invocation_id}:prefetch:rank:{dst_rank}",
                    predecessors=planning_keys,
                    metadata={
                        "expert_id": expert,
                        "prefetch_slot": slot,
                        "projections": ["gate", "up", "down"],
                    },
                )
                prefetch_by_rank[dst_rank].add(key)
                replica_records.append(
                    {
                        "expert_id": expert,
                        "home_rank": home,
                        "execution_rank": dst_rank,
                        "prefetch_slot": slot,
                    }
                )

        tokens_by_src_execution: dict[tuple[int, int], set[Token]] = defaultdict(set)
        routes_by_rank_expert: dict[tuple[int, int], list[RoutingAssignment]] = (
            defaultdict(list)
        )
        for assignment in invocation.sorted_assignments():
            route_key = (
                assignment.src_rank,
                assignment.token_id,
                assignment.topk_slot,
            )
            execution_rank = plan.execution_rank[route_key]
            tokens_by_src_execution[(assignment.src_rank, execution_rank)].add(
                (assignment.src_rank, assignment.token_id)
            )
            routes_by_rank_expert[(execution_rank, assignment.expert_id)].append(
                assignment
            )

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        for (src, dst), tokens in sorted(tokens_by_src_execution.items()):
            if src == dst:
                continue
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.dispatch."
                    f"src{src}.dst{dst}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    src,
                    dst,
                    len(token_chunk) * invocation.dispatch_token_bytes,
                    "dispatch_primary_row",
                    f"{invocation.invocation_id}:dispatch:rank:{src}",
                    predecessors=planning_keys,
                    chunk_id=chunk_id,
                    metadata={
                        "tokens": list(token_chunk),
                        "deduplicated_by_destination_rank": True,
                    },
                )
                dispatch_arrivals[dst].add(key)

        expert_keys: dict[int, str] = {}
        padded_routes_by_rank: dict[int, int] = {}
        for rank in range(placement.num_ranks):
            expert_counts = {
                expert: len(routes)
                for (item_rank, expert), routes in routes_by_rank_expert.items()
                if item_rank == rank and routes
            }
            padded_routes = sum(
                ceil(count / self.config.token_padding) * self.config.token_padding
                for count in expert_counts.values()
            )
            if padded_routes == 0:
                continue
            padded_routes_by_rank[rank] = padded_routes
            key = f"{invocation.invocation_id}.expert.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    padded_routes * 6 * invocation.hidden * invocation.ffn_hidden,
                    overlaps_communication=self.config.overlap_expert_compute,
                ),
                predecessors=(
                    planning_keys | prefetch_by_rank[rank] | dispatch_arrivals[rank]
                ),
                metadata={
                    "algorithm": "moonep",
                    "operation": "expert_ffn",
                    "real_token_routes": plan.real_routes_by_rank[rank],
                    "padded_token_routes": padded_routes,
                    "expert_route_counts": expert_counts,
                },
            )
            expert_keys[rank] = key

        combine_arrivals: dict[int, set[str]] = defaultdict(set)
        for (origin, execution_rank), tokens in sorted(
            tokens_by_src_execution.items()
        ):
            if origin == execution_rank:
                continue
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.combine."
                    f"src{execution_rank}.dst{origin}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    execution_rank,
                    origin,
                    len(token_chunk) * invocation.combine_token_bytes,
                    "combine_primary_row",
                    f"{invocation.invocation_id}:combine:rank:{origin}",
                    predecessors={expert_keys[execution_rank]},
                    chunk_id=chunk_id,
                    metadata={"tokens": list(token_chunk)},
                )
                combine_arrivals[origin].add(key)

        rank_terminals: dict[int, frozenset[str]] = {}
        terminal_keys: set[str] = set()
        for rank, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            local_execution = {
                execution_rank
                for (src, execution_rank), tokens in tokens_by_src_execution.items()
                if src == rank and execution_rank == rank and tokens
            }
            predecessors = combine_arrivals[rank] | {
                expert_keys[item]
                for item in local_execution
                if item in expert_keys
            }
            key = f"{invocation.invocation_id}.combine_reduce.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    max(1, token_count * invocation.topk * invocation.hidden * 2)
                ),
                predecessors=predecessors,
                metadata={
                    "algorithm": "moonep",
                    "operation": "combine_local_reduce",
                    "token_count": token_count,
                },
            )
            terminal_keys.add(key)
            rank_terminals[rank] = frozenset({key})

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind == "transfer":
                transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
        return AlgorithmBuildResult(
            algorithm="moonep_forward",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "planner": "deterministic_capacity_balancer_v1",
                "replicas_per_rank": self.config.replicas_per_rank,
                "token_padding": self.config.token_padding,
                "chunk_tokens": self.config.chunk_tokens,
                "real_routes_by_rank": plan.real_routes_by_rank,
                "padded_routes_by_rank": padded_routes_by_rank,
                "replicas": replica_records,
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
            },
        )
