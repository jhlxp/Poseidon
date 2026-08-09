from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..load_profile import ExpertInstance, build_expert_load_profile
from ..schema import MoEInvocation, RoutingAssignment, ValidationError
from .common import (
    AlgorithmBuildResult,
    HierarchicalTransferSummary,
    build_hierarchical_combine,
    build_hierarchical_dispatch,
    hierarchical_token_server_pair_profile,
    plan_hierarchical_token_payloads,
)


RouteKey = tuple[int, int, int]


@dataclass(frozen=True)
class MoonEPConfig:
    replicas_per_rank: int
    token_padding: int = 128
    chunk_tokens: int = 128
    overlap_expert_compute: bool = True
    payload_metadata_sample_limit: int = 8

    def __post_init__(self) -> None:
        if self.replicas_per_rank < 0:
            raise ValidationError("replicas_per_rank must be non-negative")
        if self.token_padding <= 0:
            raise ValidationError("token_padding must be positive")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")
        if self.payload_metadata_sample_limit < 0:
            raise ValidationError(
                "payload_metadata_sample_limit must be non-negative"
            )


@dataclass(frozen=True)
class MoonEPPlan:
    execution_rank: dict[RouteKey, int]
    replicas_by_rank: dict[int, tuple[int, ...]]
    target_routes_by_rank: dict[int, int]
    real_routes_by_rank: dict[int, int]
    routes_by_server: dict[int, int]


class MoonEPBuilder:
    def __init__(self, cost_model: ComputeCostModel, config: MoonEPConfig) -> None:
        self.cost_model = cost_model
        self.config = config

    def plan(self, invocation: MoEInvocation) -> MoonEPPlan:
        placement = invocation.placement
        assignments_by_server: dict[int, list[RoutingAssignment]] = {
            server: [] for server in range(placement.num_servers)
        }
        for assignment in invocation.sorted_assignments():
            home_rank = placement.expert_rank(assignment.expert_id)
            assignments_by_server[placement.rank_server(home_rank)].append(assignment)

        execution: dict[RouteKey, int] = {}
        replicas: dict[int, set[int]] = {
            rank: set() for rank in range(placement.num_ranks)
        }
        target_routes: dict[int, int] = {
            rank: 0 for rank in range(placement.num_ranks)
        }
        real_routes: dict[int, int] = {
            rank: 0 for rank in range(placement.num_ranks)
        }
        routes_by_server: dict[int, int] = {}

        for server, server_assignments in assignments_by_server.items():
            ranks = tuple(
                placement.server_rank(server, local_rank)
                for local_rank in range(placement.gpus_per_server)
            )
            total_routes = len(server_assignments)
            routes_by_server[server] = total_routes
            base, extra = divmod(total_routes, placement.gpus_per_server)
            remaining = {}
            for local_rank, rank in enumerate(ranks):
                target = base + (1 if local_rank < extra else 0)
                target_routes[rank] = target
                remaining[rank] = target

            routes_by_expert: dict[int, list[RoutingAssignment]] = defaultdict(list)
            for assignment in server_assignments:
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
                        for rank in ranks
                        if expert in replicas[rank]
                    ),
                }
                while sum(remaining[rank] for rank in candidates) < len(routes):
                    available = [
                        rank
                        for rank in ranks
                        if rank not in candidates
                        and rank != home
                        and remaining[rank] > 0
                        and len(replicas[rank]) < self.config.replicas_per_rank
                    ]
                    if not available:
                        raise ValidationError(
                            "MoonEP per-server planner cannot balance expert "
                            f"{expert} on server {server} with "
                            f"replicas_per_rank={self.config.replicas_per_rank}"
                        )
                    replica_rank = min(
                        available, key=lambda rank: (-remaining[rank], rank)
                    )
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
                    real_routes[rank] += 1

            if any(remaining.values()):
                raise ValidationError(
                    "MoonEP per-server planner left unused rank capacity on "
                    f"server {server}: {remaining}"
                )

        return MoonEPPlan(
            execution_rank=execution,
            replicas_by_rank={
                rank: tuple(sorted(experts)) for rank, experts in replicas.items()
            },
            target_routes_by_rank=target_routes,
            real_routes_by_rank=real_routes,
            routes_by_server=routes_by_server,
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

        planning_keys_by_server: dict[int, set[str]] = defaultdict(set)
        for server in range(placement.num_servers):
            if plan.routes_by_server[server] == 0:
                continue
            planning_flops = (
                plan.routes_by_server[server]
                * max(1, placement.num_experts.bit_length())
                * 2
            )
            for local_rank in range(placement.gpus_per_server):
                rank = placement.server_rank(server, local_rank)
                key = f"{invocation.invocation_id}.plan.server{server}.rank{rank}"
                graph.add_compute(
                    key,
                    rank,
                    self.cost_model.estimate(
                        planning_flops,
                        operation="per_server_planning_proxy",
                        token_count=plan.routes_by_server[server],
                    ),
                    predecessors=roots,
                    barrier_group=f"{invocation.invocation_id}.planning_join.server{server}",
                    metadata={
                        "algorithm": "moonep",
                        "operation": "per_server_planning_proxy",
                        "server": server,
                        "server_route_count": plan.routes_by_server[server],
                        "operation_flops_status": "theoretical_placeholder",
                    },
                )
                planning_keys_by_server[server].add(key)

        prefetch_by_rank: dict[int, set[str]] = defaultdict(set)
        replica_records: list[dict[str, int]] = []
        for dst_rank, experts in sorted(plan.replicas_by_rank.items()):
            dst_server = placement.rank_server(dst_rank)
            for slot, expert in enumerate(experts):
                home = placement.expert_rank(expert)
                if home == dst_rank:
                    continue
                if placement.rank_server(home) != dst_server:
                    raise ValidationError(
                        "MoonEP replica crossed its expert home server"
                    )
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
                    f"{invocation.invocation_id}:prefetch:server:{dst_server}",
                    predecessors=planning_keys_by_server[dst_server],
                    metadata={
                        "expert_id": expert,
                        "prefetch_slot": slot,
                        "home_server": dst_server,
                        "projections": ["gate", "up", "down"],
                        "transport": "server_local_fullmesh",
                    },
                )
                prefetch_by_rank[dst_rank].add(key)
                replica_records.append(
                    {
                        "expert_id": expert,
                        "home_rank": home,
                        "execution_rank": dst_rank,
                        "home_server": dst_server,
                        "prefetch_slot": slot,
                    }
                )

        routes_by_rank_expert: dict[tuple[int, int], list[RoutingAssignment]] = (
            defaultdict(list)
        )
        for assignment in invocation.sorted_assignments():
            execution_rank = plan.execution_rank[self._route_key(assignment)]
            home_rank = placement.expert_rank(assignment.expert_id)
            if placement.rank_server(execution_rank) != placement.rank_server(home_rank):
                raise ValidationError("MoonEP execution rank left expert home server")
            routes_by_rank_expert[(execution_rank, assignment.expert_id)].append(
                assignment
            )

        payload_plan = plan_hierarchical_token_payloads(
            invocation.sorted_assignments(),
            lambda assignment: plan.execution_rank[self._route_key(assignment)],
            placement,
        )

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        transfer_summary = HierarchicalTransferSummary()
        build_hierarchical_dispatch(
            graph,
            invocation,
            payload_plan,
            algorithm="moonep",
            chunk_tokens=self.config.chunk_tokens,
            predecessors_for_server=lambda server: set(
                planning_keys_by_server[server]
            ),
            metadata_sample_limit=self.config.payload_metadata_sample_limit,
            arrivals=dispatch_arrivals,
            summary=transfer_summary,
        )

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
            server = placement.rank_server(rank)
            key = f"{invocation.invocation_id}.expert.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    padded_routes * 6 * invocation.hidden * invocation.ffn_hidden,
                    operation="expert_ffn",
                    overlaps_communication=self.config.overlap_expert_compute,
                    token_count=padded_routes,
                ),
                predecessors=(
                    planning_keys_by_server[server]
                    | prefetch_by_rank[rank]
                    | dispatch_arrivals[rank]
                ),
                metadata={
                    "algorithm": "moonep",
                    "operation": "expert_ffn",
                    "server": server,
                    "real_token_routes": plan.real_routes_by_rank[rank],
                    "padded_token_routes": padded_routes,
                    "expert_route_counts": expert_counts,
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
            algorithm="moonep",
            chunk_tokens=self.config.chunk_tokens,
            metadata_sample_limit=self.config.payload_metadata_sample_limit,
            arrivals=combine_arrivals,
            local_expert_origins=local_expert_origins,
            summary=transfer_summary,
        )

        rank_terminals: dict[int, frozenset[str]] = {}
        terminal_keys: set[str] = set()
        for rank, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            predecessors = set(combine_arrivals[rank])
            if rank in local_expert_origins and rank in expert_keys:
                predecessors.add(expert_keys[rank])
            key = f"{invocation.invocation_id}.combine_reduce.rank{rank}"
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    max(1, token_count * invocation.topk * invocation.hidden * 2),
                    operation="combine_reduce",
                    token_count=token_count,
                ),
                predecessors=predecessors,
                metadata={
                    "algorithm": "moonep",
                    "operation": "combine_reduce",
                    "token_count": token_count,
                },
            )
            terminal_keys.add(key)
            rank_terminals[rank] = frozenset({key})

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind != "transfer":
                continue
            transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
        after_instance_values = [
            ExpertInstance(
                instance_id=f"logical:{expert}:rank:{placement.expert_rank(expert)}",
                logical_expert=expert,
                rank=placement.expert_rank(expert),
                kind="master",
                physical_expert=None,
                replica_index=0,
            )
            for expert in range(placement.num_experts)
        ]
        for rank, experts in sorted(plan.replicas_by_rank.items()):
            for replica_index, expert in enumerate(experts, start=1):
                after_instance_values.append(
                    ExpertInstance(
                        instance_id=f"logical:{expert}:rank:{rank}",
                        logical_expert=expert,
                        rank=rank,
                        kind="replica",
                        physical_expert=None,
                        replica_index=replica_index,
                    )
                )
        expert_load_profile = build_expert_load_profile(
            invocation,
            after_instances=tuple(after_instance_values),
            select_after_instance=lambda assignment: (
                f"logical:{assignment.expert_id}:rank:"
                f"{plan.execution_rank[self._route_key(assignment)]}"
            ),
        )
        return AlgorithmBuildResult(
            algorithm="moonep_deepep_hierarchical",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "planner": "deterministic_per_server_capacity_balancer_v2",
                "scale_out_transport": "deepep_hierarchical",
                "replica_scope": "home_server",
                "replicas_per_rank": self.config.replicas_per_rank,
                "token_padding": self.config.token_padding,
                "chunk_tokens": self.config.chunk_tokens,
                "token_payload_policy": {
                    "deduplicate": True,
                    "scope": "destination_rank_then_server",
                },
                "routes_by_server": plan.routes_by_server,
                "target_routes_by_rank": plan.target_routes_by_rank,
                "real_routes_by_rank": plan.real_routes_by_rank,
                "padded_routes_by_rank": padded_routes_by_rank,
                "replicas": replica_records,
                "route_count": payload_plan.route_count,
                "unique_token_payload_count": payload_plan.rank_payload_count,
                "unique_server_payload_count": payload_plan.server_payload_count,
                "deduplicated_route_count": (
                    payload_plan.route_count - payload_plan.rank_payload_count
                ),
                "scaleout_deduplicated_route_count": (
                    payload_plan.route_count - payload_plan.server_payload_count
                ),
                "server_forward_task_count": 0,
                "hierarchical_transfer": transfer_summary.manifest(),
                "token_server_pair_transport": (
                    hierarchical_token_server_pair_profile(
                        invocation, payload_plan
                    )
                ),
                "expert_weight_prefetch_bytes": transfer_bytes.get(
                    "expert_weight_prefetch", 0
                ),
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
                "expert_load_profile": expert_load_profile,
            },
        )

    @staticmethod
    def _route_key(assignment: RoutingAssignment) -> RouteKey:
        return (
            assignment.src_rank,
            assignment.token_id,
            assignment.topk_slot,
        )
