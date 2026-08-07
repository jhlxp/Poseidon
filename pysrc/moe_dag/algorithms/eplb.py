from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from ..cost import H100CostModel
from ..graph import TaskGraph
from ..schema import MoEInvocation, RoutingAssignment, ValidationError
from .common import (
    AlgorithmBuildResult,
    TokenPayload,
    TokenPayloadPolicy,
    chunked,
    destination_forward_route,
    plan_token_payloads,
)


RouteKey = tuple[int, int, int]


def _balanced_packing(
    weights: tuple[float, ...], num_packs: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if num_packs <= 0 or len(weights) % num_packs != 0:
        raise ValidationError("EPLB balanced packing requires equal pack sizes")
    items_per_pack = len(weights) // num_packs
    if items_per_pack == 1:
        return tuple(range(len(weights))), (0,) * len(weights)

    pack_index = [-1] * len(weights)
    rank_in_pack = [-1] * len(weights)
    pack_weights = [0.0] * num_packs
    pack_items = [0] * num_packs
    for item in sorted(range(len(weights)), key=lambda index: (-weights[index], index)):
        pack = min(
            (index for index in range(num_packs)
             if pack_items[index] < items_per_pack),
            key=lambda index: (pack_weights[index], index),
        )
        pack_index[item] = pack
        rank_in_pack[item] = pack_items[pack]
        pack_weights[pack] += weights[item]
        pack_items[pack] += 1
    return tuple(pack_index), tuple(rank_in_pack)


def _replicate_experts(
    weights: tuple[float, ...], num_physical_experts: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    num_logical_experts = len(weights)
    if num_physical_experts < num_logical_experts:
        raise ValidationError(
            "EPLB physical expert count must cover every logical expert"
        )
    physical_to_logical = list(range(num_logical_experts))
    physical_replica_rank = [0] * num_logical_experts
    logical_count = [1] * num_logical_experts
    for _ in range(num_logical_experts, num_physical_experts):
        expert = max(
            range(num_logical_experts),
            key=lambda index: (weights[index] / logical_count[index], -index),
        )
        physical_to_logical.append(expert)
        physical_replica_rank.append(logical_count[expert])
        logical_count[expert] += 1
    return (
        tuple(physical_to_logical),
        tuple(physical_replica_rank),
        tuple(logical_count),
    )


def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [-1] * len(permutation)
    for source, destination in enumerate(permutation):
        inverse[destination] = source
    return tuple(inverse)


@dataclass(frozen=True)
class EPLBPlacementPlan:
    estimated_loads: tuple[float, ...]
    physical_to_logical: tuple[int, ...]
    physical_replica_rank: tuple[int, ...]
    physical_to_rank: tuple[int, ...]
    logical_to_physical: tuple[tuple[int, ...], ...]
    logical_count: tuple[int, ...]
    physical_experts_per_rank: int


def plan_hierarchical_placement(
    estimated_loads: tuple[float, ...],
    *,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> EPLBPlacementPlan:
    num_logical_experts = len(estimated_loads)
    if num_logical_experts == 0:
        raise ValidationError("EPLB requires at least one logical expert")
    if any(not isfinite(load) or load < 0 for load in estimated_loads):
        raise ValidationError("EPLB estimated loads must be finite and non-negative")
    if num_groups <= 0 or num_nodes <= 0 or num_gpus <= 0:
        raise ValidationError("EPLB group, node, and GPU counts must be positive")
    if num_logical_experts % num_groups != 0:
        raise ValidationError("EPLB logical experts must be divisible by groups")
    if num_groups % num_nodes != 0:
        raise ValidationError(
            "training/prefill EPLB requires groups divisible by nodes"
        )
    if num_gpus % num_nodes != 0:
        raise ValidationError("EPLB GPUs must be divisible by nodes")
    if num_physical_experts < num_logical_experts:
        raise ValidationError(
            "EPLB physical experts must not be fewer than logical experts"
        )
    if num_physical_experts % num_gpus != 0:
        raise ValidationError("EPLB physical experts must be divisible by GPUs")
    if num_physical_experts % num_nodes != 0:
        raise ValidationError("EPLB physical experts must be divisible by nodes")

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    physical_per_node = num_physical_experts // num_nodes
    physical_per_gpu = num_physical_experts // num_gpus
    logical_per_node = num_logical_experts // num_nodes
    if physical_per_node < logical_per_node:
        raise ValidationError("EPLB node lacks physical slots for logical experts")

    group_loads = tuple(
        sum(estimated_loads[group * group_size:(group + 1) * group_size])
        for group in range(num_groups)
    )
    group_pack, group_rank = _balanced_packing(group_loads, num_nodes)
    logical_to_moved = tuple(
        (
            group_pack[expert // group_size] * groups_per_node
            + group_rank[expert // group_size]
        )
        * group_size
        + expert % group_size
        for expert in range(num_logical_experts)
    )
    moved_to_logical = _inverse_permutation(logical_to_moved)
    moved_loads = tuple(estimated_loads[index] for index in moved_to_logical)

    packed_physical_to_logical: list[int] = []
    packed_replica_rank: list[int] = []
    moved_counts: list[int] = []
    for node in range(num_nodes):
        begin = node * logical_per_node
        node_loads = moved_loads[begin:begin + logical_per_node]
        phy_to_moved, replica_rank, moved_count = _replicate_experts(
            node_loads, physical_per_node
        )
        replica_loads = tuple(
            node_loads[moved] / moved_count[moved]
            for moved in phy_to_moved
        )
        pack_index, rank_in_pack = _balanced_packing(
            replica_loads, gpus_per_node
        )
        physical_to_packed = tuple(
            pack_index[index] * physical_per_gpu + rank_in_pack[index]
            for index in range(physical_per_node)
        )
        packed_to_physical = _inverse_permutation(physical_to_packed)
        for packed_index in range(physical_per_node):
            physical_index = packed_to_physical[packed_index]
            moved = begin + phy_to_moved[physical_index]
            packed_physical_to_logical.append(moved_to_logical[moved])
            packed_replica_rank.append(replica_rank[physical_index])
        moved_counts.extend(moved_count)

    logical_count = tuple(moved_counts[logical_to_moved[expert]]
                          for expert in range(num_logical_experts))
    logical_to_physical_lists: list[list[int]] = [
        [-1] * logical_count[expert] for expert in range(num_logical_experts)
    ]
    for physical, (logical, replica_rank) in enumerate(
        zip(packed_physical_to_logical, packed_replica_rank)
    ):
        logical_to_physical_lists[logical][replica_rank] = physical
    if any(-1 in replicas for replicas in logical_to_physical_lists):
        raise ValidationError("EPLB generated an incomplete inverse placement map")

    return EPLBPlacementPlan(
        estimated_loads=estimated_loads,
        physical_to_logical=tuple(packed_physical_to_logical),
        physical_replica_rank=tuple(packed_replica_rank),
        physical_to_rank=tuple(
            physical // physical_per_gpu
            for physical in range(num_physical_experts)
        ),
        logical_to_physical=tuple(
            tuple(replicas) for replicas in logical_to_physical_lists
        ),
        logical_count=logical_count,
        physical_experts_per_rank=physical_per_gpu,
    )


@dataclass(frozen=True)
class EPLBConfig:
    num_physical_experts: int
    num_groups: int
    chunk_tokens: int = 128
    estimated_loads: tuple[float, ...] | None = None
    load_source: str = "current_invocation_proxy"
    overlap_expert_compute: bool = True

    def __post_init__(self) -> None:
        if self.num_physical_experts <= 0:
            raise ValidationError("EPLB physical expert count must be positive")
        if self.num_groups <= 0:
            raise ValidationError("EPLB group count must be positive")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")
        if not self.load_source:
            raise ValidationError("EPLB load source must not be empty")
        if self.estimated_loads is not None and any(
            not isfinite(load) or load < 0 for load in self.estimated_loads
        ):
            raise ValidationError(
                "EPLB estimated loads must be finite and non-negative"
            )


@dataclass(frozen=True)
class EPLBExecution:
    logical_expert: int
    physical_expert: int
    rank: int


class EPLBBuilder:
    def __init__(self, cost_model: H100CostModel, config: EPLBConfig) -> None:
        self.cost_model = cost_model
        self.config = config
        self._placement_plan: EPLBPlacementPlan | None = None

    def plan_placement(self, invocation: MoEInvocation) -> EPLBPlacementPlan:
        if self._placement_plan is not None:
            return self._placement_plan
        if self.config.estimated_loads is None:
            loads = [0.0] * invocation.placement.num_experts
            for assignment in invocation.sorted_assignments():
                loads[assignment.expert_id] += 1
            estimated_loads = tuple(loads)
        else:
            estimated_loads = self.config.estimated_loads
        if len(estimated_loads) != invocation.placement.num_experts:
            raise ValidationError(
                "EPLB estimated load count must equal logical expert count"
            )
        self._placement_plan = plan_hierarchical_placement(
            estimated_loads,
            num_physical_experts=self.config.num_physical_experts,
            num_groups=self.config.num_groups,
            num_nodes=invocation.placement.num_servers,
            num_gpus=invocation.placement.num_ranks,
        )
        return self._placement_plan

    def plan_execution(
        self,
        invocation: MoEInvocation,
        placement_plan: EPLBPlacementPlan,
    ) -> dict[RouteKey, EPLBExecution]:
        counters = [0] * invocation.placement.num_experts
        execution: dict[RouteKey, EPLBExecution] = {}
        for assignment in invocation.sorted_assignments():
            expert = assignment.expert_id
            replicas = placement_plan.logical_to_physical[expert]
            physical = replicas[counters[expert] % len(replicas)]
            counters[expert] += 1
            execution[self._route_key(assignment)] = EPLBExecution(
                logical_expert=expert,
                physical_expert=physical,
                rank=placement_plan.physical_to_rank[physical],
            )
        return execution

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
        placement_plan = self.plan_placement(invocation)
        execution = self.plan_execution(invocation, placement_plan)

        payloads_by_pair = plan_token_payloads(
            invocation.sorted_assignments(),
            lambda assignment: execution[self._route_key(assignment)].rank,
            TokenPayloadPolicy(),
        )
        route_count_by_rank: dict[int, int] = defaultdict(int)
        route_count_by_origin: dict[int, int] = defaultdict(int)
        route_count_by_physical: dict[int, int] = defaultdict(int)
        for assignment in invocation.sorted_assignments():
            selected = execution[self._route_key(assignment)]
            route_count_by_rank[selected.rank] += 1
            route_count_by_physical[selected.physical_expert] += 1
            route_count_by_origin[assignment.src_rank] += 1

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        self._build_dispatch(
            graph,
            invocation,
            roots,
            payloads_by_pair,
            execution,
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
                    overlaps_communication=self.config.overlap_expert_compute,
                ),
                predecessors=roots | dispatch_arrivals[rank],
                metadata={
                    "algorithm": "eplb",
                    "operation": "expert_ffn",
                    "real_token_routes": route_count,
                    "physical_expert_routes": {
                        physical: count
                        for physical, count in sorted(route_count_by_physical.items())
                        if placement_plan.physical_to_rank[physical] == rank
                    },
                },
            )
            expert_keys[rank] = key

        combine_arrivals: dict[int, set[str]] = defaultdict(set)
        self._build_combine(
            graph,
            invocation,
            payloads_by_pair,
            execution,
            expert_keys,
            combine_arrivals,
        )

        terminal_keys: set[str] = set()
        rank_terminals: dict[int, frozenset[str]] = {}
        local_execution_pairs = set(payloads_by_pair)
        for origin, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            predecessors = set(combine_arrivals[origin])
            if (origin, origin) in local_execution_pairs and origin in expert_keys:
                predecessors.add(expert_keys[origin])
            key = f"{invocation.invocation_id}.combine_reduce.rank{origin}"
            graph.add_compute(
                key,
                origin,
                self.cost_model.estimate(
                    max(1, route_count_by_origin[origin] * invocation.hidden * 2)
                ),
                predecessors=predecessors,
                metadata={
                    "algorithm": "eplb",
                    "operation": "combine_reduce",
                    "token_count": token_count,
                    "route_partials": route_count_by_origin[origin],
                },
            )
            terminal_keys.add(key)
            rank_terminals[origin] = frozenset({key})

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        server_forward_tasks = 0
        for task in created:
            if task.kind != "transfer":
                continue
            transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
            if task.route_spec and task.route_spec.startswith("server_forward "):
                server_forward_tasks += 1
        payload_count = sum(len(payloads) for payloads in payloads_by_pair.values())
        return AlgorithmBuildResult(
            algorithm="eplb_deepep_steady_state_forward",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "planner": "deepseek_eplb_hierarchical_python_v1",
                "source_commit": "d52c72d",
                "policy": "hierarchical",
                "load_source": self.config.load_source,
                "placement_epoch": "steady_state",
                "weight_migration_modeled": False,
                "num_logical_experts": invocation.placement.num_experts,
                "num_physical_experts": self.config.num_physical_experts,
                "num_redundant_experts": (
                    self.config.num_physical_experts
                    - invocation.placement.num_experts
                ),
                "num_groups": self.config.num_groups,
                "estimated_loads": placement_plan.estimated_loads,
                "physical_to_logical": placement_plan.physical_to_logical,
                "physical_replica_rank": placement_plan.physical_replica_rank,
                "physical_to_rank": placement_plan.physical_to_rank,
                "logical_to_physical": placement_plan.logical_to_physical,
                "logical_count": placement_plan.logical_count,
                "physical_experts_per_rank": (
                    placement_plan.physical_experts_per_rank
                ),
                "replica_selection": "deterministic_round_robin_v1",
                "scale_out_transport": "deepep_destination_forward",
                "token_payload_policy": {
                    "deduplicate": True,
                    "scope": "destination_rank",
                },
                "route_count_by_rank": dict(sorted(route_count_by_rank.items())),
                "route_count_by_physical_expert": dict(
                    sorted(route_count_by_physical.items())
                ),
                "route_count": len(invocation.assignments),
                "unique_token_payload_count": payload_count,
                "deduplicated_route_count": (
                    len(invocation.assignments) - payload_count
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
        execution: dict[RouteKey, EPLBExecution],
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
                        payload_chunk, execution, relay, route_spec
                    ),
                )
                arrivals[dst].add(key)

    def _build_combine(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        payloads_by_pair: dict[tuple[int, int], tuple[TokenPayload, ...]],
        execution: dict[RouteKey, EPLBExecution],
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
                        payload_chunk, execution, relay, route_spec
                    ),
                )
                arrivals[origin].add(key)

    @staticmethod
    def _route_key(assignment: RoutingAssignment) -> RouteKey:
        return (
            assignment.src_rank,
            assignment.token_id,
            assignment.topk_slot,
        )

    @classmethod
    def _transfer_metadata(
        cls,
        payloads: tuple[TokenPayload, ...],
        execution: dict[RouteKey, EPLBExecution],
        relay: int | None,
        route_spec: str | None,
    ) -> dict[str, object]:
        return {
            "payloads": [
                {
                    "src_rank": payload.src_rank,
                    "token_id": payload.token_id,
                    "route_count": len(payload.routes),
                    "logical_experts": [
                        route.expert_id for route in payload.routes
                    ],
                    "physical_experts": [
                        execution[cls._route_key(route)].physical_expert
                        for route in payload.routes
                    ],
                }
                for payload in payloads
            ],
            "deduplicated_by_destination_rank": True,
            "forwarding": "destination" if route_spec else "local",
            "dst_relay": relay,
        }
