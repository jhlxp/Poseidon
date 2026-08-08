from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from math import ceil, isfinite

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..load_profile import ExpertInstance, build_expert_load_profile
from ..schema import MoEInvocation, Placement, RoutingAssignment, ValidationError
from .common import (
    AlgorithmBuildResult,
    HierarchicalTransferSummary,
    build_hierarchical_combine,
    build_hierarchical_dispatch,
    plan_hierarchical_token_payloads,
)


RouteKey = tuple[int, int, int]


@dataclass(frozen=True)
class ProbeNICControllerConfig:
    initial_budget_bytes: int = 16 * 1024 * 1024
    min_budget_bytes: int = 0
    max_budget_bytes: int = 128 * 1024 * 1024
    multiplicative_decrease: float = 0.9
    additive_increase_bytes: int = 1024 * 1024
    deadband_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.initial_budget_bytes < 0:
            raise ValidationError("initial_budget_bytes must be non-negative")
        if self.min_budget_bytes < 0:
            raise ValidationError("min_budget_bytes must be non-negative")
        if self.max_budget_bytes <= 0:
            raise ValidationError("max_budget_bytes must be positive")
        if not (
            self.min_budget_bytes
            <= self.initial_budget_bytes
            <= self.max_budget_bytes
        ):
            raise ValidationError(
                "NIC budgets require min <= initial <= max"
            )
        if (
            not isfinite(self.multiplicative_decrease)
            or self.multiplicative_decrease <= 0
            or self.multiplicative_decrease >= 1
        ):
            raise ValidationError(
                "multiplicative_decrease must be in (0, 1)"
            )
        if self.additive_increase_bytes <= 0:
            raise ValidationError("additive_increase_bytes must be positive")
        if (
            not isfinite(self.deadband_ratio)
            or self.deadband_ratio < 0
            or self.deadband_ratio >= 1
        ):
            raise ValidationError("deadband_ratio must be in [0, 1)")


@dataclass(frozen=True)
class ProbeSampleFeedback:
    sample_id: int
    compute_us_by_rank: tuple[float, ...]
    nic_us_by_rank: tuple[float, ...]
    migration_bytes_by_rank: tuple[int, ...]
    pending_migration_exists: bool
    source: str = "external_measurement"

    def validate(self, num_ranks: int) -> None:
        if self.sample_id < 0:
            raise ValidationError("feedback sample_id must be non-negative")
        for name, values in (
            ("compute_us_by_rank", self.compute_us_by_rank),
            ("nic_us_by_rank", self.nic_us_by_rank),
            ("migration_bytes_by_rank", self.migration_bytes_by_rank),
        ):
            if len(values) != num_ranks:
                raise ValidationError(
                    f"feedback {name} needs {num_ranks} entries"
                )
        if any(not isfinite(value) or value < 0 for value in self.compute_us_by_rank):
            raise ValidationError("feedback compute times must be finite and non-negative")
        if any(not isfinite(value) or value < 0 for value in self.nic_us_by_rank):
            raise ValidationError("feedback NIC times must be finite and non-negative")
        if any(value < 0 for value in self.migration_bytes_by_rank):
            raise ValidationError("feedback migration bytes must be non-negative")
        if not self.source:
            raise ValidationError("feedback source must not be empty")


@dataclass(frozen=True)
class ProbeControllerUpdate:
    sample_id: int
    action: str
    compute_max_us: float
    nic_max_us: float
    bottleneck_ranks: tuple[int, ...]
    budget_before: tuple[int, ...]
    budget_after: tuple[int, ...]
    migration_bytes_by_rank: tuple[int, ...]
    pending_migration_exists: bool
    feedback_source: str

    def manifest(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "action": self.action,
            "compute_max_us": self.compute_max_us,
            "nic_max_us": self.nic_max_us,
            "bottleneck_ranks": list(self.bottleneck_ranks),
            "budget_before": list(self.budget_before),
            "budget_after": list(self.budget_after),
            "migration_bytes_by_rank": list(self.migration_bytes_by_rank),
            "pending_migration_exists": self.pending_migration_exists,
            "feedback_source": self.feedback_source,
        }


class ProbeNICController:
    def __init__(
        self,
        num_ranks: int,
        config: ProbeNICControllerConfig = ProbeNICControllerConfig(),
    ) -> None:
        if num_ranks <= 0:
            raise ValidationError("controller num_ranks must be positive")
        self.num_ranks = num_ranks
        self.config = config
        self._budgets = [config.initial_budget_bytes] * num_ranks
        self._last_update: ProbeControllerUpdate | None = None

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(self._budgets)

    @property
    def last_update(self) -> ProbeControllerUpdate | None:
        return self._last_update

    def update(self, feedback: ProbeSampleFeedback) -> ProbeControllerUpdate:
        feedback.validate(self.num_ranks)
        before = tuple(self._budgets)
        compute_max = max(feedback.compute_us_by_rank, default=0.0)
        nic_max = max(feedback.nic_us_by_rank, default=0.0)
        deadband = self.config.deadband_ratio
        bottlenecks: tuple[int, ...] = ()
        if nic_max > (1.0 + deadband) * compute_max:
            threshold = (1.0 - deadband) * nic_max
            bottlenecks = tuple(
                rank
                for rank, value in enumerate(feedback.nic_us_by_rank)
                if value >= threshold
            )
            for rank in bottlenecks:
                reduced = int(
                    self._budgets[rank]
                    * self.config.multiplicative_decrease
                )
                self._budgets[rank] = max(
                    self.config.min_budget_bytes, reduced
                )
            action = "multiplicative_decrease"
        elif (
            nic_max < (1.0 - deadband) * compute_max
            and feedback.pending_migration_exists
        ):
            for rank in range(self.num_ranks):
                self._budgets[rank] = min(
                    self.config.max_budget_bytes,
                    self._budgets[rank]
                    + self.config.additive_increase_bytes,
                )
            action = "additive_increase"
        else:
            action = "hold"
        update = ProbeControllerUpdate(
            sample_id=feedback.sample_id,
            action=action,
            compute_max_us=compute_max,
            nic_max_us=nic_max,
            bottleneck_ranks=bottlenecks,
            budget_before=before,
            budget_after=tuple(self._budgets),
            migration_bytes_by_rank=feedback.migration_bytes_by_rank,
            pending_migration_exists=feedback.pending_migration_exists,
            feedback_source=feedback.source,
        )
        self._last_update = update
        return update


@dataclass(frozen=True)
class ProbeEPConfig:
    replicas_per_rank: int
    token_padding: int = 128
    chunk_tokens: int = 128
    route_chunk_tokens: int = 128
    weight_chunk_bytes: int = 4 * 1024 * 1024
    max_remote_replicas: int = 64
    overlap_expert_compute: bool = True
    payload_metadata_sample_limit: int = 8
    nic_controller: ProbeNICControllerConfig = ProbeNICControllerConfig()

    def __post_init__(self) -> None:
        if self.replicas_per_rank < 0:
            raise ValidationError("replicas_per_rank must be non-negative")
        if self.token_padding <= 0:
            raise ValidationError("token_padding must be positive")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")
        if self.route_chunk_tokens <= 0:
            raise ValidationError("route_chunk_tokens must be positive")
        if self.weight_chunk_bytes <= 0:
            raise ValidationError("weight_chunk_bytes must be positive")
        if self.max_remote_replicas < 0:
            raise ValidationError("max_remote_replicas must be non-negative")
        if self.payload_metadata_sample_limit < 0:
            raise ValidationError(
                "payload_metadata_sample_limit must be non-negative"
            )


@dataclass(frozen=True)
class MigrationIntent:
    intent_id: int
    priority: int
    expert_id: int
    source_server: int
    destination_server: int
    moved_route_keys: tuple[RouteKey, ...]
    moved_compute_us: float

    def manifest(self, status: str) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "priority": self.priority,
            "expert_id": self.expert_id,
            "source_server": self.source_server,
            "destination_server": self.destination_server,
            "moved_route_count": len(self.moved_route_keys),
            "moved_compute_us": self.moved_compute_us,
            "status": status,
        }


@dataclass(frozen=True)
class WeightChunkPlan:
    chunk_id: int
    offset_bytes: int
    transfer_bytes: int
    rail: int
    source_relay: int
    destination_relay: int

    def manifest(self) -> dict[str, int]:
        return {
            "chunk_id": self.chunk_id,
            "offset_bytes": self.offset_bytes,
            "transfer_bytes": self.transfer_bytes,
            "plane": 0,
            "rail": self.rail,
            "source_relay": self.source_relay,
            "destination_relay": self.destination_relay,
        }


@dataclass(frozen=True)
class RemoteReplicaPlan:
    expert_id: int
    source_server: int
    destination_server: int
    home_rank: int
    destination_rank: int
    moved_route_keys: tuple[RouteKey, ...]
    weight_chunks: tuple[WeightChunkPlan, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "source_server": self.source_server,
            "destination_server": self.destination_server,
            "home_rank": self.home_rank,
            "destination_rank": self.destination_rank,
            "moved_route_count": len(self.moved_route_keys),
            "weight_bytes": sum(
                chunk.transfer_bytes for chunk in self.weight_chunks
            ),
            "weight_chunk_count": len(self.weight_chunks),
            "weight_chunks": [chunk.manifest() for chunk in self.weight_chunks],
        }


@dataclass(frozen=True)
class LocalReplicaTransfer:
    expert_id: int
    source_rank: int
    destination_rank: int
    scope: str


@dataclass(frozen=True)
class _IntraServerPlan:
    execution_rank: dict[RouteKey, int]
    real_routes_by_rank: dict[int, int]
    temporary_experts_by_rank: dict[int, tuple[int, ...]]
    anchor_rank: dict[tuple[int, int], int]
    local_prefetches: tuple[LocalReplicaTransfer, ...]


@dataclass(frozen=True)
class ProbeEPPlan:
    sample_id: int
    execution_rank: dict[RouteKey, int]
    planned_intents: tuple[MigrationIntent, ...]
    admitted_intents: tuple[MigrationIntent, ...]
    deferred_intents: tuple[MigrationIntent, ...]
    remote_replicas: tuple[RemoteReplicaPlan, ...]
    local_prefetches: tuple[LocalReplicaTransfer, ...]
    baseline_server_routes: dict[int, int]
    planned_server_routes: dict[int, int]
    admitted_server_routes: dict[int, int]
    baseline_real_routes_by_rank: dict[int, int]
    final_real_routes_by_rank: dict[int, int]
    baseline_padded_routes_by_rank: dict[int, int]
    final_padded_routes_by_rank: dict[int, int]
    baseline_compute_us_by_rank: dict[int, float]
    final_compute_us_by_rank: dict[int, float]
    nic_budget_before: tuple[int, ...]
    assigned_migration_tx_bytes: tuple[int, ...]
    assigned_migration_rx_bytes: tuple[int, ...]
    predicted_token_tx_bytes: tuple[int, ...]
    predicted_token_rx_bytes: tuple[int, ...]
    controller_last_update: ProbeControllerUpdate | None


class ProbeEPBuilder:
    """Standalone two-stage compute planner plus feedback NIC admission."""

    def __init__(
        self,
        cost_model: ComputeCostModel,
        config: ProbeEPConfig,
        *,
        controller: ProbeNICController | None = None,
    ) -> None:
        self.cost_model = cost_model
        self.config = config
        self._controller = controller
        self._next_sample_id = 0

    @property
    def controller(self) -> ProbeNICController | None:
        return self._controller

    def apply_feedback(
        self, feedback: ProbeSampleFeedback
    ) -> ProbeControllerUpdate:
        if self._controller is None:
            raise ValidationError(
                "ProbeEP controller is initialized by the first plan/build"
            )
        return self._controller.update(feedback)

    def plan(self, invocation: MoEInvocation) -> ProbeEPPlan:
        placement = invocation.placement
        controller = self._ensure_controller(placement.num_ranks)
        assignments = invocation.assignments
        assignments_by_key = {
            self._route_key(assignment): assignment
            for assignment in assignments
        }
        home_server_by_route = {
            route_key: placement.rank_server(
                placement.expert_rank(assignment.expert_id)
            )
            for route_key, assignment in assignments_by_key.items()
        }
        baseline_intra = self._plan_intra_server(
            invocation, assignments_by_key, home_server_by_route
        )
        planned_server_by_route, planned_intents = self._plan_inter_server(
            invocation,
            assignments_by_key,
            home_server_by_route,
        )
        (
            admitted_server_by_route,
            admitted_intents,
            deferred_intents,
            chunks_by_replica,
            assigned_tx,
            assigned_rx,
        ) = self._admit_intents(
            invocation,
            home_server_by_route,
            planned_intents,
            controller.budgets,
        )
        final_intra = self._plan_intra_server(
            invocation, assignments_by_key, admitted_server_by_route
        )
        baseline_padded, baseline_compute = self._compute_state(
            invocation, assignments_by_key, baseline_intra.execution_rank
        )
        final_padded, final_compute = self._compute_state(
            invocation, assignments_by_key, final_intra.execution_rank
        )
        predicted_token_tx, predicted_token_rx = self._predicted_token_nic_bytes(
            invocation, admitted_server_by_route
        )
        admitted_routes_by_replica: dict[
            tuple[int, int], list[RouteKey]
        ] = defaultdict(list)
        source_server_by_replica: dict[tuple[int, int], int] = {}
        for intent in admitted_intents:
            key = (intent.expert_id, intent.destination_server)
            admitted_routes_by_replica[key].extend(intent.moved_route_keys)
            source_server_by_replica[key] = intent.source_server
        remote_replicas: list[RemoteReplicaPlan] = []
        for replica_key, moved_keys in sorted(admitted_routes_by_replica.items()):
            expert, destination_server = replica_key
            remote_replicas.append(
                RemoteReplicaPlan(
                    expert_id=expert,
                    source_server=source_server_by_replica[replica_key],
                    destination_server=destination_server,
                    home_rank=placement.expert_rank(expert),
                    destination_rank=final_intra.anchor_rank[replica_key],
                    moved_route_keys=tuple(moved_keys),
                    weight_chunks=chunks_by_replica[replica_key],
                )
            )
        return ProbeEPPlan(
            sample_id=self._next_sample_id,
            execution_rank=final_intra.execution_rank,
            planned_intents=planned_intents,
            admitted_intents=admitted_intents,
            deferred_intents=deferred_intents,
            remote_replicas=tuple(remote_replicas),
            local_prefetches=final_intra.local_prefetches,
            baseline_server_routes=self._server_route_counts(
                placement, home_server_by_route
            ),
            planned_server_routes=self._server_route_counts(
                placement, planned_server_by_route
            ),
            admitted_server_routes=self._server_route_counts(
                placement, admitted_server_by_route
            ),
            baseline_real_routes_by_rank=baseline_intra.real_routes_by_rank,
            final_real_routes_by_rank=final_intra.real_routes_by_rank,
            baseline_padded_routes_by_rank=baseline_padded,
            final_padded_routes_by_rank=final_padded,
            baseline_compute_us_by_rank=baseline_compute,
            final_compute_us_by_rank=final_compute,
            nic_budget_before=controller.budgets,
            assigned_migration_tx_bytes=tuple(assigned_tx),
            assigned_migration_rx_bytes=tuple(assigned_rx),
            predicted_token_tx_bytes=predicted_token_tx,
            predicted_token_rx_bytes=predicted_token_rx,
            controller_last_update=controller.last_update,
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

        remote_ready_by_replica: dict[tuple[int, int], set[str]] = defaultdict(set)
        for replica in plan.remote_replicas:
            replica_key = (replica.expert_id, replica.destination_server)
            for chunk in replica.weight_chunks:
                prefix = (
                    f"{invocation.invocation_id}.probe.remote_prefetch."
                    f"expert{replica.expert_id}.server{replica.destination_server}."
                    f"chunk{chunk.chunk_id}"
                )
                predecessors = set(roots)
                if replica.home_rank != chunk.source_relay:
                    scatter_key = f"{prefix}.scatter"
                    graph.add_transfer(
                        scatter_key,
                        replica.home_rank,
                        chunk.source_relay,
                        chunk.transfer_bytes,
                        "expert_weight_scatter",
                        f"{invocation.invocation_id}:probe:remote_prefetch",
                        predecessors=predecessors,
                        chunk_id=chunk.chunk_id,
                        metadata=self._weight_metadata(
                            plan.sample_id,
                            replica,
                            chunk,
                            "source_local_scatter",
                        ),
                    )
                    predecessors = {scatter_key}
                rdma_key = f"{prefix}.rdma"
                graph.add_transfer(
                    rdma_key,
                    chunk.source_relay,
                    chunk.destination_relay,
                    chunk.transfer_bytes,
                    "expert_weight_rdma",
                    f"{invocation.invocation_id}:probe:remote_prefetch",
                    predecessors=predecessors,
                    chunk_id=chunk.chunk_id,
                    metadata=self._weight_metadata(
                        plan.sample_id, replica, chunk, "same_rail_rdma"
                    ),
                )
                final_key = rdma_key
                if chunk.destination_relay != replica.destination_rank:
                    gather_key = f"{prefix}.gather"
                    graph.add_transfer(
                        gather_key,
                        chunk.destination_relay,
                        replica.destination_rank,
                        chunk.transfer_bytes,
                        "expert_weight_gather",
                        f"{invocation.invocation_id}:probe:remote_prefetch",
                        predecessors={rdma_key},
                        chunk_id=chunk.chunk_id,
                        metadata=self._weight_metadata(
                            plan.sample_id,
                            replica,
                            chunk,
                            "destination_local_gather",
                        ),
                    )
                    final_key = gather_key
                remote_ready_by_replica[replica_key].add(final_key)

        local_ready_by_rank: dict[int, set[str]] = defaultdict(set)
        for index, prefetch in enumerate(plan.local_prefetches):
            source_server = placement.rank_server(prefetch.source_rank)
            home_server = placement.rank_server(
                placement.expert_rank(prefetch.expert_id)
            )
            predecessors = set(roots)
            if source_server != home_server:
                predecessors = set(
                    remote_ready_by_replica[
                        (prefetch.expert_id, source_server)
                    ]
                )
            key = (
                f"{invocation.invocation_id}.probe.local_prefetch."
                f"expert{prefetch.expert_id}.src{prefetch.source_rank}."
                f"dst{prefetch.destination_rank}.slot{index}"
            )
            graph.add_transfer(
                key,
                prefetch.source_rank,
                prefetch.destination_rank,
                invocation.expert_weight_bytes,
                "expert_weight_prefetch",
                f"{invocation.invocation_id}:probe:local_prefetch",
                predecessors=predecessors,
                metadata={
                    "algorithm": "probeep",
                    "sample_id": plan.sample_id,
                    "expert_id": prefetch.expert_id,
                    "replica_scope": prefetch.scope,
                    "transport": "server_local_fullmesh",
                },
            )
            local_ready_by_rank[prefetch.destination_rank].add(key)

        payload_plan = plan_hierarchical_token_payloads(
            invocation.assignments,
            lambda assignment: plan.execution_rank[self._route_key(assignment)],
            placement,
        )
        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        transfer_summary = HierarchicalTransferSummary()
        build_hierarchical_dispatch(
            graph,
            invocation,
            payload_plan,
            algorithm="probeep",
            chunk_tokens=self.config.chunk_tokens,
            predecessors_for_server=lambda _server: set(roots),
            metadata_sample_limit=self.config.payload_metadata_sample_limit,
            arrivals=dispatch_arrivals,
            summary=transfer_summary,
        )

        remote_seed_ready_by_rank: dict[int, set[str]] = defaultdict(set)
        for replica in plan.remote_replicas:
            remote_seed_ready_by_rank[replica.destination_rank].update(
                remote_ready_by_replica[
                    (replica.expert_id, replica.destination_server)
                ]
            )
        assignments_by_rank_expert: dict[
            tuple[int, int], list[RoutingAssignment]
        ] = defaultdict(list)
        for assignment in invocation.assignments:
            rank = plan.execution_rank[self._route_key(assignment)]
            assignments_by_rank_expert[(rank, assignment.expert_id)].append(
                assignment
            )

        expert_keys: dict[int, str] = {}
        for rank in range(placement.num_ranks):
            padded_routes = plan.final_padded_routes_by_rank[rank]
            if padded_routes == 0:
                continue
            expert_counts = {
                expert: len(routes)
                for (item_rank, expert), routes in assignments_by_rank_expert.items()
                if item_rank == rank and routes
            }
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
                    roots
                    | remote_seed_ready_by_rank[rank]
                    | local_ready_by_rank[rank]
                    | dispatch_arrivals[rank]
                ),
                metadata={
                    "algorithm": "probeep",
                    "operation": "expert_ffn",
                    "sample_id": plan.sample_id,
                    "real_token_routes": plan.final_real_routes_by_rank[rank],
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
            algorithm="probeep",
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
                    "algorithm": "probeep",
                    "operation": "combine_reduce",
                    "sample_id": plan.sample_id,
                    "token_count": token_count,
                },
            )
            terminal_keys.add(key)
            rank_terminals[rank] = frozenset({key})

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind == "transfer":
                transfer_bytes[task.payload_kind or "unspecified"] += (
                    task.transfer_bytes
                )
        expert_load_profile = build_expert_load_profile(
            invocation,
            after_instances=self._expert_instances(invocation, plan),
            select_after_instance=lambda assignment: (
                f"logical:{assignment.expert_id}:rank:"
                f"{plan.execution_rank[self._route_key(assignment)]}"
            ),
        )
        controller_update = (
            plan.controller_last_update.manifest()
            if plan.controller_last_update is not None
            else None
        )
        result = AlgorithmBuildResult(
            algorithm="probeep_deepep_hierarchical",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "sample_id": plan.sample_id,
                "planner": "two_stage_histogram_greedy_v3",
                "planning_mode": "oracle_current_routes",
                "planner_runtime_model": "not_in_dag",
                "planner_complexity_scope": "paper_analysis_only",
                "planner_decision_input": "server_expert_histogram",
                "planner_search_complexity": {
                    "inter_server": "O(E log E + M log S)",
                    "intra_server": "O(E log E + M log G)",
                    "controller": "O(P)",
                    "chunk_assignment": "O(C * G)",
                },
                "route_lowering_runtime_model": "offline_not_in_dag",
                "feedback_mode": "external_sample_feedback",
                "controller_last_update": controller_update,
                "nic_budget_before": list(plan.nic_budget_before),
                "assigned_migration_tx_bytes": list(
                    plan.assigned_migration_tx_bytes
                ),
                "assigned_migration_rx_bytes": list(
                    plan.assigned_migration_rx_bytes
                ),
                "predicted_token_tx_bytes": list(
                    plan.predicted_token_tx_bytes
                ),
                "predicted_token_rx_bytes": list(
                    plan.predicted_token_rx_bytes
                ),
                "planned_migration_intents": [
                    intent.manifest("planned")
                    for intent in plan.planned_intents
                ],
                "admitted_migration_intents": [
                    intent.manifest("admitted")
                    for intent in plan.admitted_intents
                ],
                "deferred_migration_intents": [
                    intent.manifest("deferred")
                    for intent in plan.deferred_intents
                ],
                "baseline_server_routes": plan.baseline_server_routes,
                "planned_server_routes": plan.planned_server_routes,
                "admitted_server_routes": plan.admitted_server_routes,
                "baseline_real_routes_by_rank": (
                    plan.baseline_real_routes_by_rank
                ),
                "final_real_routes_by_rank": plan.final_real_routes_by_rank,
                "baseline_padded_routes_by_rank": (
                    plan.baseline_padded_routes_by_rank
                ),
                "final_padded_routes_by_rank": (
                    plan.final_padded_routes_by_rank
                ),
                "baseline_compute_us_by_rank": (
                    plan.baseline_compute_us_by_rank
                ),
                "final_compute_us_by_rank": plan.final_compute_us_by_rank,
                "remote_replicas": [
                    replica.manifest() for replica in plan.remote_replicas
                ],
                "local_prefetches": [
                    {
                        "expert_id": item.expert_id,
                        "source_rank": item.source_rank,
                        "destination_rank": item.destination_rank,
                        "scope": item.scope,
                    }
                    for item in plan.local_prefetches
                ],
                "route_chunk_tokens": self.config.route_chunk_tokens,
                "weight_chunk_bytes": self.config.weight_chunk_bytes,
                "replicas_per_rank": self.config.replicas_per_rank,
                "token_padding": self.config.token_padding,
                "scale_out_transport": "deepep_hierarchical",
                "token_payload_policy": {
                    "deduplicate": True,
                    "scope": "destination_rank_then_server",
                },
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
                "remote_weight_rdma_bytes": transfer_bytes.get(
                    "expert_weight_rdma", 0
                ),
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
                "expert_load_profile": expert_load_profile,
            },
        )
        self._next_sample_id += 1
        return result

    def _plan_inter_server(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        home_server_by_route: dict[RouteKey, int],
    ) -> tuple[dict[RouteKey, int], tuple[MigrationIntent, ...]]:
        placement = invocation.placement
        route_server = dict(home_server_by_route)
        routes_by_server_expert: dict[
            tuple[int, int], deque[RouteKey]
        ] = defaultdict(deque)
        for route_key, assignment in assignments_by_key.items():
            routes_by_server_expert[
                (home_server_by_route[route_key], assignment.expert_id)
            ].append(route_key)
        expert_heaps: dict[int, list[tuple[int, int]]] = {
            server: [] for server in range(placement.num_servers)
        }
        for (server, expert), routes in routes_by_server_expert.items():
            heapq.heappush(expert_heaps[server], (-len(routes), expert))
        server_load = self._server_route_counts(placement, route_server)
        total_routes = len(assignments_by_key)
        target_floor, extra = divmod(total_routes, placement.num_servers)
        target_ceil = target_floor + (1 if extra else 0)
        per_route_us = self._expert_route_us(invocation)
        aggregated: dict[tuple[int, int, int], list[RouteKey]] = {}
        priority_by_key: dict[tuple[int, int, int], int] = {}
        next_priority = 0

        while True:
            source = min(
                range(placement.num_servers),
                key=lambda server: (-server_load[server], server),
            )
            destination = min(
                range(placement.num_servers),
                key=lambda server: (server_load[server], server),
            )
            if (
                source == destination
                or server_load[source] <= target_ceil
                or server_load[destination] >= target_floor
            ):
                break
            heap = expert_heaps[source]
            while heap:
                negative_count, expert = heap[0]
                current_count = len(routes_by_server_expert[(source, expert)])
                if -negative_count == current_count and current_count > 0:
                    break
                heapq.heappop(heap)
            if not heap:
                break
            _, expert = heapq.heappop(heap)
            routes = routes_by_server_expert[(source, expert)]
            move_count = min(
                self.config.route_chunk_tokens,
                server_load[source] - target_ceil,
                target_floor - server_load[destination],
                len(routes),
            )
            if move_count <= 0:
                break
            moved = tuple(routes.popleft() for _ in range(move_count))
            if routes:
                heapq.heappush(heap, (-len(routes), expert))
            key = (expert, source, destination)
            if key not in aggregated:
                aggregated[key] = []
                priority_by_key[key] = next_priority
                next_priority += 1
            aggregated[key].extend(moved)
            for route_key in moved:
                route_server[route_key] = destination
            server_load[source] -= move_count
            server_load[destination] += move_count

        intents = tuple(
            MigrationIntent(
                intent_id=index,
                priority=priority_by_key[key],
                expert_id=key[0],
                source_server=key[1],
                destination_server=key[2],
                moved_route_keys=tuple(route_keys),
                moved_compute_us=len(route_keys) * per_route_us,
            )
            for index, (key, route_keys) in enumerate(
                sorted(aggregated.items(), key=lambda item: priority_by_key[item[0]])
            )
        )
        return route_server, intents

    def _admit_intents(
        self,
        invocation: MoEInvocation,
        home_server_by_route: dict[RouteKey, int],
        intents: tuple[MigrationIntent, ...],
        budgets: tuple[int, ...],
    ) -> tuple[
        dict[RouteKey, int],
        tuple[MigrationIntent, ...],
        tuple[MigrationIntent, ...],
        dict[tuple[int, int], tuple[WeightChunkPlan, ...]],
        list[int],
        list[int],
    ]:
        route_server = dict(home_server_by_route)
        admitted: list[MigrationIntent] = []
        deferred: list[MigrationIntent] = []
        chunks_by_replica: dict[
            tuple[int, int], tuple[WeightChunkPlan, ...]
        ] = {}
        assigned_tx = [0] * invocation.placement.num_ranks
        assigned_rx = [0] * invocation.placement.num_ranks
        for intent in sorted(intents, key=lambda item: (item.priority, item.intent_id)):
            replica_key = (intent.expert_id, intent.destination_server)
            if replica_key not in chunks_by_replica:
                if len(chunks_by_replica) >= self.config.max_remote_replicas:
                    deferred.append(intent)
                    continue
                candidate_route_server = dict(route_server)
                for route_key in intent.moved_route_keys:
                    candidate_route_server[route_key] = intent.destination_server
                token_tx, token_rx = self._predicted_token_nic_bytes(
                    invocation, candidate_route_server
                )
                schedule = self._schedule_weight_chunks(
                    invocation,
                    intent.source_server,
                    intent.destination_server,
                    budgets,
                    assigned_tx,
                    assigned_rx,
                    token_tx,
                    token_rx,
                )
                if schedule is None:
                    deferred.append(intent)
                    continue
                chunks, next_tx, next_rx = schedule
                chunks_by_replica[replica_key] = chunks
                assigned_tx = next_tx
                assigned_rx = next_rx
            admitted.append(intent)
            for route_key in intent.moved_route_keys:
                route_server[route_key] = intent.destination_server
        return (
            route_server,
            tuple(admitted),
            tuple(deferred),
            chunks_by_replica,
            assigned_tx,
            assigned_rx,
        )

    def _schedule_weight_chunks(
        self,
        invocation: MoEInvocation,
        source_server: int,
        destination_server: int,
        budgets: tuple[int, ...],
        assigned_tx: list[int],
        assigned_rx: list[int],
        token_tx: tuple[int, ...],
        token_rx: tuple[int, ...],
    ) -> tuple[tuple[WeightChunkPlan, ...], list[int], list[int]] | None:
        placement = invocation.placement
        next_tx = list(assigned_tx)
        next_rx = list(assigned_rx)
        chunks: list[WeightChunkPlan] = []
        remaining = invocation.expert_weight_bytes
        offset = 0
        while remaining:
            chunk_bytes = min(self.config.weight_chunk_bytes, remaining)
            candidates: list[tuple[int, int, int, int, int]] = []
            for rail in range(placement.gpus_per_server):
                source_rank = placement.server_rank(source_server, rail)
                destination_rank = placement.server_rank(
                    destination_server, rail
                )
                if (
                    next_tx[source_rank] + chunk_bytes > budgets[source_rank]
                    or next_rx[destination_rank] + chunk_bytes
                    > budgets[destination_rank]
                    or budgets[source_rank] == 0
                    or budgets[destination_rank] == 0
                ):
                    continue
                projected_tx = (
                    token_tx[source_rank]
                    + next_tx[source_rank]
                    + chunk_bytes
                )
                projected_rx = (
                    token_rx[destination_rank]
                    + next_rx[destination_rank]
                    + chunk_bytes
                )
                candidates.append(
                    (
                        max(projected_tx, projected_rx),
                        projected_tx + projected_rx,
                        rail,
                        source_rank,
                        destination_rank,
                    )
                )
            if not candidates:
                return None
            _, _, rail, source_rank, destination_rank = min(candidates)
            next_tx[source_rank] += chunk_bytes
            next_rx[destination_rank] += chunk_bytes
            chunks.append(
                WeightChunkPlan(
                    chunk_id=len(chunks),
                    offset_bytes=offset,
                    transfer_bytes=chunk_bytes,
                    rail=rail,
                    source_relay=source_rank,
                    destination_relay=destination_rank,
                )
            )
            offset += chunk_bytes
            remaining -= chunk_bytes
        return tuple(chunks), next_tx, next_rx

    def _predicted_token_nic_bytes(
        self,
        invocation: MoEInvocation,
        route_server: dict[RouteKey, int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Predict deduplicated DeepEP fabric bytes for NIC load ordering."""
        placement = invocation.placement
        remote_payloads: set[tuple[int, int, int]] = set()
        for assignment in invocation.assignments:
            route_key = self._route_key(assignment)
            destination_server = route_server[route_key]
            if placement.rank_server(assignment.src_rank) == destination_server:
                continue
            remote_payloads.add(
                (assignment.src_rank, assignment.token_id, destination_server)
            )

        tx = [0] * placement.num_ranks
        rx = [0] * placement.num_ranks
        for source_rank, _token_id, destination_server in remote_payloads:
            relay = placement.server_rank(
                destination_server, placement.rank_local(source_rank)
            )
            tx[source_rank] += invocation.dispatch_token_bytes
            rx[relay] += invocation.dispatch_token_bytes
            tx[relay] += invocation.combine_token_bytes
            rx[source_rank] += invocation.combine_token_bytes
        return tuple(tx), tuple(rx)

    def _plan_intra_server(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        route_server: dict[RouteKey, int],
    ) -> _IntraServerPlan:
        placement = invocation.placement
        routes_by_server_expert: dict[
            tuple[int, int], list[RouteKey]
        ] = defaultdict(list)
        for route_key, assignment in assignments_by_key.items():
            routes_by_server_expert[
                (route_server[route_key], assignment.expert_id)
            ].append(route_key)
        execution: dict[RouteKey, int] = {}
        real_routes = {rank: 0 for rank in range(placement.num_ranks)}
        temporary: dict[int, set[int]] = {
            rank: set() for rank in range(placement.num_ranks)
        }
        anchor_rank: dict[tuple[int, int], int] = {}
        instances: dict[tuple[int, int], set[int]] = defaultdict(set)

        for server in range(placement.num_servers):
            ranks = tuple(
                placement.server_rank(server, local)
                for local in range(placement.gpus_per_server)
            )
            server_groups = {
                expert: routes
                for (item_server, expert), routes in routes_by_server_expert.items()
                if item_server == server and routes
            }
            total = sum(len(routes) for routes in server_groups.values())
            base, extra = divmod(total, placement.gpus_per_server)
            remaining = {
                rank: base + (1 if local < extra else 0)
                for local, rank in enumerate(ranks)
            }
            ordered_experts = sorted(
                server_groups,
                key=lambda expert: (-len(server_groups[expert]), expert),
            )
            for expert in ordered_experts:
                home_rank = placement.expert_rank(expert)
                home_server = placement.rank_server(home_rank)
                if home_server == server:
                    anchor = home_rank
                else:
                    available_anchors = [
                        rank
                        for rank in ranks
                        if remaining[rank] > 0
                        and len(temporary[rank]) < self.config.replicas_per_rank
                    ]
                    if not available_anchors:
                        raise ValidationError(
                            "ProbeEP cannot place remote seed for expert "
                            f"{expert} on server {server}"
                        )
                    anchor = min(
                        available_anchors,
                        key=lambda rank: (-remaining[rank], rank),
                    )
                    temporary[anchor].add(expert)
                anchor_rank[(expert, server)] = anchor
                instances[(expert, server)].add(anchor)
                candidates = instances[(expert, server)]
                routes = server_groups[expert]
                while sum(remaining[rank] for rank in candidates) < len(routes):
                    available = [
                        rank
                        for rank in ranks
                        if rank not in candidates
                        and remaining[rank] > 0
                        and len(temporary[rank]) < self.config.replicas_per_rank
                    ]
                    if not available:
                        raise ValidationError(
                            "ProbeEP intra-server planner cannot balance expert "
                            f"{expert} on server {server} with "
                            f"replicas_per_rank={self.config.replicas_per_rank}"
                        )
                    rank = min(
                        available, key=lambda item: (-remaining[item], item)
                    )
                    temporary[rank].add(expert)
                    candidates.add(rank)
                for route_key in routes:
                    rank = min(
                        (item for item in candidates if remaining[item] > 0),
                        key=lambda item: (-remaining[item], item),
                    )
                    execution[route_key] = rank
                    remaining[rank] -= 1
                    real_routes[rank] += 1
            if any(remaining.values()):
                raise ValidationError(
                    "ProbeEP intra-server planner left unused capacity on "
                    f"server {server}: {remaining}"
                )

        local_prefetches: list[LocalReplicaTransfer] = []
        for (expert, server), ranks in sorted(instances.items()):
            anchor = anchor_rank[(expert, server)]
            home_server = placement.rank_server(placement.expert_rank(expert))
            scope = (
                "home_server"
                if server == home_server
                else "remote_server"
            )
            for rank in sorted(ranks):
                if rank == anchor:
                    continue
                local_prefetches.append(
                    LocalReplicaTransfer(expert, anchor, rank, scope)
                )
        return _IntraServerPlan(
            execution_rank=execution,
            real_routes_by_rank=real_routes,
            temporary_experts_by_rank={
                rank: tuple(sorted(experts))
                for rank, experts in temporary.items()
            },
            anchor_rank=anchor_rank,
            local_prefetches=tuple(local_prefetches),
        )

    def _compute_state(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        execution_rank: dict[RouteKey, int],
    ) -> tuple[dict[int, int], dict[int, float]]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for route_key, assignment in assignments_by_key.items():
            counts[(execution_rank[route_key], assignment.expert_id)] += 1
        padded: dict[int, int] = {}
        compute: dict[int, float] = {}
        for rank in range(invocation.placement.num_ranks):
            padded_routes = sum(
                ceil(count / self.config.token_padding)
                * self.config.token_padding
                for (item_rank, _), count in counts.items()
                if item_rank == rank and count > 0
            )
            padded[rank] = padded_routes
            if padded_routes == 0:
                compute[rank] = 0.0
                continue
            compute[rank] = self.cost_model.estimate(
                padded_routes * 6 * invocation.hidden * invocation.ffn_hidden,
                operation="expert_ffn",
                overlaps_communication=self.config.overlap_expert_compute,
                token_count=padded_routes,
            ).duration_us
        return padded, compute

    def _expert_route_us(self, invocation: MoEInvocation) -> float:
        return self.cost_model.estimate(
            6 * invocation.hidden * invocation.ffn_hidden,
            operation="expert_ffn",
            overlaps_communication=self.config.overlap_expert_compute,
            token_count=1,
        ).duration_us

    def _ensure_controller(self, num_ranks: int) -> ProbeNICController:
        if self._controller is None:
            self._controller = ProbeNICController(
                num_ranks, self.config.nic_controller
            )
        elif self._controller.num_ranks != num_ranks:
            raise ValidationError(
                "ProbeEP controller rank count differs from placement"
            )
        return self._controller

    @staticmethod
    def _server_route_counts(
        placement: Placement, route_server: dict[RouteKey, int]
    ) -> dict[int, int]:
        result = {server: 0 for server in range(placement.num_servers)}
        for server in route_server.values():
            result[server] += 1
        return result

    @staticmethod
    def _weight_metadata(
        sample_id: int,
        replica: RemoteReplicaPlan,
        chunk: WeightChunkPlan,
        leg: str,
    ) -> dict[str, object]:
        return {
            "algorithm": "probeep",
            "sample_id": sample_id,
            "expert_id": replica.expert_id,
            "home_rank": replica.home_rank,
            "destination_rank": replica.destination_rank,
            "remote_weight_leg": leg,
            "plane": 0,
            "rail": chunk.rail,
            "source_relay": chunk.source_relay,
            "destination_relay": chunk.destination_relay,
            "chunk_offset_bytes": chunk.offset_bytes,
            "chunk_bytes": chunk.transfer_bytes,
        }

    @staticmethod
    def _expert_instances(
        invocation: MoEInvocation, plan: ProbeEPPlan
    ) -> tuple[ExpertInstance, ...]:
        placement = invocation.placement
        used = {
            (assignment.expert_id, plan.execution_rank[ProbeEPBuilder._route_key(assignment)])
            for assignment in invocation.assignments
        }
        values = [
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
        replica_index: dict[int, int] = defaultdict(int)
        for expert, rank in sorted(used):
            if rank == placement.expert_rank(expert):
                continue
            replica_index[expert] += 1
            values.append(
                ExpertInstance(
                    instance_id=f"logical:{expert}:rank:{rank}",
                    logical_expert=expert,
                    rank=rank,
                    kind=(
                        "local_replica"
                        if placement.rank_server(rank)
                        == placement.rank_server(placement.expert_rank(expert))
                        else "remote_replica"
                    ),
                    physical_expert=None,
                    replica_index=replica_index[expert],
                )
            )
        return tuple(values)

    @staticmethod
    def _route_key(assignment: RoutingAssignment) -> RouteKey:
        return (
            assignment.src_rank,
            assignment.token_id,
            assignment.topk_slot,
        )
