from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

from ..cost import ComputeCostModel
from ..graph import TaskGraph
from ..load_profile import ExpertInstance, build_expert_load_profile
from ..schema import MoEInvocation, Placement, RoutingAssignment, ValidationError
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
class ProbeNICControllerConfig:
    initial_budget_bytes: int = 16 * 1024 * 1024
    nic_line_rate_gbps: float = 400.0
    target_overlap_ratio: float = 0.90

    def __post_init__(self) -> None:
        if self.initial_budget_bytes < 0:
            raise ValidationError("initial_budget_bytes must be non-negative")
        if not isfinite(self.nic_line_rate_gbps) or self.nic_line_rate_gbps <= 0:
            raise ValidationError("nic_line_rate_gbps must be positive")
        if (
            not isfinite(self.target_overlap_ratio)
            or self.target_overlap_ratio <= 0
            or self.target_overlap_ratio > 1
        ):
            raise ValidationError("target_overlap_ratio must be in (0, 1]")

    @property
    def line_rate_bytes_per_us(self) -> float:
        return self.nic_line_rate_gbps * 125.0


DispatchOverlapComputeKind = Literal["attention", "moe"]


@dataclass(frozen=True)
class ProbeDispatchFeedback:
    observation_id: int
    attention_compute_us_by_rank: tuple[float, ...]
    moe_compute_us_by_rank: tuple[float, ...]
    dispatch_overlap_compute_kind: DispatchOverlapComputeKind
    weight_dispatch_us_by_rank: tuple[float, ...]
    dispatch_baseline_bytes_by_rank: tuple[int, ...]
    migration_bytes_by_rank: tuple[int, ...]
    migration_tx_bytes_by_rank: tuple[int, ...]
    migration_rx_bytes_by_rank: tuple[int, ...]
    pending_migration_exists: bool
    source: str = "external_measurement"
    dispatch_tx_bytes_by_rank: tuple[int, ...] = ()
    dispatch_rx_bytes_by_rank: tuple[int, ...] = ()

    def validate(self, num_ranks: int) -> None:
        if self.observation_id < 0:
            raise ValidationError(
                "Dispatch feedback observation_id must be non-negative"
            )
        for name, values in (
            ("attention_compute_us_by_rank", self.attention_compute_us_by_rank),
            ("moe_compute_us_by_rank", self.moe_compute_us_by_rank),
            ("weight_dispatch_us_by_rank", self.weight_dispatch_us_by_rank),
            (
                "dispatch_baseline_bytes_by_rank",
                self.dispatch_baseline_bytes_by_rank,
            ),
            ("migration_bytes_by_rank", self.migration_bytes_by_rank),
        ):
            if len(values) != num_ranks:
                raise ValidationError(
                    f"feedback {name} needs {num_ranks} entries"
                )
        for name, values in (
            ("attention", self.attention_compute_us_by_rank),
            ("moe", self.moe_compute_us_by_rank),
        ):
            if any(not isfinite(value) or value < 0 for value in values):
                raise ValidationError(
                    f"feedback {name} times must be finite and non-negative"
                )
        if self.dispatch_overlap_compute_kind not in {"attention", "moe"}:
            raise ValidationError(
                "Dispatch feedback compute kind is invalid"
            )
        if any(
            not isfinite(value) or value < 0
            for value in self.weight_dispatch_us_by_rank
        ):
            raise ValidationError(
                "Weight+Dispatch times must be finite and non-negative"
            )
        for name, values in (
            ("Dispatch baseline", self.dispatch_baseline_bytes_by_rank),
            ("migration", self.migration_bytes_by_rank),
        ):
            if any(value < 0 for value in values):
                raise ValidationError(f"feedback {name} bytes must be non-negative")
        for name, values in (
            ("migration TX", self.migration_tx_bytes_by_rank),
            ("migration RX", self.migration_rx_bytes_by_rank),
        ):
            if len(values) != num_ranks:
                raise ValidationError(
                    f"feedback {name} needs {num_ranks} entries"
                )
            if any(value < 0 for value in values):
                raise ValidationError(
                    f"feedback {name} bytes must be non-negative"
                )
        if bool(self.dispatch_tx_bytes_by_rank) != bool(
            self.dispatch_rx_bytes_by_rank
        ):
            raise ValidationError(
                "Dispatch TX/RX feedback must be provided together"
            )
        for name, values in (
            ("Dispatch TX", self.dispatch_tx_bytes_by_rank),
            ("Dispatch RX", self.dispatch_rx_bytes_by_rank),
        ):
            if values and len(values) != num_ranks:
                raise ValidationError(
                    f"feedback {name} needs {num_ranks} entries"
                )
            if any(value < 0 for value in values):
                raise ValidationError(
                    f"feedback {name} bytes must be non-negative"
                )
        if self.dispatch_tx_bytes_by_rank and tuple(
            max(tx, rx)
            for tx, rx in zip(
                self.dispatch_tx_bytes_by_rank,
                self.dispatch_rx_bytes_by_rank,
            )
        ) != self.dispatch_baseline_bytes_by_rank:
            raise ValidationError(
                "Dispatch baseline must equal max(TX,RX) per rank"
            )
        expected_endpoint = tuple(
            max(tx, rx)
            for tx, rx in zip(
                self.migration_tx_bytes_by_rank,
                self.migration_rx_bytes_by_rank,
            )
        )
        if self.migration_bytes_by_rank != expected_endpoint:
            raise ValidationError(
                "migration endpoint bytes must equal max(TX,RX) per rank"
            )
        if not self.source:
            raise ValidationError("feedback source must not be empty")


@dataclass(frozen=True)
class ProbeControllerUpdate:
    observation_id: int
    action: str
    dispatch_overlap_compute_kind: DispatchOverlapComputeKind
    attention_max_us: float
    moe_max_us: float
    selected_compute_ref_us: float
    target_nic_us: float
    weight_dispatch_max_us: float
    global_network_to_compute_ratio: float
    global_adjustment_factor: float
    nic_theoretical_max_bytes: int
    observed_total_bytes_by_rank: tuple[int, ...]
    bottleneck_observed_total_bytes: int
    probed_total_nic_max_bytes: int
    effective_total_nic_max_bytes: int
    bottleneck_ranks: tuple[int, ...]
    budget_before: tuple[int, ...]
    effective_rate_bytes_per_us_by_rank: tuple[float, ...]
    adjustment_factor_by_rank: tuple[float, ...]
    unclamped_target_budget_by_rank: tuple[int, ...]
    hard_migration_cap_by_rank: tuple[int, ...]
    budget_after: tuple[int, ...]
    dispatch_baseline_bytes_by_rank: tuple[int, ...]
    dispatch_tx_bytes_by_rank: tuple[int, ...]
    dispatch_rx_bytes_by_rank: tuple[int, ...]
    migration_bytes_by_rank: tuple[int, ...]
    migration_tx_bytes_by_rank: tuple[int, ...]
    migration_rx_bytes_by_rank: tuple[int, ...]
    pending_migration_exists: bool
    feedback_source: str

    def manifest(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "communication_kind": "dispatch",
            "observation_scope": "expert_weight_plus_dispatch",
            "action": self.action,
            "dispatch_overlap_compute_kind": (
                self.dispatch_overlap_compute_kind
            ),
            "attention_max_us": self.attention_max_us,
            "moe_max_us": self.moe_max_us,
            "selected_compute_ref_us": self.selected_compute_ref_us,
            "target_nic_us": self.target_nic_us,
            "weight_dispatch_max_us": self.weight_dispatch_max_us,
            "global_network_to_compute_ratio": (
                self.global_network_to_compute_ratio
            ),
            "global_adjustment_factor": self.global_adjustment_factor,
            "nic_theoretical_max_bytes": self.nic_theoretical_max_bytes,
            "observed_total_bytes_by_rank": list(
                self.observed_total_bytes_by_rank
            ),
            "bottleneck_observed_total_bytes": (
                self.bottleneck_observed_total_bytes
            ),
            "probed_total_nic_max_bytes": self.probed_total_nic_max_bytes,
            "effective_total_nic_max_bytes": (
                self.effective_total_nic_max_bytes
            ),
            "bottleneck_ranks": list(self.bottleneck_ranks),
            "budget_before": list(self.budget_before),
            "effective_rate_bytes_per_us_by_rank": list(
                self.effective_rate_bytes_per_us_by_rank
            ),
            "adjustment_factor_by_rank": list(
                self.adjustment_factor_by_rank
            ),
            "unclamped_target_budget_by_rank": list(
                self.unclamped_target_budget_by_rank
            ),
            "hard_migration_cap_by_rank": list(
                self.hard_migration_cap_by_rank
            ),
            "budget_after": list(self.budget_after),
            "dispatch_baseline_bytes_by_rank": list(
                self.dispatch_baseline_bytes_by_rank
            ),
            "dispatch_tx_bytes_by_rank": list(
                self.dispatch_tx_bytes_by_rank
            ),
            "dispatch_rx_bytes_by_rank": list(
                self.dispatch_rx_bytes_by_rank
            ),
            "migration_bytes_by_rank": list(self.migration_bytes_by_rank),
            "migration_bytes_semantics": "max_tx_rx_per_rank_full_duplex_endpoint",
            "migration_tx_bytes_by_rank": list(
                self.migration_tx_bytes_by_rank
            ),
            "migration_rx_bytes_by_rank": list(
                self.migration_rx_bytes_by_rank
            ),
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
        self._budgets_by_kind = {
            kind: [config.initial_budget_bytes] * num_ranks
            for kind in ("attention", "moe")
        }
        self._last_update_by_kind: dict[
            DispatchOverlapComputeKind, ProbeControllerUpdate | None
        ] = {"attention": None, "moe": None}

    def budgets_for(
        self, kind: DispatchOverlapComputeKind
    ) -> tuple[int, ...]:
        if kind not in {"attention", "moe"}:
            raise ValidationError("invalid ProbeEP compute window")
        return tuple(self._budgets_by_kind[kind])

    def last_update_for(
        self, kind: DispatchOverlapComputeKind
    ) -> ProbeControllerUpdate | None:
        if kind not in {"attention", "moe"}:
            raise ValidationError("invalid ProbeEP compute window")
        return self._last_update_by_kind[kind]

    def update(
        self, observation: ProbeDispatchFeedback
    ) -> ProbeControllerUpdate:
        observation.validate(self.num_ranks)
        kind = observation.dispatch_overlap_compute_kind
        before = self.budgets_for(kind)
        attention_max = max(
            observation.attention_compute_us_by_rank, default=0.0
        )
        moe_max = max(observation.moe_compute_us_by_rank, default=0.0)
        compute_ref = (
            attention_max
            if observation.dispatch_overlap_compute_kind == "attention"
            else moe_max
        )
        weight_dispatch_max = max(
            observation.weight_dispatch_us_by_rank, default=0.0
        )
        target_nic_us = self.config.target_overlap_ratio * compute_ref
        if weight_dispatch_max > 0 and compute_ref > 0:
            global_ratio = weight_dispatch_max / compute_ref
            global_factor = target_nic_us / weight_dispatch_max
        else:
            global_ratio = 0.0
            global_factor = 1.0
        theoretical_max = int(
            self.config.line_rate_bytes_per_us * compute_ref
        )
        hard_caps = tuple(
            max(0, theoretical_max - token_bytes)
            for token_bytes in observation.dispatch_baseline_bytes_by_rank
        )
        bottlenecks = (
            tuple(
                rank
                for rank, value in enumerate(
                    observation.weight_dispatch_us_by_rank
                )
                if abs(value - weight_dispatch_max) <= 1e-9
            )
            if weight_dispatch_max > 0 else ()
        )
        dispatch_tx = (
            observation.dispatch_tx_bytes_by_rank
            or observation.dispatch_baseline_bytes_by_rank
        )
        dispatch_rx = (
            observation.dispatch_rx_bytes_by_rank
            or observation.dispatch_baseline_bytes_by_rank
        )
        observed_totals = tuple(
            max(token_tx + migration_tx, token_rx + migration_rx)
            for token_tx, token_rx, migration_tx, migration_rx in zip(
                dispatch_tx,
                dispatch_rx,
                observation.migration_tx_bytes_by_rank,
                observation.migration_rx_bytes_by_rank,
            )
        )
        bottleneck_observed_total = max(
            (observed_totals[rank] for rank in bottlenecks),
            default=max(observed_totals, default=0),
        )
        measurement_valid = (
            weight_dispatch_max > 0
            and compute_ref > 0
            and bottleneck_observed_total > 0
        )
        probed_total_max = (
            max(0, int(bottleneck_observed_total * global_factor))
            if measurement_valid
            else max(
                (
                    dispatch_bytes + budget
                    for dispatch_bytes, budget in zip(
                        observation.dispatch_baseline_bytes_by_rank, before
                    )
                ),
                default=0,
            )
        )
        effective_total_max = min(probed_total_max, theoretical_max)

        effective_rates: list[float] = []
        adjustment_factors: list[float] = []
        targets: list[int] = []
        for rank in range(self.num_ranks):
            dispatch_bytes = observation.dispatch_baseline_bytes_by_rank[rank]
            total_bytes = observed_totals[rank]
            if weight_dispatch_max > 0 and total_bytes > 0 and compute_ref > 0:
                effective_rate = total_bytes / weight_dispatch_max
            else:
                effective_rate = 0.0
            if measurement_valid:
                target_budget = max(0, probed_total_max - dispatch_bytes)
            else:
                target_budget = before[rank]
            if before[rank] > 0:
                factor = target_budget / before[rank]
            else:
                # Keep manifest JSON finite when a disabled window is reopened.
                factor = 1.0 if target_budget == 0 else 0.0
            effective_rates.append(effective_rate)
            adjustment_factors.append(factor)
            targets.append(target_budget)

        after = [
            min(targets[rank], hard_caps[rank])
            for rank in range(self.num_ranks)
        ]
        if not measurement_valid:
            action = "hold"
        elif global_factor > 1.0:
            action = "ratio_increase"
        elif global_factor < 1.0:
            action = "ratio_decrease"
        else:
            action = "hold"
        self._budgets_by_kind[kind] = after
        update = ProbeControllerUpdate(
            observation_id=observation.observation_id,
            action=action,
            dispatch_overlap_compute_kind=(
                observation.dispatch_overlap_compute_kind
            ),
            attention_max_us=attention_max,
            moe_max_us=moe_max,
            selected_compute_ref_us=compute_ref,
            target_nic_us=target_nic_us,
            weight_dispatch_max_us=weight_dispatch_max,
            global_network_to_compute_ratio=global_ratio,
            global_adjustment_factor=global_factor,
            nic_theoretical_max_bytes=theoretical_max,
            observed_total_bytes_by_rank=observed_totals,
            bottleneck_observed_total_bytes=bottleneck_observed_total,
            probed_total_nic_max_bytes=probed_total_max,
            effective_total_nic_max_bytes=effective_total_max,
            bottleneck_ranks=bottlenecks,
            budget_before=before,
            effective_rate_bytes_per_us_by_rank=tuple(effective_rates),
            adjustment_factor_by_rank=tuple(adjustment_factors),
            unclamped_target_budget_by_rank=tuple(targets),
            hard_migration_cap_by_rank=hard_caps,
            budget_after=tuple(after),
            dispatch_baseline_bytes_by_rank=(
                observation.dispatch_baseline_bytes_by_rank
            ),
            dispatch_tx_bytes_by_rank=tuple(dispatch_tx),
            dispatch_rx_bytes_by_rank=tuple(dispatch_rx),
            migration_bytes_by_rank=observation.migration_bytes_by_rank,
            migration_tx_bytes_by_rank=(
                observation.migration_tx_bytes_by_rank
            ),
            migration_rx_bytes_by_rank=(
                observation.migration_rx_bytes_by_rank
            ),
            pending_migration_exists=observation.pending_migration_exists,
            feedback_source=observation.source,
        )
        self._last_update_by_kind[kind] = update
        return update


@dataclass(frozen=True)
class ProbeEPConfig:
    token_padding: int = 128
    chunk_tokens: int = 128
    route_chunk_tokens: int = 128
    weight_chunk_bytes: int = 4 * 1024 * 1024
    expert_weight_scale: float = 1.0
    overlap_expert_compute: bool = True
    payload_metadata_sample_limit: int = 8
    nic_controller: ProbeNICControllerConfig = ProbeNICControllerConfig()

    def __post_init__(self) -> None:
        if self.token_padding <= 0:
            raise ValidationError("token_padding must be positive")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")
        if self.route_chunk_tokens <= 0:
            raise ValidationError("route_chunk_tokens must be positive")
        if self.weight_chunk_bytes <= 0:
            raise ValidationError("weight_chunk_bytes must be positive")
        if (
            not isfinite(self.expert_weight_scale)
            or self.expert_weight_scale <= 0
        ):
            raise ValidationError("expert_weight_scale must be positive")
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
    anchor_rank: dict[tuple[int, int], int]
    local_prefetches: tuple[LocalReplicaTransfer, ...]


@dataclass(frozen=True)
class ProbeEPPlan:
    invocation_index: int
    dispatch_overlap_compute_kind: DispatchOverlapComputeKind
    attention_compute_us_by_rank: tuple[float, ...]
    moe_compute_us_by_rank: tuple[float, ...]
    selected_compute_ref_us: float
    nic_theoretical_max_bytes: int
    execution_rank: dict[RouteKey, int]
    planned_intents: tuple[MigrationIntent, ...]
    admitted_intents: tuple[MigrationIntent, ...]
    deferred_intents: tuple[MigrationIntent, ...]
    remote_replicas: tuple[RemoteReplicaPlan, ...]
    local_prefetches: tuple[LocalReplicaTransfer, ...]
    baseline_server_routes: dict[int, int]
    server_target_routes: dict[int, int]
    donor_surplus_routes: dict[int, int]
    receiver_deficit_routes: dict[int, int]
    planned_server_routes: dict[int, int]
    admitted_server_routes: dict[int, int]
    baseline_server_padded_routes: dict[int, int]
    planned_server_padded_routes: dict[int, int]
    admitted_server_padded_routes: dict[int, int]
    baseline_real_routes_by_rank: dict[int, int]
    final_real_routes_by_rank: dict[int, int]
    baseline_padded_routes_by_rank: dict[int, int]
    final_padded_routes_by_rank: dict[int, int]
    baseline_compute_us_by_rank: dict[int, float]
    final_compute_us_by_rank: dict[int, float]
    nic_budget_before: tuple[int, ...]
    hard_migration_cap_by_rank: tuple[int, ...]
    effective_nic_budget_by_rank: tuple[int, ...]
    assigned_migration_tx_bytes: tuple[int, ...]
    assigned_migration_rx_bytes: tuple[int, ...]
    assigned_migration_bytes_by_server_pair: dict[
        tuple[int, int], tuple[int, ...]
    ]
    predicted_dispatch_tx_bytes: tuple[int, ...]
    predicted_dispatch_rx_bytes: tuple[int, ...]
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
        self._next_invocation_index = 0

    @property
    def controller(self) -> ProbeNICController | None:
        return self._controller

    def apply_feedback(
        self, observation: ProbeDispatchFeedback
    ) -> ProbeControllerUpdate:
        if self._controller is None:
            raise ValidationError(
                "ProbeEP controller is initialized by the first plan/build"
            )
        return self._controller.update(observation)

    def plan(
        self,
        invocation: MoEInvocation,
        *,
        dispatch_overlap_compute_kind: DispatchOverlapComputeKind = "moe",
        attention_compute_us_by_rank: tuple[float, ...] | None = None,
        moe_compute_us_by_rank: tuple[float, ...] | None = None,
        nic_budget_override: tuple[int, ...] | None = None,
    ) -> ProbeEPPlan:
        placement = invocation.placement
        controller = self._ensure_controller(placement.num_ranks)
        if dispatch_overlap_compute_kind not in {"attention", "moe"}:
            raise ValidationError("invalid ProbeEP Dispatch overlap compute kind")
        selected_budgets = controller.budgets_for(
            dispatch_overlap_compute_kind
        )
        if nic_budget_override is not None:
            if len(nic_budget_override) != placement.num_ranks:
                raise ValidationError(
                    "ProbeEP NIC budget override length must equal num_ranks"
                )
            if any(value < 0 for value in nic_budget_override):
                raise ValidationError(
                    "ProbeEP NIC budget overrides must be non-negative"
                )
            selected_budgets = tuple(nic_budget_override)
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
        baseline_padded, baseline_compute = self._compute_state(
            invocation, assignments_by_key, baseline_intra.execution_rank
        )
        attention_times = self._compute_reference_times(
            "attention",
            attention_compute_us_by_rank,
            placement.num_ranks,
            fallback=baseline_compute,
        )
        moe_times = self._compute_reference_times(
            "moe",
            moe_compute_us_by_rank,
            placement.num_ranks,
            fallback=baseline_compute,
        )
        compute_ref = max(
            attention_times
            if dispatch_overlap_compute_kind == "attention"
            else moe_times,
            default=0.0,
        )
        nic_theoretical_max_bytes = int(
            controller.config.line_rate_bytes_per_us * compute_ref
        )
        (
            planned_server_by_route,
            planned_intents,
            server_targets,
            donor_surplus,
            receiver_deficit,
        ) = self._plan_inter_server(
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
            assigned_by_server_pair,
        ) = self._admit_intents(
            invocation,
            home_server_by_route,
            planned_intents,
            selected_budgets,
            nic_theoretical_max_bytes,
        )
        final_intra = self._plan_intra_server(
            invocation, assignments_by_key, admitted_server_by_route
        )
        final_padded, final_compute = self._compute_state(
            invocation, assignments_by_key, final_intra.execution_rank
        )
        predicted_dispatch_tx, predicted_dispatch_rx = (
            self._predicted_dispatch_nic_bytes(
                invocation, admitted_server_by_route
            )
        )
        dispatch_baseline = tuple(
            max(predicted_dispatch_tx[rank], predicted_dispatch_rx[rank])
            for rank in range(placement.num_ranks)
        )
        hard_caps = tuple(
            max(0, nic_theoretical_max_bytes - dispatch_baseline[rank])
            for rank in range(placement.num_ranks)
        )
        effective_budgets = tuple(
            min(selected_budgets[rank], hard_caps[rank])
            for rank in range(placement.num_ranks)
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
            invocation_index=self._next_invocation_index,
            dispatch_overlap_compute_kind=dispatch_overlap_compute_kind,
            attention_compute_us_by_rank=attention_times,
            moe_compute_us_by_rank=moe_times,
            selected_compute_ref_us=compute_ref,
            nic_theoretical_max_bytes=nic_theoretical_max_bytes,
            execution_rank=final_intra.execution_rank,
            planned_intents=planned_intents,
            admitted_intents=admitted_intents,
            deferred_intents=deferred_intents,
            remote_replicas=tuple(remote_replicas),
            local_prefetches=final_intra.local_prefetches,
            baseline_server_routes=self._server_route_counts(
                placement, home_server_by_route
            ),
            server_target_routes=server_targets,
            donor_surplus_routes=donor_surplus,
            receiver_deficit_routes=receiver_deficit,
            planned_server_routes=self._server_route_counts(
                placement, planned_server_by_route
            ),
            admitted_server_routes=self._server_route_counts(
                placement, admitted_server_by_route
            ),
            baseline_server_padded_routes=self._server_padded_route_counts(
                invocation, assignments_by_key, home_server_by_route
            ),
            planned_server_padded_routes=self._server_padded_route_counts(
                invocation, assignments_by_key, planned_server_by_route
            ),
            admitted_server_padded_routes=self._server_padded_route_counts(
                invocation, assignments_by_key, admitted_server_by_route
            ),
            baseline_real_routes_by_rank=baseline_intra.real_routes_by_rank,
            final_real_routes_by_rank=final_intra.real_routes_by_rank,
            baseline_padded_routes_by_rank=baseline_padded,
            final_padded_routes_by_rank=final_padded,
            baseline_compute_us_by_rank=baseline_compute,
            final_compute_us_by_rank=final_compute,
            nic_budget_before=selected_budgets,
            hard_migration_cap_by_rank=hard_caps,
            effective_nic_budget_by_rank=effective_budgets,
            assigned_migration_tx_bytes=tuple(assigned_tx),
            assigned_migration_rx_bytes=tuple(assigned_rx),
            assigned_migration_bytes_by_server_pair=(
                assigned_by_server_pair
            ),
            predicted_dispatch_tx_bytes=predicted_dispatch_tx,
            predicted_dispatch_rx_bytes=predicted_dispatch_rx,
            controller_last_update=controller.last_update_for(
                dispatch_overlap_compute_kind
            ),
        )

    def build(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        *,
        entry_keys: set[str] | None = None,
        dispatch_overlap_compute_kind: DispatchOverlapComputeKind = "moe",
        attention_compute_us_by_rank: tuple[float, ...] | None = None,
        moe_compute_us_by_rank: tuple[float, ...] | None = None,
        nic_budget_override: tuple[int, ...] | None = None,
    ) -> AlgorithmBuildResult:
        if graph.num_ranks != invocation.placement.num_ranks:
            raise ValidationError("graph and placement rank counts differ")
        roots = set(entry_keys or ())
        before = len(graph.tasks)
        plan = self.plan(
            invocation,
            dispatch_overlap_compute_kind=dispatch_overlap_compute_kind,
            attention_compute_us_by_rank=attention_compute_us_by_rank,
            moe_compute_us_by_rank=moe_compute_us_by_rank,
            nic_budget_override=nic_budget_override,
        )
        placement = invocation.placement

        remote_ready_by_replica: dict[tuple[int, int], set[str]] = defaultdict(set)
        remote_weight_tx_by_rank: dict[int, set[str]] = defaultdict(set)
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
                            plan.invocation_index,
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
                        plan.invocation_index, replica, chunk, "same_rail_rdma"
                    ),
                )
                remote_weight_tx_by_rank[chunk.source_relay].add(rdma_key)
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
                            plan.invocation_index,
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
                f"dst{prefetch.destination_rank}.transfer{index}"
            )
            graph.add_transfer(
                key,
                prefetch.source_rank,
                prefetch.destination_rank,
                self._expert_weight_bytes(invocation),
                "expert_weight_prefetch",
                f"{invocation.invocation_id}:probe:local_prefetch",
                predecessors=predecessors,
                metadata={
                    "algorithm": "probeep",
                    "invocation_index": plan.invocation_index,
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
        dispatch_start = len(graph.tasks)
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
        for task in graph.tasks[dispatch_start:]:
            if task.metadata.get("hierarchical_leg") != "dispatch_fabric":
                continue
            assert task.src_rank is not None
            for weight_tx_key in remote_weight_tx_by_rank[task.src_rank]:
                graph.add_dependency(task.key, weight_tx_key)
            task.metadata["weight_dispatch_ordering"] = (
                "same_source_rail_remote_weight_tx"
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
                    "invocation_index": plan.invocation_index,
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
                    "invocation_index": plan.invocation_index,
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
        replicas_sent = {server: 0 for server in range(placement.num_servers)}
        replicas_received = {
            server: 0 for server in range(placement.num_servers)
        }
        weight_bytes_sent = {
            server: 0 for server in range(placement.num_servers)
        }
        weight_bytes_received = {
            server: 0 for server in range(placement.num_servers)
        }
        for replica in plan.remote_replicas:
            weight_bytes = sum(
                chunk.transfer_bytes for chunk in replica.weight_chunks
            )
            replicas_sent[replica.source_server] += 1
            replicas_received[replica.destination_server] += 1
            weight_bytes_sent[replica.source_server] += weight_bytes
            weight_bytes_received[replica.destination_server] += weight_bytes
        result = AlgorithmBuildResult(
            algorithm="probeep_deepep_hierarchical",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "invocation_index": plan.invocation_index,
                "planner": "global_server_first_padding_aware_v5",
                "planning_mode": "oracle_current_routes",
                "planner_runtime_model": "not_in_dag",
                "planner_complexity_scope": "paper_analysis_only",
                "planner_decision_input": "server_expert_histogram",
                "planner_search_complexity": {
                    "inter_server": "O(E log E + M + I*S^2*E)",
                    "intra_server": "O(E log E + B*G)",
                    "controller": "O(P)",
                    "chunk_assignment": "O(C * G)",
                },
                "route_lowering_runtime_model": "offline_not_in_dag",
                "feedback_mode": (
                    "in_process_dynamic_dispatch_observation"
                    if nic_budget_override is not None
                    else "controller_initial_state"
                ),
                "controlled_communication_phase": "dispatch",
                "control_observation_scope": "expert_weight_plus_dispatch",
                "combine_controller_role": "telemetry_only",
                "nic_budget_source": (
                    "in_process_dynamic_budget_override"
                    if nic_budget_override is not None
                    else "controller_state"
                ),
                "controller_last_update": controller_update,
                "dispatch_overlap_compute_kind": (
                    plan.dispatch_overlap_compute_kind
                ),
                "attention_compute_us_by_rank": list(
                    plan.attention_compute_us_by_rank
                ),
                "moe_compute_us_by_rank": list(
                    plan.moe_compute_us_by_rank
                ),
                "attention_max_us": max(
                    plan.attention_compute_us_by_rank, default=0.0
                ),
                "moe_max_us": max(plan.moe_compute_us_by_rank, default=0.0),
                "selected_compute_ref_us": plan.selected_compute_ref_us,
                "nic_line_rate_gbps": (
                    self._controller.config.nic_line_rate_gbps
                    if self._controller is not None
                    else self.config.nic_controller.nic_line_rate_gbps
                ),
                "target_overlap_ratio": (
                    self._controller.config.target_overlap_ratio
                    if self._controller is not None
                    else self.config.nic_controller.target_overlap_ratio
                ),
                "nic_theoretical_max_bytes": (
                    plan.nic_theoretical_max_bytes
                ),
                "nic_budget_before": list(plan.nic_budget_before),
                "hard_migration_cap_by_rank": list(
                    plan.hard_migration_cap_by_rank
                ),
                "effective_nic_budget_by_rank": list(
                    plan.effective_nic_budget_by_rank
                ),
                "assigned_migration_tx_bytes": list(
                    plan.assigned_migration_tx_bytes
                ),
                "assigned_migration_rx_bytes": list(
                    plan.assigned_migration_rx_bytes
                ),
                "assigned_migration_bytes_by_server_pair": [
                    {
                        "source_server": source_server,
                        "destination_server": destination_server,
                        "bytes_by_rail": list(bytes_by_rail),
                        "max_min_spread_bytes": (
                            max(bytes_by_rail, default=0)
                            - min(bytes_by_rail, default=0)
                        ),
                    }
                    for (
                        source_server,
                        destination_server,
                    ), bytes_by_rail in sorted(
                        plan.assigned_migration_bytes_by_server_pair.items()
                    )
                ],
                "predicted_dispatch_tx_bytes": list(
                    plan.predicted_dispatch_tx_bytes
                ),
                "predicted_dispatch_rx_bytes": list(
                    plan.predicted_dispatch_rx_bytes
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
                "server_target_routes": plan.server_target_routes,
                "donor_surplus_routes": plan.donor_surplus_routes,
                "receiver_deficit_routes": plan.receiver_deficit_routes,
                "planned_server_routes": plan.planned_server_routes,
                "admitted_server_routes": plan.admitted_server_routes,
                "baseline_server_padded_routes": (
                    plan.baseline_server_padded_routes
                ),
                "planned_server_padded_routes": (
                    plan.planned_server_padded_routes
                ),
                "admitted_server_padded_routes": (
                    plan.admitted_server_padded_routes
                ),
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
                "remote_replicas_sent_by_server": replicas_sent,
                "remote_replicas_received_by_server": replicas_received,
                "remote_weight_bytes_sent_by_server": weight_bytes_sent,
                "remote_weight_bytes_received_by_server": (
                    weight_bytes_received
                ),
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
                "expert_weight_scale": self.config.expert_weight_scale,
                "effective_expert_weight_bytes": self._expert_weight_bytes(
                    invocation
                ),
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
                "token_server_pair_transport": (
                    hierarchical_token_server_pair_profile(
                        invocation, payload_plan
                    )
                ),
                "remote_weight_rdma_bytes": transfer_bytes.get(
                    "expert_weight_rdma", 0
                ),
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
                "expert_load_profile": expert_load_profile,
            },
        )
        self._next_invocation_index += 1
        return result

    def _plan_inter_server(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        home_server_by_route: dict[RouteKey, int],
    ) -> tuple[
        dict[RouteKey, int],
        tuple[MigrationIntent, ...],
        dict[int, int],
        dict[int, int],
        dict[int, int],
    ]:
        placement = invocation.placement
        route_server = dict(home_server_by_route)
        routes_by_server_expert: dict[
            tuple[int, int], deque[RouteKey]
        ] = defaultdict(deque)
        for route_key, assignment in assignments_by_key.items():
            routes_by_server_expert[
                (home_server_by_route[route_key], assignment.expert_id)
            ].append(route_key)
        baseline_load = self._server_route_counts(placement, route_server)
        total_routes = len(assignments_by_key)
        target_floor, extra = divmod(total_routes, placement.num_servers)
        extra_servers = set(
            sorted(
                range(placement.num_servers),
                key=lambda server: (-baseline_load[server], server),
            )[:extra]
        )
        targets = {
            server: target_floor + (1 if server in extra_servers else 0)
            for server in range(placement.num_servers)
        }
        initial_surplus = {
            server: max(0, baseline_load[server] - targets[server])
            for server in range(placement.num_servers)
        }
        initial_deficit = {
            server: max(0, targets[server] - baseline_load[server])
            for server in range(placement.num_servers)
        }
        surplus = dict(initial_surplus)
        deficit = dict(initial_deficit)
        groups = sorted(
            (
                (source, expert, routes)
                for (source, expert), routes in routes_by_server_expert.items()
                if surplus[source] > 0 and routes
            ),
            key=lambda item: (-len(item[2]), item[0], item[1]),
        )
        for source, expert, routes in groups:
            while surplus[source] > 0 and routes:
                receivers = [
                    server
                    for server in range(placement.num_servers)
                    if deficit[server] > 0 and server != source
                ]
                if not receivers:
                    break
                destination = min(
                    receivers,
                    key=lambda server: (-deficit[server], server),
                )
                while (
                    surplus[source] > 0
                    and deficit[destination] > 0
                    and routes
                ):
                    move_count = min(
                        self.config.route_chunk_tokens,
                        surplus[source],
                        deficit[destination],
                        len(routes),
                    )
                    moved = tuple(routes.popleft() for _ in range(move_count))
                    for route_key in moved:
                        route_server[route_key] = destination
                    surplus[source] -= move_count
                    deficit[destination] -= move_count

        if any(surplus.values()) or any(deficit.values()):
            raise ValidationError(
                "ProbeEP global quota negotiation did not converge: "
                f"surplus={surplus}, deficit={deficit}"
            )

        self._refine_inter_server_padding(
            invocation,
            assignments_by_key,
            route_server,
        )

        # Lower the final mapping directly into independent home->destination
        # intents. Intermediate quota/refinement moves are planner internals.
        aggregated: dict[tuple[int, int, int], list[RouteKey]] = defaultdict(list)
        for route_key, destination in route_server.items():
            source = home_server_by_route[route_key]
            if destination == source:
                continue
            expert = assignments_by_key[route_key].expert_id
            aggregated[(expert, source, destination)].append(route_key)
        ordered_moves = sorted(
            aggregated.items(),
            key=lambda item: (
                -len(item[1]),
                item[0][1],
                item[0][2],
                item[0][0],
            ),
        )
        per_route_us = self._expert_route_us(invocation)
        intents = tuple(
            MigrationIntent(
                intent_id=index,
                priority=index,
                expert_id=key[0],
                source_server=key[1],
                destination_server=key[2],
                moved_route_keys=tuple(route_keys),
                moved_compute_us=len(route_keys) * per_route_us,
            )
            for index, (key, route_keys) in enumerate(ordered_moves)
        )
        return (
            route_server,
            intents,
            targets,
            initial_surplus,
            initial_deficit,
        )

    def _refine_inter_server_padding(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        route_server: dict[RouteKey, int],
    ) -> None:
        """Reduce the global max server padding proxy after raw-route quota."""
        padding = self.config.token_padding
        placement = invocation.placement
        if padding <= 1 or placement.num_servers <= 1:
            return

        routes_by_server_expert: dict[tuple[int, int], list[RouteKey]] = (
            defaultdict(list)
        )
        for route_key, assignment in assignments_by_key.items():
            routes_by_server_expert[
                (route_server[route_key], assignment.expert_id)
            ].append(route_key)

        padded = self._server_padded_route_counts(
            invocation, assignments_by_key, route_server
        )
        experts = sorted({item.expert_id for item in assignments_by_key.values()})
        max_steps = max(1, placement.num_servers * len(experts))
        for _ in range(max_steps):
            current_max = max(padded.values(), default=0)
            current_spread = current_max - min(padded.values(), default=0)
            if current_spread <= padding:
                break

            best: tuple[
                tuple[int, int, int, int, int, int, int],
                int,
                int,
                int,
                int,
            ] | None = None
            for donor in range(placement.num_servers):
                for receiver in range(placement.num_servers):
                    if donor == receiver:
                        continue
                    for expert in experts:
                        donor_routes = routes_by_server_expert[(donor, expert)]
                        donor_count = len(donor_routes)
                        if donor_count == 0:
                            continue
                        # Minimum route movement that removes one padded block
                        # from this (server, expert) group.
                        move_count = donor_count - (ceil(donor_count / padding) - 1) * padding
                        receiver_count = len(
                            routes_by_server_expert[(receiver, expert)]
                        )
                        receiver_before = (
                            ceil(receiver_count / padding) * padding
                            if receiver_count
                            else 0
                        )
                        receiver_after = (
                            ceil((receiver_count + move_count) / padding) * padding
                        )
                        receiver_increase = receiver_after - receiver_before
                        candidate_padded = dict(padded)
                        candidate_padded[donor] -= padding
                        candidate_padded[receiver] += receiver_increase
                        candidate_max = max(candidate_padded.values())
                        candidate_spread = candidate_max - min(
                            candidate_padded.values()
                        )
                        if (candidate_max, candidate_spread) >= (
                            current_max,
                            current_spread,
                        ):
                            continue
                        score = (
                            candidate_max,
                            candidate_spread,
                            receiver_increase,
                            0 if receiver_count else 1,
                            move_count,
                            donor,
                            receiver,
                        )
                        candidate = (
                            score,
                            donor,
                            receiver,
                            expert,
                            move_count,
                        )
                        if best is None or candidate[0] < best[0]:
                            best = candidate
            if best is None:
                break
            _, donor, receiver, expert, move_count = best
            donor_routes = routes_by_server_expert[(donor, expert)]
            receiver_routes = routes_by_server_expert[(receiver, expert)]
            receiver_count = len(receiver_routes)
            receiver_before = (
                ceil(receiver_count / padding) * padding
                if receiver_count
                else 0
            )
            receiver_after = (
                ceil((receiver_count + move_count) / padding) * padding
            )
            moved = donor_routes[-move_count:]
            del donor_routes[-move_count:]
            receiver_routes.extend(moved)
            for route_key in moved:
                route_server[route_key] = receiver
            padded[donor] -= padding
            padded[receiver] += receiver_after - receiver_before

    def _admit_intents(
        self,
        invocation: MoEInvocation,
        home_server_by_route: dict[RouteKey, int],
        intents: tuple[MigrationIntent, ...],
        budgets: tuple[int, ...],
        nic_theoretical_max_bytes: int,
    ) -> tuple[
        dict[RouteKey, int],
        tuple[MigrationIntent, ...],
        tuple[MigrationIntent, ...],
        dict[tuple[int, int], tuple[WeightChunkPlan, ...]],
        list[int],
        list[int],
        dict[tuple[int, int], tuple[int, ...]],
    ]:
        route_server = dict(home_server_by_route)
        admitted: list[MigrationIntent] = []
        deferred: list[MigrationIntent] = []
        chunks_by_replica: dict[
            tuple[int, int], tuple[WeightChunkPlan, ...]
        ] = {}
        assigned_tx = [0] * invocation.placement.num_ranks
        assigned_rx = [0] * invocation.placement.num_ranks
        assigned_by_server_pair: dict[tuple[int, int], tuple[int, ...]] = {}
        server_expert_counts: dict[tuple[int, int], int] = defaultdict(int)
        for assignment in invocation.assignments:
            route_key = self._route_key(assignment)
            server_expert_counts[
                (home_server_by_route[route_key], assignment.expert_id)
            ] += 1
        current_server_padded = {
            server: 0
            for server in range(invocation.placement.num_servers)
        }
        for (server, _expert), count in server_expert_counts.items():
            current_server_padded[server] += (
                ceil(count / self.config.token_padding)
                * self.config.token_padding
            )
        for intent in sorted(intents, key=lambda item: (item.priority, item.intent_id)):
            source_key = (intent.source_server, intent.expert_id)
            destination_key = (intent.destination_server, intent.expert_id)
            moved_count = len(intent.moved_route_keys)
            source_count = server_expert_counts[source_key]
            destination_count = server_expert_counts[destination_key]
            if moved_count > source_count:
                raise ValidationError(
                    "ProbeEP intent moves more routes than its current source "
                    f"group: intent={intent.intent_id}"
                )
            candidate_server_padded = dict(current_server_padded)
            candidate_server_padded[intent.source_server] += (
                ceil((source_count - moved_count) / self.config.token_padding)
                * self.config.token_padding
                - ceil(source_count / self.config.token_padding)
                * self.config.token_padding
            )
            candidate_server_padded[intent.destination_server] += (
                ceil((destination_count + moved_count) / self.config.token_padding)
                * self.config.token_padding
                - (
                    ceil(destination_count / self.config.token_padding)
                    * self.config.token_padding
                    if destination_count
                    else 0
                )
            )
            current_objective = (
                max(current_server_padded.values()),
                max(current_server_padded.values())
                - min(current_server_padded.values()),
            )
            candidate_objective = (
                max(candidate_server_padded.values()),
                max(candidate_server_padded.values())
                - min(candidate_server_padded.values()),
            )
            if candidate_objective >= current_objective:
                deferred.append(intent)
                continue
            replica_key = (intent.expert_id, intent.destination_server)
            if replica_key not in chunks_by_replica:
                candidate_route_server = dict(route_server)
                for route_key in intent.moved_route_keys:
                    candidate_route_server[route_key] = intent.destination_server
                dispatch_tx, dispatch_rx = self._predicted_dispatch_nic_bytes(
                    invocation, candidate_route_server
                )
                if any(
                    assigned_tx[rank]
                    > min(
                        budgets[rank],
                        max(
                            0,
                            nic_theoretical_max_bytes - dispatch_tx[rank],
                        ),
                    )
                    or assigned_rx[rank]
                    > min(
                        budgets[rank],
                        max(
                            0,
                            nic_theoretical_max_bytes - dispatch_rx[rank],
                        ),
                    )
                    for rank in range(invocation.placement.num_ranks)
                ):
                    deferred.append(intent)
                    continue
                schedule = self._schedule_weight_chunks(
                    invocation,
                    intent.source_server,
                    intent.destination_server,
                    budgets,
                    nic_theoretical_max_bytes,
                    assigned_tx,
                    assigned_rx,
                    dispatch_tx,
                    dispatch_rx,
                    assigned_by_server_pair.get(
                        (intent.source_server, intent.destination_server),
                        (0,) * invocation.placement.gpus_per_server,
                    ),
                )
                if schedule is None:
                    deferred.append(intent)
                    continue
                chunks, next_tx, next_rx, next_pair_load = schedule
                chunks_by_replica[replica_key] = chunks
                assigned_tx = next_tx
                assigned_rx = next_rx
                assigned_by_server_pair[
                    (intent.source_server, intent.destination_server)
                ] = next_pair_load
            admitted.append(intent)
            for route_key in intent.moved_route_keys:
                route_server[route_key] = intent.destination_server
            server_expert_counts[source_key] -= moved_count
            server_expert_counts[destination_key] += moved_count
            current_server_padded = candidate_server_padded
        return (
            route_server,
            tuple(admitted),
            tuple(deferred),
            chunks_by_replica,
            assigned_tx,
            assigned_rx,
            assigned_by_server_pair,
        )

    def _schedule_weight_chunks(
        self,
        invocation: MoEInvocation,
        source_server: int,
        destination_server: int,
        budgets: tuple[int, ...],
        nic_theoretical_max_bytes: int,
        assigned_tx: list[int],
        assigned_rx: list[int],
        dispatch_tx: tuple[int, ...],
        dispatch_rx: tuple[int, ...],
        pair_load: tuple[int, ...],
    ) -> tuple[
        tuple[WeightChunkPlan, ...],
        list[int],
        list[int],
        tuple[int, ...],
    ] | None:
        placement = invocation.placement
        if len(pair_load) != placement.gpus_per_server:
            raise ValidationError("ProbeEP server-pair load length is invalid")
        next_tx = list(assigned_tx)
        next_rx = list(assigned_rx)
        next_pair_load = list(pair_load)
        chunks: list[WeightChunkPlan] = []
        remaining = self._expert_weight_bytes(invocation)
        offset = 0
        while remaining:
            chunk_bytes = min(self.config.weight_chunk_bytes, remaining)
            candidates: list[tuple[int, int, int, int, int, int]] = []
            for rail in range(placement.gpus_per_server):
                source_rank = placement.server_rank(source_server, rail)
                destination_rank = placement.server_rank(
                    destination_server, rail
                )
                source_hard_cap = max(
                    0,
                    nic_theoretical_max_bytes
                    - dispatch_tx[source_rank],
                )
                destination_hard_cap = max(
                    0,
                    nic_theoretical_max_bytes
                    - dispatch_rx[destination_rank],
                )
                source_budget = min(budgets[source_rank], source_hard_cap)
                destination_budget = min(
                    budgets[destination_rank], destination_hard_cap
                )
                if (
                    next_tx[source_rank] + chunk_bytes > source_budget
                    or next_rx[destination_rank] + chunk_bytes
                    > destination_budget
                    or source_budget == 0
                    or destination_budget == 0
                ):
                    continue
                projected_tx = (
                    dispatch_tx[source_rank]
                    + next_tx[source_rank]
                    + chunk_bytes
                )
                projected_rx = (
                    dispatch_rx[destination_rank]
                    + next_rx[destination_rank]
                    + chunk_bytes
                )
                candidates.append(
                    (
                        next_pair_load[rail] + chunk_bytes,
                        max(projected_tx, projected_rx),
                        projected_tx + projected_rx,
                        rail,
                        source_rank,
                        destination_rank,
                    )
                )
            if not candidates:
                return None
            _, _, _, rail, source_rank, destination_rank = min(candidates)
            next_tx[source_rank] += chunk_bytes
            next_rx[destination_rank] += chunk_bytes
            next_pair_load[rail] += chunk_bytes
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
        return tuple(chunks), next_tx, next_rx, tuple(next_pair_load)

    def _predicted_dispatch_nic_bytes(
        self,
        invocation: MoEInvocation,
        route_server: dict[RouteKey, int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Predict the deduplicated Dispatch fabric bytes sharing the weight window."""
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
            total_blocks = sum(
                ceil(len(routes) / self.config.token_padding)
                for routes in server_groups.values()
            )
            base_blocks, extra_blocks = divmod(
                total_blocks, placement.gpus_per_server
            )
            remaining_blocks = {
                rank: base_blocks + (1 if local < extra_blocks else 0)
                for local, rank in enumerate(ranks)
            }
            ordered_experts = sorted(
                server_groups,
                key=lambda expert: (-len(server_groups[expert]), expert),
            )
            for expert in ordered_experts:
                home_rank = placement.expert_rank(expert)
                home_server = placement.rank_server(home_rank)
                routes = server_groups[expert]
                route_chunks = [
                    routes[offset : offset + self.config.token_padding]
                    for offset in range(0, len(routes), self.config.token_padding)
                ]
                next_chunk = 0
                first_execution_rank: int | None = None
                while next_chunk < len(route_chunks):
                    candidates = [
                        rank for rank in ranks if remaining_blocks[rank] > 0
                    ]
                    if not candidates:
                        raise ValidationError(
                            "ProbeEP intra-server block packing exhausted capacity "
                            f"for expert {expert} on server {server}"
                        )
                    rank = min(
                        candidates,
                        key=lambda item: (
                            -remaining_blocks[item],
                            0 if item == home_rank else 1,
                            item,
                        ),
                    )
                    if first_execution_rank is None:
                        first_execution_rank = rank
                    take = min(
                        remaining_blocks[rank], len(route_chunks) - next_chunk
                    )
                    instances[(expert, server)].add(rank)
                    for route_chunk in route_chunks[next_chunk : next_chunk + take]:
                        for route_key in route_chunk:
                            execution[route_key] = rank
                            real_routes[rank] += 1
                    remaining_blocks[rank] -= take
                    next_chunk += take
                assert first_execution_rank is not None
                anchor_rank[(expert, server)] = (
                    home_rank if home_server == server else first_execution_rank
                )
            if any(remaining_blocks.values()):
                raise ValidationError(
                    "ProbeEP intra-server block packing left unused capacity on "
                    f"server {server}: {remaining_blocks}"
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

    def _expert_weight_bytes(self, invocation: MoEInvocation) -> int:
        return max(
            1,
            int(ceil(
                invocation.expert_weight_bytes
                * self.config.expert_weight_scale
            )),
        )

    @staticmethod
    def _compute_reference_times(
        kind: DispatchOverlapComputeKind,
        values: tuple[float, ...] | None,
        num_ranks: int,
        *,
        fallback: dict[int, float],
    ) -> tuple[float, ...]:
        if values is None:
            if kind == "attention":
                return (0.0,) * num_ranks
            return tuple(fallback[rank] for rank in range(num_ranks))
        if len(values) != num_ranks:
            raise ValidationError(
                f"ProbeEP {kind} compute reference needs {num_ranks} entries"
            )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValidationError(
                f"ProbeEP {kind} compute reference must be finite and non-negative"
            )
        return tuple(values)

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

    def _server_padded_route_counts(
        self,
        invocation: MoEInvocation,
        assignments_by_key: dict[RouteKey, RoutingAssignment],
        route_server: dict[RouteKey, int],
    ) -> dict[int, int]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for route_key, assignment in assignments_by_key.items():
            counts[(route_server[route_key], assignment.expert_id)] += 1
        result = {
            server: 0 for server in range(invocation.placement.num_servers)
        }
        for (server, _expert), count in counts.items():
            result[server] += (
                ceil(count / self.config.token_padding)
                * self.config.token_padding
            )
        return result

    @staticmethod
    def _weight_metadata(
        invocation_index: int,
        replica: RemoteReplicaPlan,
        chunk: WeightChunkPlan,
        leg: str,
    ) -> dict[str, object]:
        return {
            "algorithm": "probeep",
            "invocation_index": invocation_index,
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
