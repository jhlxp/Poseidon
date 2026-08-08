from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import struct
from typing import Protocol

from .schema import Placement, RoutingAssignment, ValidationError


@dataclass(frozen=True)
class GateSample:
    assignments: tuple[RoutingAssignment, ...]
    metadata: dict[str, object]


class GateProvider(Protocol):
    name: str

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample: ...


def _imbalance(values: list[int]) -> dict[str, float | int]:
    total = sum(values)
    mean = total / len(values) if values else 0.0
    maximum = max(values, default=0)
    minimum = min(values, default=0)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "max_mean": maximum / mean if mean else 0.0,
    }


def make_gate_sample(
    assignments: tuple[RoutingAssignment, ...],
    *,
    placement: Placement,
    provider_name: str,
    provider_parameters: dict[str, object],
    seed: int,
    routing_fidelity: str,
    target_expert_weights: tuple[float, ...] | None = None,
) -> GateSample:
    if not provider_name:
        raise ValidationError("gate provider name must not be empty")
    if not routing_fidelity:
        raise ValidationError("gate routing_fidelity must not be empty")

    logical_loads = [0] * placement.num_experts
    rank_loads = [0] * placement.num_ranks
    server_loads = [0] * placement.num_servers
    source_to_rank = [
        [0] * placement.num_ranks for _ in range(placement.num_ranks)
    ]
    source_to_server = [
        [0] * placement.num_servers for _ in range(placement.num_ranks)
    ]
    digest = hashlib.sha256()
    for assignment in assignments:
        rank = placement.expert_rank(assignment.expert_id)
        server = placement.rank_server(rank)
        logical_loads[assignment.expert_id] += 1
        rank_loads[rank] += 1
        server_loads[server] += 1
        source_to_rank[assignment.src_rank][rank] += 1
        source_to_server[assignment.src_rank][server] += 1
        digest.update(
            struct.pack(
                "<IIIId",
                assignment.src_rank,
                assignment.token_id,
                assignment.topk_slot,
                assignment.expert_id,
                assignment.route_weight,
            )
        )

    normalized_target: list[float] | None = None
    distance: float | None = None
    if target_expert_weights is not None:
        if len(target_expert_weights) != placement.num_experts:
            raise ValidationError(
                "gate target expert weight count must equal logical expert count"
            )
        if any(not math.isfinite(value) or value < 0 for value in target_expert_weights):
            raise ValidationError("gate target expert weights must be finite and non-negative")
        target_total = sum(target_expert_weights)
        if target_total <= 0:
            raise ValidationError("gate target expert weights must contain positive mass")
        normalized_target = [value / target_total for value in target_expert_weights]
        realized_total = sum(logical_loads)
        if realized_total:
            distance = 0.5 * sum(
                abs(target - realized / realized_total)
                for target, realized in zip(normalized_target, logical_loads)
            )

    return GateSample(
        assignments=assignments,
        metadata={
            "schema": "gate_sample_v1",
            "name": provider_name,
            "parameters": provider_parameters,
            "seed": seed,
            "routing_fidelity": routing_fidelity,
            "assignment_digest_sha256": digest.hexdigest(),
            "logical_route_count": len(assignments),
            "logical_expert_loads": logical_loads,
            "baseline_rank_loads": rank_loads,
            "baseline_server_loads": server_loads,
            "source_to_rank_route_matrix": source_to_rank,
            "source_to_server_route_matrix": source_to_server,
            "rank_imbalance": _imbalance(rank_loads),
            "server_imbalance": _imbalance(server_loads),
            "target_expert_weights": normalized_target,
            "target_realized_total_variation": distance,
        },
    )


@dataclass(frozen=True)
class BalancedPermutedGateProvider:
    seed: int = 0
    name: str = "balanced_permuted"

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample:
        if topk <= 0 or topk > placement.num_experts:
            raise ValidationError("topk must be in [1, num_experts]")
        expert_order = list(range(placement.num_experts))
        random.Random(self.seed).shuffle(expert_order)
        assignments: list[RoutingAssignment] = []
        global_token_id = 0
        for src_rank, token_count in enumerate(tokens_per_source_rank):
            for token_id in range(token_count):
                first_slot = (global_token_id * topk) % placement.num_experts
                for slot in range(topk):
                    assignments.append(
                        RoutingAssignment(
                            src_rank=src_rank,
                            token_id=token_id,
                            topk_slot=slot,
                            expert_id=expert_order[
                                (first_slot + slot) % placement.num_experts
                            ],
                            route_weight=1.0 / topk,
                        )
                    )
                global_token_id += 1
        return make_gate_sample(
            tuple(assignments),
            placement=placement,
            provider_name=self.name,
            provider_parameters={
                "assignment": "global_cycle_after_expert_permutation",
                "layer_varying": False,
                "microbatch_varying": False,
            },
            seed=self.seed,
            routing_fidelity="synthetic_assignments",
            target_expert_weights=(1.0,) * placement.num_experts,
        )
