from __future__ import annotations

import hashlib
import heapq
from typing import Sequence

import numpy as np

from moe_dag.gate import GateProvider, GateSample, make_gate_sample
from moe_dag.schema import Placement, RoutingAssignment, ValidationError


def stable_seed(base_seed: int, *coordinates: int) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(base_seed).encode("ascii"))
    for coordinate in coordinates:
        digest.update(b":")
        digest.update(str(coordinate).encode("ascii"))
    return int.from_bytes(digest.digest(), "little")


def normalize_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValidationError("gate weights must be a non-empty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValidationError("gate weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValidationError("gate weights must contain positive mass")
    return weights / total


def _weighted_topk(
    weights: np.ndarray,
    token_count: int,
    topk: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if token_count == 0:
        return np.empty((0, topk), dtype=np.int32)
    if int(np.count_nonzero(weights)) < topk:
        raise ValidationError("gate distribution has fewer than topk positive experts")

    # Exponential races provide weighted sampling without replacement per row.
    uniform = np.maximum(
        rng.random((token_count, weights.size), dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    keys = np.full(uniform.shape, np.inf, dtype=np.float64)
    positive = weights > 0
    keys[:, positive] = -np.log(uniform[:, positive]) / weights[positive]
    selected = np.argpartition(keys, topk - 1, axis=1)[:, :topk]
    selected_keys = np.take_along_axis(keys, selected, axis=1)
    order = np.argsort(selected_keys, axis=1, kind="stable")
    return np.take_along_axis(selected, order, axis=1).astype(np.int32, copy=False)


def sample_weight_rows(
    *,
    weight_rows: np.ndarray,
    tokens_per_source_rank: tuple[int, ...],
    placement: Placement,
    topk: int,
    base_seed: int,
    layer_id: int,
    microbatch_id: int,
    provider_name: str,
    provider_parameters: dict[str, object],
    routing_fidelity: str,
) -> GateSample:
    rows = np.asarray(weight_rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = np.repeat(rows[np.newaxis, :], placement.num_ranks, axis=0)
    if rows.shape != (placement.num_ranks, placement.num_experts):
        raise ValidationError(
            "gate weight rows must have shape [num_ranks, num_experts]"
        )
    normalized = np.vstack([normalize_weights(row) for row in rows])

    assignments: list[RoutingAssignment] = []
    for src_rank, token_count in enumerate(tokens_per_source_rank):
        rng = np.random.default_rng(
            stable_seed(base_seed, layer_id, microbatch_id, src_rank)
        )
        selected = _weighted_topk(normalized[src_rank], token_count, topk, rng)
        for token_id, experts in enumerate(selected):
            for topk_slot, expert_id in enumerate(experts.tolist()):
                assignments.append(
                    RoutingAssignment(
                        src_rank=src_rank,
                        token_id=token_id,
                        topk_slot=topk_slot,
                        expert_id=int(expert_id),
                        route_weight=1.0 / topk,
                    )
                )

    target = tuple(float(value) for value in normalized.mean(axis=0))
    return make_gate_sample(
        tuple(assignments),
        placement=placement,
        provider_name=provider_name,
        provider_parameters=provider_parameters,
        seed=base_seed,
        routing_fidelity=routing_fidelity,
        target_expert_weights=target,
    )


def sample_exact_global_quotas(
    *,
    expert_weights: Sequence[float],
    tokens_per_source_rank: tuple[int, ...],
    placement: Placement,
    topk: int,
    base_seed: int,
    layer_id: int,
    microbatch_id: int,
    provider_name: str,
    provider_parameters: dict[str, object],
    routing_fidelity: str,
) -> GateSample:
    weights = normalize_weights(expert_weights)
    if weights.size != placement.num_experts:
        raise ValidationError(
            "global Gate weights must match placement logical experts"
        )
    if topk <= 0 or topk > placement.num_experts:
        raise ValidationError("topk must be in [1, num_experts]")
    if len(tokens_per_source_rank) != placement.num_ranks:
        raise ValidationError("Gate token counts must match placement ranks")
    if any(count < 0 for count in tokens_per_source_rank):
        raise ValidationError("Gate token counts must be non-negative")

    total_tokens = sum(tokens_per_source_rank)
    if total_tokens <= 0:
        raise ValidationError("exact Gate quota assignment requires tokens")
    total_routes = total_tokens * topk
    expected = weights * total_routes
    quotas = np.floor(expected).astype(np.int64)
    remainder = total_routes - int(quotas.sum())
    if remainder:
        fractions = expected - quotas
        order = np.argsort(-fractions, kind="stable")
        quotas[order[:remainder]] += 1

    if int(quotas.max(initial=0)) > total_tokens:
        raise ValidationError(
            "target expert quota exceeds one selection per token"
        )
    if int(np.count_nonzero(quotas)) < topk:
        raise ValidationError("target expert quotas cover fewer than topk experts")

    rng = np.random.default_rng(
        stable_seed(base_seed, layer_id, microbatch_id, 0x51554F54)
    )
    heap: list[tuple[int, float, int]] = [
        (-int(quota), float(rng.random()), expert)
        for expert, quota in enumerate(quotas.tolist())
        if quota
    ]
    heapq.heapify(heap)
    token_order = rng.permutation(total_tokens)
    rank_boundaries = np.cumsum(tokens_per_source_rank, dtype=np.int64)
    assignments: list[RoutingAssignment] = []

    for row_index in range(total_tokens):
        selected: list[tuple[int, int]] = []
        for _ in range(topk):
            if not heap:
                raise ValidationError("exact Gate quota scheduler exhausted experts")
            negative_remaining, _tie, expert = heapq.heappop(heap)
            selected.append((-negative_remaining, expert))
        rng.shuffle(selected)

        global_token = int(token_order[row_index])
        src_rank = int(
            np.searchsorted(rank_boundaries, global_token, side="right")
        )
        rank_begin = 0 if src_rank == 0 else int(rank_boundaries[src_rank - 1])
        token_id = global_token - rank_begin
        for topk_slot, (remaining, expert) in enumerate(selected):
            assignments.append(
                RoutingAssignment(
                    src_rank=src_rank,
                    token_id=token_id,
                    topk_slot=topk_slot,
                    expert_id=expert,
                    route_weight=1.0 / topk,
                )
            )
            remaining -= 1
            if remaining:
                heapq.heappush(
                    heap,
                    (-remaining, float(rng.random()), expert),
                )

    if heap:
        raise ValidationError("exact Gate quota scheduler left unassigned routes")
    parameters = {
        **provider_parameters,
        "quota_rounding": "largest_remainder",
        "quota_assignment": "seeded_bipartite_greedy",
        "quota_max_absolute_error_routes": 1,
    }
    return make_gate_sample(
        tuple(assignments),
        placement=placement,
        provider_name=provider_name,
        provider_parameters=parameters,
        seed=base_seed,
        routing_fidelity=routing_fidelity,
        target_expert_weights=tuple(float(value) for value in weights),
    )


__all__ = [
    "GateProvider",
    "GateSample",
    "normalize_weights",
    "sample_exact_global_quotas",
    "sample_weight_rows",
    "stable_seed",
]
