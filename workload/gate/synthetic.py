from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from moe_dag.gate import GateSample
from moe_dag.schema import Placement, ValidationError

from .base import normalize_weights, sample_weight_rows, stable_seed


def _topk_inclusion_probabilities(weights: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0 or topk > weights.size:
        raise ValidationError("topk must be in [1, num_experts]")
    if topk == weights.size:
        return np.ones_like(weights)
    lo, hi = 0.0, 1.0
    while float(np.sum(1.0 - np.exp(-hi * weights))) < topk:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if float(np.sum(1.0 - np.exp(-mid * weights))) < topk:
            lo = mid
        else:
            hi = mid
    return 1.0 - np.exp(-hi * weights)


def _ultra_weights(
    placement: Placement,
    rank_alpha: float,
    local_alpha: float,
    layer_seed: int,
) -> np.ndarray:
    if rank_alpha < 0 or local_alpha < 0:
        raise ValidationError("Ultra-style Zipf alphas must be non-negative")
    experts_by_rank: list[list[int]] = [
        [] for _ in range(placement.num_ranks)
    ]
    for expert, rank in enumerate(placement.expert_to_rank):
        experts_by_rank[rank].append(expert)

    rng = np.random.default_rng(layer_seed)
    rank_order = rng.permutation(placement.num_ranks)
    weights = np.zeros(placement.num_experts, dtype=np.float64)
    for heat_index, rank_value in enumerate(rank_order.tolist(), start=1):
        experts = experts_by_rank[rank_value]
        if not experts:
            continue
        local_order = rng.permutation(len(experts))
        for local_heat_index, expert_offset in enumerate(local_order.tolist(), start=1):
            expert = experts[expert_offset]
            weights[expert] = (
                1.0 / math.pow(heat_index, rank_alpha)
                * 1.0 / math.pow(local_heat_index, local_alpha)
            )
    return normalize_weights(weights)


def _rank_imbalance(
    placement: Placement,
    weights: np.ndarray,
    topk: int,
) -> float:
    inclusion = _topk_inclusion_probabilities(weights, topk)
    rank_loads = np.zeros(placement.num_ranks, dtype=np.float64)
    for expert, value in enumerate(inclusion):
        rank_loads[placement.expert_rank(expert)] += value
    mean = float(rank_loads.mean())
    return float(rank_loads.max()) / mean if mean else 0.0


@dataclass(frozen=True)
class UniformRandomGateProvider:
    seed: int = 0
    name: str = "uniform_random"

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample:
        return sample_weight_rows(
            weight_rows=np.ones(placement.num_experts, dtype=np.float64),
            tokens_per_source_rank=tokens_per_source_rank,
            placement=placement,
            topk=topk,
            base_seed=self.seed,
            layer_id=layer_id,
            microbatch_id=microbatch_id,
            provider_name=self.name,
            provider_parameters={"sampling": "weighted_without_replacement"},
            routing_fidelity="synthetic_assignments",
        )


@dataclass(frozen=True)
class UltraRankZipfGateProvider:
    seed: int = 0
    rank_alpha: float | None = None
    local_alpha: float = 4.0
    target_rank_imbalance: float | None = 2.0
    name: str = "ultra_rank_zipf"

    def __post_init__(self) -> None:
        if self.rank_alpha is not None and self.target_rank_imbalance is not None:
            raise ValidationError(
                "set either rank_alpha or target_rank_imbalance, not both"
            )
        if self.rank_alpha is None and self.target_rank_imbalance is None:
            raise ValidationError("Ultra-style Zipf requires an alpha or target imbalance")
        if self.rank_alpha is not None and self.rank_alpha < 0:
            raise ValidationError("rank_alpha must be non-negative")
        if self.local_alpha < 0:
            raise ValidationError("local_alpha must be non-negative")
        if self.target_rank_imbalance is not None and self.target_rank_imbalance < 1:
            raise ValidationError("target_rank_imbalance must be at least 1")

    def _weights_and_alpha(
        self, placement: Placement, topk: int, layer_id: int
    ) -> tuple[np.ndarray, float]:
        layer_seed = stable_seed(self.seed, layer_id, 0x554C5452)
        if self.rank_alpha is not None:
            return (
                _ultra_weights(
                    placement,
                    self.rank_alpha,
                    self.local_alpha,
                    layer_seed,
                ),
                self.rank_alpha,
            )
        assert self.target_rank_imbalance is not None
        lo, hi = 0.0, 8.0
        for _ in range(40):
            mid = (lo + hi) * 0.5
            weights = _ultra_weights(
                placement, mid, self.local_alpha, layer_seed
            )
            if _rank_imbalance(placement, weights, topk) < self.target_rank_imbalance:
                lo = mid
            else:
                hi = mid
        alpha = hi
        return (
            _ultra_weights(placement, alpha, self.local_alpha, layer_seed),
            alpha,
        )

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample:
        weights, effective_alpha = self._weights_and_alpha(
            placement, topk, layer_id
        )
        expected_imbalance = _rank_imbalance(placement, weights, topk)
        return sample_weight_rows(
            weight_rows=weights,
            tokens_per_source_rank=tokens_per_source_rank,
            placement=placement,
            topk=topk,
            base_seed=self.seed,
            layer_id=layer_id,
            microbatch_id=microbatch_id,
            provider_name=self.name,
            provider_parameters={
                "rank_alpha": effective_alpha,
                "local_alpha": self.local_alpha,
                "target_rank_imbalance": self.target_rank_imbalance,
                "expected_rank_imbalance": expected_imbalance,
                "layer_rank_permutation": True,
                "sampling": "weighted_without_replacement",
            },
            routing_fidelity="synthetic_assignments",
        )


@dataclass(frozen=True)
class FastMatrixZipfGateProvider:
    seed: int = 0
    skew: float = 0.8
    support: int = 10_000
    name: str = "fast_matrix_zipf"

    def __post_init__(self) -> None:
        if not 0 < self.skew < 1:
            raise ValidationError("FAST-style skew must be in (0, 1)")
        if self.support <= 1:
            raise ValidationError("FAST-style support must exceed 1")

    def _weight_rows(self, placement: Placement, layer_id: int) -> np.ndarray:
        values = np.arange(1, self.support + 1, dtype=np.float64)
        cdf = np.cumsum(np.power(values, -self.skew))
        cdf /= cdf[-1]
        rng = np.random.default_rng(
            stable_seed(self.seed, layer_id, 0x46415354)
        )
        uniforms = rng.random((placement.num_ranks, placement.num_experts))
        return np.searchsorted(cdf, uniforms, side="left").astype(np.float64) + 1.0

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample:
        return sample_weight_rows(
            weight_rows=self._weight_rows(placement, layer_id),
            tokens_per_source_rank=tokens_per_source_rank,
            placement=placement,
            topk=topk,
            base_seed=self.seed,
            layer_id=layer_id,
            microbatch_id=microbatch_id,
            provider_name=self.name,
            provider_parameters={
                "skew": self.skew,
                "support": self.support,
                "matrix_shape": [placement.num_ranks, placement.num_experts],
                "row_budget": "fixed_by_tokens_times_topk",
                "sampling": "finite_inverse_zipf_cdf",
            },
            routing_fidelity="sampled_from_source_expert_matrix",
        )
