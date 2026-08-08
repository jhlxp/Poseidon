from __future__ import annotations

from pathlib import Path

from moe_dag.gate import BalancedPermutedGateProvider, GateProvider, GateSample
from moe_dag.schema import ValidationError

from .raw_receive import RawReceiveCDFGateProvider, RawReceiveDataset
from .synthetic import (
    FastMatrixZipfGateProvider,
    UltraRankZipfGateProvider,
    UniformRandomGateProvider,
)


GATE_PROVIDER_NAMES = (
    "balanced_permuted",
    "uniform_random",
    "ultra_rank_zipf",
    "fast_matrix_zipf",
    "raw_receive_cdf",
)


def create_gate_provider(
    name: str,
    *,
    seed: int = 0,
    rank_alpha: float | None = None,
    local_alpha: float = 4.0,
    target_rank_imbalance: float | None = 2.0,
    fast_skew: float = 0.8,
    raw_placement_json: Path | str | None = None,
    raw_csv_pattern: str = "decode_{rank}.csv",
    layer_map: tuple[int, ...] | None = None,
) -> GateProvider:
    if name == "balanced_permuted":
        return BalancedPermutedGateProvider(seed=seed)
    if name == "uniform_random":
        return UniformRandomGateProvider(seed=seed)
    if name == "ultra_rank_zipf":
        effective_target = target_rank_imbalance if rank_alpha is None else None
        return UltraRankZipfGateProvider(
            seed=seed,
            rank_alpha=rank_alpha,
            local_alpha=local_alpha,
            target_rank_imbalance=effective_target,
        )
    if name == "fast_matrix_zipf":
        return FastMatrixZipfGateProvider(seed=seed, skew=fast_skew)
    if name == "raw_receive_cdf":
        if raw_placement_json is None:
            raise ValidationError("raw_receive_cdf requires a placement JSON")
        return RawReceiveCDFGateProvider(
            dataset=RawReceiveDataset.load(
                raw_placement_json, csv_pattern=raw_csv_pattern
            ),
            seed=seed,
            layer_map=layer_map,
        )
    raise ValidationError(f"unsupported gate provider: {name}")


__all__ = [
    "BalancedPermutedGateProvider",
    "FastMatrixZipfGateProvider",
    "GATE_PROVIDER_NAMES",
    "GateProvider",
    "GateSample",
    "RawReceiveCDFGateProvider",
    "RawReceiveDataset",
    "UltraRankZipfGateProvider",
    "UniformRandomGateProvider",
    "create_gate_provider",
]
