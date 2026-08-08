#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_dag import (
    JsonComputeCostModel,
    ModelSpec,
    Placement,
    ValidationError,
    emit_workload,
    make_contiguous_expert_placement,
)
from moe_dag.models import TransformerWorkloadConfig, build_transformer_workload
from workload.gate import GATE_PROVIDER_NAMES, create_gate_provider


DEFAULT_RAW_PLACEMENT = (
    ROOT
    / "workload"
    / "raw_data"
    / "ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTSim DAG for a Transformer MoE workload."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=("nccl", "deepep", "eplb", "moonep", "probeep"),
        default="deepep",
    )
    parser.add_argument("--name", default="moe_block")
    parser.add_argument("--num-ranks", type=int, default=16)
    parser.add_argument("--gpus-per-server", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--tokens-per-rank", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--ffn-hidden", type=int, default=18432)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--replicas-per-rank", type=int, default=2)
    parser.add_argument("--token-padding", type=int, default=128)
    parser.add_argument(
        "--probeep-route-chunk-tokens",
        type=int,
        default=0,
        help="Routes moved per probe step; 0 reuses --chunk-tokens.",
    )
    parser.add_argument(
        "--probeep-weight-chunk-bytes",
        type=int,
        default=4 * 1024 * 1024,
    )
    parser.add_argument(
        "--probeep-max-remote-replicas",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--probeep-expert-slots-per-rank",
        type=int,
        default=40,
        help="Total home plus temporary expert slots available on each rank.",
    )
    parser.add_argument(
        "--probeep-initial-nic-budget-bytes",
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument(
        "--probeep-min-nic-budget-bytes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--probeep-max-nic-budget-bytes",
        type=int,
        default=128 * 1024 * 1024,
    )
    parser.add_argument(
        "--probeep-multiplicative-decrease",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--probeep-additive-increase-bytes",
        type=int,
        default=1024 * 1024,
    )
    parser.add_argument(
        "--probeep-deadband-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--eplb-num-physical-experts",
        type=int,
        default=0,
        help="Total physical expert slots; 0 adds one redundant slot per rank.",
    )
    parser.add_argument(
        "--eplb-num-groups",
        type=int,
        default=0,
        help="Logical expert groups; 0 uses the number of servers.",
    )
    parser.add_argument(
        "--eplb-loads",
        help="Comma-separated estimated load for each logical expert.",
    )
    parser.add_argument("--dispatch-dtype", default="fp8")
    parser.add_argument("--combine-dtype", default="bf16")
    parser.add_argument("--weight-dtype", default="bf16")
    parser.add_argument(
        "--compute-config",
        type=Path,
        help="JSON file containing fixed per-module theoretical/profiled times.",
    )
    parser.add_argument(
        "--compute-time-source",
        choices=("theoretical", "profiled"),
        help="Override selected_source from --compute-config.",
    )
    parser.add_argument(
        "--gate-provider",
        choices=GATE_PROVIDER_NAMES,
        default="balanced_permuted",
    )
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument("--gate-rank-alpha", type=float)
    parser.add_argument("--gate-local-alpha", type=float, default=4.0)
    parser.add_argument(
        "--gate-target-rank-imbalance", type=float, default=2.0
    )
    parser.add_argument("--gate-fast-skew", type=float, default=0.8)
    parser.add_argument(
        "--gate-raw-placement-json",
        type=Path,
        default=DEFAULT_RAW_PLACEMENT,
    )
    parser.add_argument("--gate-raw-csv-pattern", default="decode_{rank}.csv")
    parser.add_argument(
        "--gate-layer-map",
        help="Comma-separated raw layer IDs, one for each model layer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        eplb_loads = None
        if args.eplb_loads is not None:
            try:
                eplb_loads = tuple(
                    float(value.strip()) for value in args.eplb_loads.split(",")
                )
            except ValueError as exc:
                raise ValidationError(
                    "--eplb-loads must be comma-separated numbers"
                ) from exc
        placement = Placement(
            num_ranks=args.num_ranks,
            gpus_per_server=args.gpus_per_server,
            expert_to_rank=make_contiguous_expert_placement(
                args.num_experts,
                args.num_ranks,
            ),
        )
        model = ModelSpec(
            name=args.name,
            hidden=args.hidden,
            ffn_hidden=args.ffn_hidden,
            num_attention_heads=(
                args.num_attention_heads or max(1, args.hidden // args.head_dim)
            ),
            num_kv_heads=(
                args.num_kv_heads or max(1, args.hidden // args.head_dim)
            ),
            head_dim=args.head_dim,
            num_experts=args.num_experts,
            topk=args.topk,
            sequence_length=args.sequence_length,
            num_layers=args.num_layers,
            micro_batches=args.micro_batches,
        )
        if args.compute_time_source and args.compute_config is None:
            raise ValidationError(
                "--compute-time-source requires --compute-config"
            )
        gate_layer_map = None
        if args.gate_layer_map:
            try:
                gate_layer_map = tuple(
                    int(value.strip())
                    for value in args.gate_layer_map.split(",")
                )
            except ValueError as exc:
                raise ValidationError(
                    "--gate-layer-map must be comma-separated integers"
                ) from exc
            if len(gate_layer_map) != args.num_layers:
                raise ValidationError(
                    "--gate-layer-map needs exactly --num-layers entries"
                )
        gate_provider = create_gate_provider(
            args.gate_provider,
            seed=args.gate_seed,
            rank_alpha=args.gate_rank_alpha,
            local_alpha=args.gate_local_alpha,
            target_rank_imbalance=args.gate_target_rank_imbalance,
            fast_skew=args.gate_fast_skew,
            raw_placement_json=args.gate_raw_placement_json,
            raw_csv_pattern=args.gate_raw_csv_pattern,
            layer_map=gate_layer_map,
        )
        cost_model = (
            JsonComputeCostModel.from_path(
                args.compute_config,
                selected_source=args.compute_time_source,
            )
            if args.compute_config is not None
            else None
        )
        result = build_transformer_workload(
            TransformerWorkloadConfig(
                model=model,
                placement=placement,
                tokens_per_rank=args.tokens_per_rank,
                algorithm=args.algorithm,
                chunk_tokens=args.chunk_tokens,
                replicas_per_rank=args.replicas_per_rank,
                token_padding=args.token_padding,
                probeep_route_chunk_tokens=(
                    args.probeep_route_chunk_tokens
                ),
                probeep_weight_chunk_bytes=args.probeep_weight_chunk_bytes,
                probeep_max_remote_replicas=(
                    args.probeep_max_remote_replicas
                ),
                probeep_expert_slots_per_rank=(
                    args.probeep_expert_slots_per_rank
                ),
                probeep_initial_nic_budget_bytes=(
                    args.probeep_initial_nic_budget_bytes
                ),
                probeep_min_nic_budget_bytes=(
                    args.probeep_min_nic_budget_bytes
                ),
                probeep_max_nic_budget_bytes=(
                    args.probeep_max_nic_budget_bytes
                ),
                probeep_multiplicative_decrease=(
                    args.probeep_multiplicative_decrease
                ),
                probeep_additive_increase_bytes=(
                    args.probeep_additive_increase_bytes
                ),
                probeep_deadband_ratio=args.probeep_deadband_ratio,
                eplb_num_physical_experts=args.eplb_num_physical_experts,
                eplb_num_groups=args.eplb_num_groups,
                eplb_estimated_loads=eplb_loads,
                eplb_load_source=(
                    "cli_explicit_snapshot"
                    if eplb_loads is not None
                    else "current_invocation_proxy"
                ),
                gate_provider=gate_provider,
                dispatch_dtype=args.dispatch_dtype,
                combine_dtype=args.combine_dtype,
                weight_dtype=args.weight_dtype,
            ),
            cost_model=cost_model,
        )
        emitted = emit_workload(result.graph, args.output, metadata=result.metadata)
    except ValidationError as exc:
        raise SystemExit(f"invalid workload: {exc}") from exc

    print(f"generated {len(result.graph.tasks)} tasks")
    print(f"DAG: {emitted.dag_path}")
    print(f"CM: {emitted.matrix_path}")
    print(f"manifest: {emitted.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
