#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from moe_dag import (
    JsonComputeCostModel,
    ModelSpec,
    Placement,
    ValidationError,
    emit_workload,
    make_contiguous_expert_placement,
)
from moe_dag.models import TransformerWorkloadConfig, build_transformer_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTSim DAG for a Transformer MoE workload."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=("nccl", "deepep", "eplb", "moonep"),
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
                eplb_num_physical_experts=args.eplb_num_physical_experts,
                eplb_num_groups=args.eplb_num_groups,
                eplb_estimated_loads=eplb_loads,
                eplb_load_source=(
                    "cli_explicit_snapshot"
                    if eplb_loads is not None
                    else "current_invocation_proxy"
                ),
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
