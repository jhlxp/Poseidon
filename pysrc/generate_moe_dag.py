#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from moe_dag import ModelSpec, Placement, ValidationError, emit_workload
from moe_dag.models import TransformerWorkloadConfig, build_transformer_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTSim DAG for one Transformer MoE block."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=("nccl", "deepep", "moonep"),
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
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--replicas-per-rank", type=int, default=2)
    parser.add_argument("--token-padding", type=int, default=128)
    parser.add_argument("--dispatch-dtype", default="fp8")
    parser.add_argument("--combine-dtype", default="bf16")
    parser.add_argument("--weight-dtype", default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        placement = Placement(
            num_ranks=args.num_ranks,
            gpus_per_server=args.gpus_per_server,
            expert_to_rank=tuple(
                expert % args.num_ranks for expert in range(args.num_experts)
            ),
        )
        model = ModelSpec(
            name=args.name,
            hidden=args.hidden,
            ffn_hidden=args.ffn_hidden,
            num_attention_heads=max(1, args.hidden // 128),
            num_kv_heads=max(1, args.hidden // 128),
            head_dim=128,
            num_experts=args.num_experts,
            topk=args.topk,
            sequence_length=args.sequence_length,
            micro_batches=args.micro_batches,
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
                dispatch_dtype=args.dispatch_dtype,
                combine_dtype=args.combine_dtype,
                weight_dtype=args.weight_dtype,
            )
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
