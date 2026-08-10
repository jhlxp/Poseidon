#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pysrc"))

from moe_dag import H100CostModel, MoEInvocation, Placement, make_contiguous_expert_placement  # noqa: E402
from moe_dag.algorithms import ProbeEPBuilder, ProbeEPConfig  # noqa: E402
from workload.gate import create_gate_provider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def invocation(ep: int, experts: int, tokens: int, topk: int = 8) -> MoEInvocation:
    gpus_per_server = 8 if ep >= 8 else ep
    placement = Placement(
        ep,
        gpus_per_server,
        make_contiguous_expert_placement(experts, ep),
    )
    token_counts = (tokens,) * ep
    gate = create_gate_provider(
        "ultra_rank_zipf", seed=17, target_rank_imbalance=2.0
    ).sample(
        layer_id=0,
        microbatch_id=0,
        tokens_per_source_rank=token_counts,
        placement=placement,
        topk=topk,
    )
    return MoEInvocation(
        invocation_id=f"ep{ep}_e{experts}_t{tokens}",
        placement=placement,
        tokens_per_source_rank=token_counts,
        hidden=7168,
        ffn_hidden=2048,
        topk=topk,
        dispatch_dtype="fp8",
        combine_dtype="bf16",
        weight_dtype="bf16",
        assignments=gate.assignments,
    )


def measure(ep: int, experts: int, tokens: int, chunk_mib: int, repeats: int) -> dict[str, object]:
    work = invocation(ep, experts, tokens)
    durations: list[float] = []
    peaks: list[int] = []
    last_plan = None
    for _ in range(repeats):
        builder = ProbeEPBuilder(
            H100CostModel(),
            ProbeEPConfig(weight_chunk_bytes=chunk_mib * 1024 * 1024),
        )
        tracemalloc.start()
        start = time.perf_counter_ns()
        last_plan = builder.plan(work)
        durations.append((time.perf_counter_ns() - start) / 1e6)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    assert last_plan is not None
    return {
        "ep": ep,
        "experts": experts,
        "tokens_per_rank": tokens,
        "logical_routes": ep * tokens * work.topk,
        "chunk_mib": chunk_mib,
        "runtime_median_ms": statistics.median(durations),
        "runtime_p95_ms": sorted(durations)[max(0, int(0.95 * len(durations)) - 1)],
        "peak_memory_mib": max(peaks) / 2**20,
        "planned_intents": len(last_plan.planned_intents),
        "admitted_intents": len(last_plan.admitted_intents),
        "remote_replicas": len(last_plan.remote_replicas),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def planner_rows(repeats: int) -> list[dict[str, object]]:
    cases: list[tuple[str, int, int, int, int]] = []
    cases.extend(("ep", ep, 256, 32, 4) for ep in (8, 16, 32, 64))
    cases.extend(("experts", 32, experts, 32, 4) for experts in (64, 128, 256, 512))
    cases.extend(("routes", 32, 256, tokens, 4) for tokens in (8, 16, 32, 64, 128))
    cases.extend(("chunks", 32, 256, 32, chunk) for chunk in (1, 2, 4, 8, 16))
    rows = []
    for sweep, ep, experts, tokens, chunk in cases:
        rows.append({"sweep": sweep, **measure(ep, experts, tokens, chunk, repeats)})
    return rows


def boundary_rows() -> list[dict[str, object]]:
    cost = H100CostModel()
    flops_per_route = 6 * 7168 * 2048
    compute_us = cost.estimate(
        flops_per_route, operation="expert_ffn", token_count=1
    ).duration_us
    base_weight_bytes = 3 * 7168 * 2048 * 2
    rows: list[dict[str, object]] = []
    for weight_scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        for moved_routes in (128, 512, 2048, 8192, 32768, 131072):
            for rate_gbps in (100, 200, 400, 800):
                weight_bytes = base_weight_bytes * weight_scale
                transfer_us = weight_bytes * 8 / (rate_gbps * 1000.0)
                compute_saving_us = moved_routes * compute_us
                exposed_migration_us = transfer_us * 0.10
                rows.append(
                    {
                        "weight_scale": weight_scale,
                        "moved_routes": moved_routes,
                        "effective_rate_gbps": rate_gbps,
                        "target_overlap_ratio": 0.9,
                        "compute_saving_us": compute_saving_us,
                        "transfer_us": transfer_us,
                        "exposed_migration_us": exposed_migration_us,
                        "net_saving_us": compute_saving_us - exposed_migration_us,
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    repeats = 2 if args.mode == "quick" else 7
    write_csv(args.data_dir / "planner_scaling.csv", planner_rows(repeats))
    write_csv(args.data_dir / "break_even.csv", boundary_rows())
    metadata = {
        "schema": "probeep_overhead_boundary_v1",
        "mode": args.mode,
        "paper_eligible": args.mode == "full",
        "planner_evidence": "python_reference_planner_with_gate_generation_excluded_from_timing",
        "planner_claim_boundary": "trend_only_not_online_prototype_latency",
        "analytical_model": {
            "expert_weight_bytes": "3 * hidden * ffn_hidden * bf16_bytes",
            "compute_flops_per_route": "6 * hidden * ffn_hidden",
            "exposed_fraction": 0.10,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
