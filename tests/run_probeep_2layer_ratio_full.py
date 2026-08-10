#!/usr/bin/env python3
"""Run ProbeEP feedback with dynamic DAG append in one HTSim process."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYSRC = ROOT / "pysrc"
sys.path.insert(0, str(PYSRC))
sys.path.insert(0, str(ROOT))

from moe_dag import (  # noqa: E402
    DynamicDagEmitter,
    H100CostModel,
    JsonComputeCostModel,
    ModelSpec,
    Placement,
    make_contiguous_expert_placement,
    probeep_weight_dispatch_observations,
)
from moe_dag.algorithms import (  # noqa: E402
    ProbeDispatchFeedback,
    ProbeNICController,
    ProbeNICControllerConfig,
)
from moe_dag.models import (  # noqa: E402
    IncrementalLayerResult,
    IncrementalTransformerWorkloadBuilder,
    TransformerWorkloadConfig,
)
from tests.run_dsv3_2layer_algorithms import (  # noqa: E402
    build_simulator,
    mode_config,
    run_visualizations,
)
from workload.gate import create_gate_provider  # noqa: E402


BINARY = ROOT / "htsim" / "sim" / "build-mprail" / "datacenter" / "htsim_uec"
DEFAULT_COMPUTE_CONFIG = (
    PYSRC / "compute_profiles" / "H20_DSV3_EP32_compute_4096tpr.json"
)
RAW_PLACEMENT = (
    ROOT
    / "workload"
    / "raw_data"
    / "ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json"
)
NUM_RANKS = 32
NUM_MICROBATCHES = 2
CONTROLLER_CONFIG = ProbeNICControllerConfig(
    initial_budget_bytes=16 * 1024 * 1024,
    nic_line_rate_gbps=400.0,
    target_overlap_ratio=0.90,
)
TASK_START_RE = re.compile(r"DAG_TASK_START task=(\d+).* time_us=([0-9.eE+-]+)")
TASK_DONE_RE = re.compile(r"DAG_TASK_DONE task=(\d+).* time_us=([0-9.eE+-]+)")
OBSERVATION_RE = re.compile(
    r"DAG_OBSERVATION_READY observation=(\d+) time_us=([0-9.eE+-]+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use 4096 tokens/rank. Default is the 2-token functional test.",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--compute-config", type=Path, default=DEFAULT_COMPUTE_CONFIG
    )
    parser.add_argument(
        "--gate-layer-map",
        help="Comma-separated raw receive layers; defaults to 0..N-1.",
    )
    parser.add_argument("--gate-seed", type=int, default=17)
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip ZIP generation; intended only for protocol debugging.",
    )
    return parser.parse_args()


def _rank_values(value: object) -> tuple[float, ...]:
    if isinstance(value, dict):
        return tuple(
            float(value.get(str(rank), value.get(rank, 0.0)))
            for rank in range(NUM_RANKS)
        )
    if isinstance(value, list):
        require(len(value) == NUM_RANKS, "rank metric length mismatch")
        return tuple(float(item) for item in value)
    raise TypeError("rank metric must be a list or mapping")


def _invocation_metadata(
    builder: IncrementalTransformerWorkloadBuilder,
    layer: int,
    micro_batch: int,
) -> dict[str, object]:
    result = builder.layer_results[layer]
    for item in result.algorithm_metadata:
        if int(item["micro_batch"]) == micro_batch:
            return item
    raise AssertionError(f"missing invocation metadata L{layer}/MB{micro_batch}")


def _compute_by_rank(
    builder: IncrementalTransformerWorkloadBuilder,
    layer: int,
    micro_batch: int,
    operation: str,
) -> dict[int, object]:
    result = {
        int(task.rank): task
        for task in builder.graph.tasks
        if task.kind == "compute"
        and task.metadata.get("layer") == layer
        and task.metadata.get("micro_batch") == micro_batch
        and task.metadata.get("operation") == operation
        and task.rank is not None
    }
    require(len(result) == NUM_RANKS, f"missing {operation} tasks by rank")
    return result


def make_feedback(
    *,
    observation_id: int,
    observation_time_us: float,
    builder: IncrementalTransformerWorkloadBuilder,
    emitter: DynamicDagEmitter,
    starts: dict[int, float],
    dones: dict[int, float],
) -> tuple[ProbeDispatchFeedback, dict[str, object]]:
    layer, micro_batch = divmod(observation_id, 2)
    kind = "attention" if micro_batch == 0 else "moe"
    invocation = _invocation_metadata(builder, layer, micro_batch)
    stage_keys = probeep_weight_dispatch_observations(
        builder.graph, layer
    )[observation_id]

    if kind == "attention":
        compute_tasks = _compute_by_rank(builder, layer, 1, "attention")
    else:
        compute_tasks = _compute_by_rank(builder, layer, 0, "expert_ffn")
    release = [0.0] * NUM_RANKS
    selected_compute_us = [0.0] * NUM_RANKS
    for rank, task in compute_tasks.items():
        task_id = emitter.task_ids[task.key]
        require(task_id in starts, f"paired compute rank {rank} has not started")
        release[rank] = starts[task_id]
        selected_compute_us[rank] = float(task.duration_us)

    stage_done = list(release)
    migration_tx = [0] * NUM_RANKS
    migration_rx = [0] * NUM_RANKS
    for key in stage_keys:
        task = builder.graph.task(key)
        task_id = emitter.task_ids[key]
        require(task_id in starts and task_id in dones,
                f"observation released before task {task_id} completed")
        if task.src_rank is None or task.dst_rank is None:
            continue
        for rank in {task.src_rank, task.dst_rank}:
            stage_done[rank] = max(stage_done[rank], dones[task_id])
        if task.payload_kind == "expert_weight_rdma":
            migration_tx[task.src_rank] += task.transfer_bytes
            migration_rx[task.dst_rank] += task.transfer_bytes
    stage_us = tuple(
        max(0.0, stage_done[rank] - release[rank])
        for rank in range(NUM_RANKS)
    )
    migration = tuple(
        max(migration_tx[rank], migration_rx[rank])
        for rank in range(NUM_RANKS)
    )
    dispatch_baseline = tuple(
        max(int(tx), int(rx))
        for tx, rx in zip(
            invocation["predicted_dispatch_tx_bytes"],
            invocation["predicted_dispatch_rx_bytes"],
        )
    )
    dispatch_tx = tuple(
        int(value) for value in invocation["predicted_dispatch_tx_bytes"]
    )
    dispatch_rx = tuple(
        int(value) for value in invocation["predicted_dispatch_rx_bytes"]
    )
    attention_us = _rank_values(invocation["attention_compute_us_by_rank"])
    moe_us = _rank_values(invocation["moe_compute_us_by_rank"])
    if kind == "attention":
        attention_us = tuple(selected_compute_us)
    else:
        moe_us = tuple(selected_compute_us)
    feedback = ProbeDispatchFeedback(
        observation_id=observation_id,
        attention_compute_us_by_rank=attention_us,
        moe_compute_us_by_rank=moe_us,
        dispatch_overlap_compute_kind=kind,
        weight_dispatch_us_by_rank=stage_us,
        dispatch_baseline_bytes_by_rank=dispatch_baseline,
        migration_bytes_by_rank=migration,
        pending_migration_exists=bool(
            invocation["deferred_migration_intents"]
        ),
        source="in_process_dynamic_htsim_observation",
        migration_tx_bytes_by_rank=tuple(migration_tx),
        migration_rx_bytes_by_rank=tuple(migration_rx),
        dispatch_tx_bytes_by_rank=dispatch_tx,
        dispatch_rx_bytes_by_rank=dispatch_rx,
    )
    transfer_bytes = invocation["transfer_bytes_by_payload"]
    remote_weight_rdma_bytes = int(
        transfer_bytes.get("expert_weight_rdma", 0)
    )
    local_weight_bytes = sum(
        int(transfer_bytes.get(payload, 0))
        for payload in (
            "expert_weight_scatter",
            "expert_weight_gather",
            "expert_weight_prefetch",
        )
    )
    measurement = {
        "observation_id": observation_id,
        "layer": layer,
        "micro_batch": micro_batch,
        "simulation_time_us": observation_time_us,
        "compute_kind": kind,
        "compute_ref_us": max(selected_compute_us),
        "weight_dispatch_max_us": max(stage_us),
        "network_over_compute": (
            max(stage_us) / max(selected_compute_us)
            if max(selected_compute_us) > 0 else 0.0
        ),
        "weight_dispatch_us_by_rank": list(stage_us),
        "selected_compute_us_by_rank": selected_compute_us,
        "dispatch_baseline_bytes_by_rank": list(dispatch_baseline),
        "dispatch_tx_bytes_by_rank": list(dispatch_tx),
        "dispatch_rx_bytes_by_rank": list(dispatch_rx),
        "migration_bytes_by_rank": list(migration),
        "migration_bytes_semantics": "max_tx_rx_per_rank_full_duplex_endpoint",
        "migration_endpoint_bytes_by_rank": list(migration),
        "migration_tx_bytes_by_rank": migration_tx,
        "migration_rx_bytes_by_rank": migration_rx,
        "migration_tx_total_bytes": sum(migration_tx),
        "migration_rx_total_bytes": sum(migration_rx),
        "remote_weight_rdma_bytes": remote_weight_rdma_bytes,
        "local_weight_bytes": local_weight_bytes,
        "planned_migration_intent_count": len(
            invocation["planned_migration_intents"]
        ),
        "admitted_migration_intent_count": len(
            invocation["admitted_migration_intents"]
        ),
        "deferred_migration_intent_count": len(
            invocation["deferred_migration_intents"]
        ),
        "remote_replica_count": len(invocation["remote_replicas"]),
        "moved_route_count": sum(
            int(item["moved_route_count"])
            for item in invocation["admitted_migration_intents"]
        ),
        "budget_used_by_rank": list(invocation["nic_budget_before"]),
    }
    return feedback, measurement


def _write(process: subprocess.Popen[str], payload: str) -> None:
    assert process.stdin is not None
    process.stdin.write(payload)
    if not payload.endswith("\n"):
        process.stdin.write("\n")
    process.stdin.flush()


def _command(
    workload_dir: Path,
    simulation_dir: Path,
    end_us: int,
) -> list[str]:
    return [
        str(BINARY),
        "-topology", "mprail",
        "-mprail_planes", "1",
        "-mprail_gpus_per_server", "8",
        "-mprail_l1_eps_per_plane", "4",
        "-mprail_l0_l1_links_per_spine", "1",
        "-linkspeed", "400000",
        "-local_linkspeed", "7200000",
        "-local_latency_ns", "50",
        "-hop_latency", "0.1",
        "-switch_latency", "0.02",
        "-mtu", "4150",
        "-q", "128",
        "-end", str(end_us),
        "-strat", "ecmp_host",
        "-sender_cc_only",
        "-sender_cc_algo", "nscc",
        "-tm", str(workload_dir / "nodes.cm"),
        "-dag_control",
        "-o", str(simulation_dir / "htsim.dat"),
    ]


def _write_observations(
    run_dir: Path, records: list[dict[str, object]]
) -> None:
    (run_dir / "probeep_dispatch_observations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "observation_id", "layer", "micro_batch", "simulation_time_us",
        "compute_kind", "compute_ref_us", "weight_dispatch_max_us",
        "network_over_compute", "controller_action", "budget_before_mean_mib",
        "budget_after_mean_mib", "planned_migration_intent_count",
        "admitted_migration_intent_count", "deferred_migration_intent_count",
        "remote_replica_count", "moved_route_count", "remote_weight_rdma_bytes",
        "local_weight_bytes", "migration_tx_total_bytes",
        "migration_rx_total_bytes", "global_network_to_compute_ratio",
        "global_adjustment_factor", "network_bottleneck_ranks",
    ]
    with (run_dir / "probeep_dispatch_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _combine_telemetry(
    *,
    builder: IncrementalTransformerWorkloadBuilder,
    emitter: DynamicDagEmitter,
    starts: dict[int, float],
    dones: dict[int, float],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in range(builder.config.model.num_layers):
        for micro_batch in range(NUM_MICROBATCHES):
            intervals: list[list[tuple[float, float]]] = [
                [] for _ in range(NUM_RANKS)
            ]
            for task in builder.graph.tasks:
                if (
                    task.kind != "transfer"
                    or task.metadata.get("layer") != layer
                    or task.metadata.get("micro_batch") != micro_batch
                    or task.metadata.get("stream_phase") != "combine"
                    or task.src_rank is None
                    or task.dst_rank is None
                ):
                    continue
                task_id = emitter.task_ids[task.key]
                interval = (starts[task_id], dones[task_id])
                intervals[task.src_rank].append(interval)
                intervals[task.dst_rank].append(interval)

            if micro_batch == 0:
                partner_layer = layer
                partner_micro_batch = 1
                partner_operation = "expert_ffn"
                partner_name: str | None = "same_layer_mb1_expert_ffn"
            elif layer + 1 < builder.config.model.num_layers:
                partner_layer = layer + 1
                partner_micro_batch = 0
                partner_operation = "attention"
                partner_name = "next_layer_mb0_attention"
            else:
                partner_layer = layer
                partner_micro_batch = 0
                partner_operation = ""
                partner_name = None

            active_by_rank = [
                _merged_duration(rank_intervals)
                for rank_intervals in intervals
            ]
            compute_by_rank = [0.0] * NUM_RANKS
            overlap_by_rank = [0.0] * NUM_RANKS
            if partner_name is not None:
                partners = _compute_by_rank(
                    builder,
                    partner_layer,
                    partner_micro_batch,
                    partner_operation,
                )
                for rank, task in partners.items():
                    task_id = emitter.task_ids[task.key]
                    compute_interval = (starts[task_id], dones[task_id])
                    compute_by_rank[rank] = (
                        compute_interval[1] - compute_interval[0]
                    )
                    clipped = [
                        (
                            max(start, compute_interval[0]),
                            min(end, compute_interval[1]),
                        )
                        for start, end in intervals[rank]
                    ]
                    overlap_by_rank[rank] = _merged_duration(clipped)
            result.append({
                "layer": layer,
                "micro_batch": micro_batch,
                "communication_kind": "combine",
                "controller_eligible": False,
                "compute_partner": partner_name,
                "communication_active_max_us": max(active_by_rank),
                "compute_max_us": max(compute_by_rank),
                "overlap_max_us": max(overlap_by_rank),
                "communication_active_us_by_rank": active_by_rank,
                "compute_us_by_rank": compute_by_rank,
                "overlap_us_by_rank": overlap_by_rank,
            })
    return result


def _write_combine_telemetry(
    run_dir: Path, records: list[dict[str, object]]
) -> None:
    (run_dir / "probeep_combine_telemetry.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "layer", "micro_batch", "communication_kind", "controller_eligible",
        "compute_partner", "communication_active_max_us", "compute_max_us",
        "overlap_max_us",
    ]
    with (run_dir / "probeep_combine_telemetry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _validate_runtime(
    *,
    builder: IncrementalTransformerWorkloadBuilder,
    emitter: DynamicDagEmitter,
    starts: dict[int, float],
    dones: dict[int, float],
    dispatch_records: list[dict[str, object]],
    combine_records: list[dict[str, object]],
    require_remote_migration: bool,
) -> dict[str, object]:
    require(len(starts) == len(emitter.task_ids), "not every task started")
    require(len(dones) == len(emitter.task_ids), "not every task completed")

    source_rail_checks = 0
    for layer in range(builder.config.model.num_layers):
        for micro_batch in range(NUM_MICROBATCHES):
            scoped = [
                task
                for task in builder.graph.tasks
                if task.metadata.get("layer") == layer
                and task.metadata.get("micro_batch") == micro_batch
            ]
            weight_by_src: dict[int, list[object]] = {}
            for task in scoped:
                if task.payload_kind == "expert_weight_rdma":
                    assert task.src_rank is not None
                    weight_by_src.setdefault(task.src_rank, []).append(task)
            for task in scoped:
                if task.metadata.get("hierarchical_leg") != "dispatch_fabric":
                    continue
                assert task.src_rank is not None
                weights = weight_by_src.get(task.src_rank, [])
                if not weights:
                    continue
                source_rail_checks += 1
                latest_weight_done = max(
                    dones[emitter.task_ids[weight.key]] for weight in weights
                )
                require(
                    starts[emitter.task_ids[task.key]] + 1e-6
                    >= latest_weight_done,
                    f"Dispatch task {task.key} started before source-rail Weight TX",
                )

    same_rank_compute_overlaps = 0
    for rank in range(NUM_RANKS):
        intervals = sorted(
            (
                starts[emitter.task_ids[task.key]],
                dones[emitter.task_ids[task.key]],
                task.key,
            )
            for task in builder.graph.tasks
            if task.kind == "compute" and task.rank == rank
        )
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1] - 1e-6:
                same_rank_compute_overlaps += 1
    require(same_rank_compute_overlaps == 0, "same-rank compute tasks overlap")

    remote_replicas = sum(
        len(item["remote_replicas"])
        for result in builder.layer_results
        for item in result.algorithm_metadata
    )
    if require_remote_migration:
        require(remote_replicas > 0, "full ProbeEP run admitted no remote replicas")
        require(
            source_rail_checks > 0,
            "full ProbeEP run did not exercise Weight-to-Dispatch ordering",
        )

    cross_layer = [
        record
        for record in combine_records
        if record["compute_partner"] == "next_layer_mb0_attention"
    ]
    require(
        len(cross_layer) == builder.config.model.num_layers - 1,
        "cross-layer Combine telemetry count mismatch",
    )
    require(
        all(float(record["overlap_max_us"]) > 0 for record in cross_layer),
        "dynamic wavefront did not produce cross-layer overlap",
    )

    release_gaps: list[float] = []
    for layer in range(1, builder.config.model.num_layers):
        for micro_batch in range(NUM_MICROBATCHES):
            previous_final = _compute_by_rank(
                builder, layer - 1, micro_batch, "combine_reduce"
            )
            current_attention = _compute_by_rank(
                builder, layer, micro_batch, "attention"
            )
            for rank in range(NUM_RANKS):
                gap = (
                    starts[emitter.task_ids[current_attention[rank].key]]
                    - dones[emitter.task_ids[previous_final[rank].key]]
                )
                require(
                    abs(gap) <= 1e-6,
                    f"Layer {layer}/MB{micro_batch}/rank{rank} dynamic append "
                    f"introduced a {gap} us Attention release gap",
                )
                release_gaps.append(gap)

    records_by_scope = {
        (int(record["layer"]), int(record["micro_batch"])): record
        for record in dispatch_records
    }
    for layer in range(1, builder.config.model.num_layers):
        for micro_batch in range(NUM_MICROBATCHES):
            previous = records_by_scope[(layer - 1, micro_batch)]
            invocation = _invocation_metadata(builder, layer, micro_batch)
            require(
                list(invocation["nic_budget_before"])
                == previous["controller_update"]["budget_after"],
                f"Layer {layer}/MB{micro_batch} did not consume previous feedback",
            )
    weight_byte_conservation_checks = 0
    for record in dispatch_records:
        tx = [int(value) for value in record["migration_tx_bytes_by_rank"]]
        rx = [int(value) for value in record["migration_rx_bytes_by_rank"]]
        endpoint = [
            int(value) for value in record["migration_endpoint_bytes_by_rank"]
        ]
        rdma_bytes = int(record["remote_weight_rdma_bytes"])
        require(sum(tx) == rdma_bytes, "migration TX bytes != RDMA bytes")
        require(sum(rx) == rdma_bytes, "migration RX bytes != RDMA bytes")
        require(
            endpoint == [max(tx_value, rx_value) for tx_value, rx_value in zip(tx, rx)],
            "migration endpoint footprint is not max(TX,RX)",
        )
        require(
            int(record["remote_replica_count"])
            <= int(record["admitted_migration_intent_count"]),
            "remote replicas exceed admitted migration intents",
        )
        compute_ref = float(record["compute_ref_us"])
        network_max = float(record["weight_dispatch_max_us"])
        update = record["controller_update"]
        expected_ratio = network_max / compute_ref if compute_ref > 0 else 0.0
        expected_factor = (
            float(update["target_nic_us"]) / network_max
            if compute_ref > 0 and network_max > 0 else 1.0
        )
        require(
            abs(float(update["global_network_to_compute_ratio"]) - expected_ratio)
            <= 1e-12,
            "controller did not use global Nmax/Cmax",
        )
        require(
            abs(float(update["global_adjustment_factor"]) - expected_factor)
            <= 1e-12,
            "controller did not use one global barrier adjustment factor",
        )
        observed_totals = [
            max(int(dispatch_tx) + int(weight_tx),
                int(dispatch_rx) + int(weight_rx))
            for dispatch_tx, dispatch_rx, weight_tx, weight_rx in zip(
                record["dispatch_tx_bytes_by_rank"],
                record["dispatch_rx_bytes_by_rank"],
                tx,
                rx,
            )
        ]
        require(
            list(update["observed_total_bytes_by_rank"]) == observed_totals,
            "controller observed total bytes != Token baseline + Expert Weight",
        )
        bottleneck_total = max(
            observed_totals[int(rank)]
            for rank in update["bottleneck_ranks"]
        )
        require(
            int(update["bottleneck_observed_total_bytes"])
            == bottleneck_total,
            "controller did not bind sampled bytes to the Nmax rank",
        )
        expected_probed_total = int(bottleneck_total * expected_factor)
        require(
            int(update["probed_total_nic_max_bytes"])
            == expected_probed_total,
            "controller did not scale actual bytes by alpha*Cmax/Nmax",
        )
        require(
            int(update["effective_total_nic_max_bytes"])
            == min(
                expected_probed_total,
                int(update["nic_theoretical_max_bytes"]),
            ),
            "controller total NIC byte max did not obey the 400 Gbps cap",
        )
        weight_byte_conservation_checks += 1
    return {
        "source_rail_weight_before_dispatch_checks": source_rail_checks,
        "remote_replica_count": remote_replicas,
        "same_rank_compute_overlap_count": same_rank_compute_overlaps,
        "cross_layer_overlap_boundaries": len(cross_layer),
        "cross_layer_overlap_max_us": [
            record["overlap_max_us"] for record in cross_layer
        ],
        "cross_layer_attention_release_checks": len(release_gaps),
        "cross_layer_attention_release_gap_max_us": max(
            (abs(value) for value in release_gaps), default=0.0
        ),
        "budget_state_chain_checks": (
            builder.config.model.num_layers - 1
        ) * NUM_MICROBATCHES,
        "weight_byte_conservation_checks": weight_byte_conservation_checks,
    }


def _write_dashboard(
    path: Path,
    records: list[dict[str, object]],
    combine_records: list[dict[str, object]],
    mode: str,
    layers: int,
) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{item['observation_id']}</td><td>{item['layer']}</td>"
        f"<td>{item['micro_batch']}</td><td>{escape(str(item['compute_kind']))}</td>"
        f"<td>{float(item['compute_ref_us']):.2f}</td>"
        f"<td>{float(item['weight_dispatch_max_us']):.2f}</td>"
        f"<td>{float(item['network_over_compute']):.3f}</td>"
        f"<td>{float(item['global_adjustment_factor']):.3f}</td>"
        f"<td>{escape(','.join(str(rank) for rank in item['network_bottleneck_ranks']))}</td>"
        f"<td>{int(item['controller_update']['bottleneck_observed_total_bytes']) / 1048576:.2f}</td>"
        f"<td>{int(item['controller_update']['probed_total_nic_max_bytes']) / 1048576:.2f}</td>"
        f"<td>{int(item['controller_update']['nic_theoretical_max_bytes']) / 1048576:.2f}</td>"
        f"<td>{escape(str(item['controller_action']))}</td>"
        f"<td>{float(item['budget_before_mean_mib']):.2f}</td>"
        f"<td>{float(item['budget_after_mean_mib']):.2f}</td></tr>"
        for item in records
    )
    combine_rows = "".join(
        "<tr>"
        f"<td>{item['layer']}</td><td>{item['micro_batch']}</td>"
        f"<td>{escape(str(item['compute_partner'] or '-'))}</td>"
        f"<td>{float(item['communication_active_max_us']):.2f}</td>"
        f"<td>{float(item['compute_max_us']):.2f}</td>"
        f"<td>{float(item['overlap_max_us']):.2f}</td></tr>"
        for item in combine_records
    )
    migration_rows = "".join(
        "<tr>"
        f"<td>{item['layer']}</td><td>{item['micro_batch']}</td>"
        f"<td>{item['planned_migration_intent_count']}</td>"
        f"<td>{item['admitted_migration_intent_count']}</td>"
        f"<td>{item['deferred_migration_intent_count']}</td>"
        f"<td>{item['remote_replica_count']}</td>"
        f"<td>{item['moved_route_count']}</td>"
        f"<td>{int(item['remote_weight_rdma_bytes']) / 1048576:.2f}</td>"
        f"<td>{int(item['local_weight_bytes']) / 1048576:.2f}</td>"
        f"<td>{int(item['migration_tx_total_bytes']) / 1048576:.2f}</td>"
        f"<td>{int(item['migration_rx_total_bytes']) / 1048576:.2f}</td></tr>"
        for item in records
    )
    path.write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>ProbeEP dynamic DAG</title><style>"
        "body{font:14px system-ui;margin:24px;color:#182230;background:#f5f7fa}"
        "main{max-width:1500px;margin:auto}section{background:#fff;border:1px solid #d8dee8;"
        "padding:14px;margin:12px 0;border-radius:6px;overflow:auto}table{border-collapse:collapse;"
        "width:100%}th,td{padding:7px;border-bottom:1px solid #e5e9ef;text-align:right}"
        "a{color:#165dcc}</style></head><body><main>"
        f"<h1>ProbeEP dynamic DAG / {layers} layers / {escape(mode)}</h1>"
        "<p>One HTSim PID, one EventList, runtime DAG append. All measurements below "
        "come from the same execution log.</p>"
        "<section><h2>Views</h2><p>"
        "<a href='timeline/dag_gpu_timeline.html'>GPU 0-31 timeline</a> · "
        "<a href='gate_load/gate_load_profile.html'>Gate and expert load</a> · "
        "<a href='link_load/mprail_link_load_by_layer.png'>MpRail link load</a>"
        "</p></section>"
        "<section><h2>Dispatch observations</h2><table><thead><tr>"
        "<th>ID</th><th>Layer</th><th>MB</th><th>Compute</th><th>C us</th>"
        "<th>N us</th><th>N/C</th><th>Window scale</th>"
        "<th>Worst network ranks</th><th>Observed total MiB/NIC</th>"
        "<th>Probed total max MiB/NIC</th><th>Theoretical total max MiB/NIC</th>"
        "<th>Action</th><th>Migration budget before MiB/NIC</th>"
        f"<th>Migration budget after MiB/NIC</th></tr></thead><tbody>{rows}</tbody></table></section>"
        "<section><h2>Cross-server expert migration</h2>"
        "<p>RDMA is counted once on the wire. TX and RX totals are shown separately; "
        "summing the per-rank full-duplex max(TX,RX) footprint is not an RDMA byte total.</p>"
        "<table><thead><tr><th>Layer</th><th>MB</th><th>Planned</th>"
        "<th>Admitted</th><th>Deferred</th><th>Remote copies</th>"
        "<th>Moved routes</th><th>RDMA MiB</th><th>Local Weight MiB</th>"
        "<th>TX endpoint MiB</th><th>RX endpoint MiB</th>"
        f"</tr></thead><tbody>{migration_rows}</tbody></table></section>"
        "<section><h2>Combine telemetry (not consumed by controller)</h2>"
        "<table><thead><tr><th>Layer</th><th>MB</th><th>Compute partner</th>"
        "<th>Comm active max us</th><th>Compute max us</th><th>Overlap max us</th>"
        f"</tr></thead><tbody>{combine_rows}</tbody></table></section>"
        "</main></body></html>",
        encoding="utf-8",
    )


def _package(run_dir: Path, layers: int) -> Path:
    zip_path = run_dir / f"probeep_{layers}layer_dynamic_visualization.zip"
    members = [
        run_dir / "probeep_dynamic_dashboard.html",
        run_dir / "probeep_dispatch_observations.json",
        run_dir / "probeep_dispatch_observations.csv",
        run_dir / "probeep_combine_telemetry.json",
        run_dir / "probeep_combine_telemetry.csv",
        run_dir / "测试报告.md",
    ]
    for directory in ("timeline", "link_load", "gate_load"):
        members.extend(
            path for path in (run_dir / directory).rglob("*") if path.is_file()
        )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member, member.relative_to(run_dir))
    for html in run_dir.rglob("*.html"):
        html.unlink()
    return zip_path


def main() -> int:
    args = parse_args()
    require(args.num_layers >= 2, "dynamic wavefront test needs at least two layers")
    layer_map = (
        tuple(int(value.strip()) for value in args.gate_layer_map.split(","))
        if args.gate_layer_map else tuple(range(args.num_layers))
    )
    require(len(layer_map) == args.num_layers, "gate layer map length mismatch")
    mode = mode_config(args.full)
    if args.full:
        compute_path = args.compute_config.resolve()
        require(compute_path.is_file(), f"missing compute config: {compute_path}")
        mode = replace(mode, compute_config=compute_path)
        cost_model = JsonComputeCostModel.from_path(compute_path)
        hardware = cost_model.hardware
    else:
        cost_model = H100CostModel()
        hardware = "H100-theoretical-smoke"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / (
        f"run_{timestamp}_probeep_{args.num_layers}layer_dynamic_{mode.name}"
    )
    workload_dir = run_dir / "workload"
    simulation_dir = run_dir / "simulation"
    metrics_dir = simulation_dir / "output_metrics"
    metrics_dir.mkdir(parents=True)
    build_simulator(run_dir)

    model = ModelSpec(
        name=f"dsv3_probeep_{args.num_layers}layer_dynamic_{mode.name}",
        hidden=7168,
        ffn_hidden=2048,
        num_attention_heads=128,
        num_kv_heads=128,
        head_dim=128,
        num_experts=256,
        topk=8,
        sequence_length=4096,
        num_layers=args.num_layers,
        micro_batches=2,
    )
    placement = Placement(
        NUM_RANKS,
        8,
        make_contiguous_expert_placement(model.num_experts, NUM_RANKS),
    )
    builder = IncrementalTransformerWorkloadBuilder(
        TransformerWorkloadConfig(
            model=model,
            placement=placement,
            tokens_per_rank=mode.tokens_per_rank,
            algorithm="probeep",
            chunk_tokens=mode.chunk_tokens,
            token_padding=128,
            probeep_route_chunk_tokens=mode.chunk_tokens,
            probeep_weight_chunk_bytes=4 * 1024 * 1024,
            probeep_target_overlap_ratio=0.90,
            gate_provider=create_gate_provider(
                "raw_receive_cdf",
                seed=args.gate_seed,
                raw_placement_json=RAW_PLACEMENT,
                layer_map=layer_map,
            ),
            dispatch_dtype="fp8",
            combine_dtype="bf16",
            weight_dtype="bf16",
        ),
        cost_model=cost_model,
    )
    controller = ProbeNICController(NUM_RANKS, CONTROLLER_CONFIG)
    emitter = DynamicDagEmitter(workload_dir, NUM_RANKS)
    first = builder.build_next_layer({
        0: controller.budgets_for("attention"),
        1: controller.budgets_for("moe"),
    })
    first_keys = tuple(
        key for key in first.new_task_keys if key not in first.deferred_final_keys
    )
    initial_batch = emitter.append_tasks(
        builder.graph,
        batch_id="layer0_body",
        layer=0,
        task_keys=first_keys,
        observations=probeep_weight_dispatch_observations(builder.graph, 0),
    )

    command = _command(workload_dir, simulation_dir, mode.simulation_end_us)
    env = os.environ.copy()
    env["HTSIM_LINK_LOAD_SAMPLE"] = "1"
    env["HTSIM_LINK_LOAD_SAMPLE_US"] = str(mode.link_sample_us)
    (simulation_dir / "命令.txt").write_text(
        " ".join(command)
        + f"\nHTSIM_LINK_LOAD_SAMPLE=1\nHTSIM_LINK_LOAD_SAMPLE_US={mode.link_sample_us}\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        command,
        cwd=simulation_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    log_lines: list[str] = []
    starts: dict[int, float] = {}
    dones: dict[int, float] = {}
    records: list[dict[str, object]] = []
    sent_initial = False
    for line in process.stdout:
        log_lines.append(line)
        start_match = TASK_START_RE.search(line)
        if start_match:
            starts[int(start_match.group(1))] = float(start_match.group(2))
        done_match = TASK_DONE_RE.search(line)
        if done_match:
            dones[int(done_match.group(1))] = float(done_match.group(2))
        if "DAG_CONTROL_READY" in line:
            require(not sent_initial, "HTSim requested initial DAG twice")
            sent_initial = True
            _write(process, initial_batch.protocol)
            continue
        observation_match = OBSERVATION_RE.search(line)
        if not observation_match:
            continue
        observation_id = int(observation_match.group(1))
        observation_time = float(observation_match.group(2))
        feedback, measurement = make_feedback(
            observation_id=observation_id,
            observation_time_us=observation_time,
            builder=builder,
            emitter=emitter,
            starts=starts,
            dones=dones,
        )
        update = controller.update(feedback)
        record = {
            **measurement,
            "controller_action": update.action,
            "budget_before_mean_mib": (
                sum(update.budget_before) / NUM_RANKS / 1024 / 1024
            ),
            "budget_after_mean_mib": (
                sum(update.budget_after) / NUM_RANKS / 1024 / 1024
            ),
            "global_network_to_compute_ratio": (
                update.global_network_to_compute_ratio
            ),
            "global_adjustment_factor": update.global_adjustment_factor,
            "network_bottleneck_ranks": list(update.bottleneck_ranks),
            "controller_update": update.manifest(),
        }
        records.append(record)
        layer, micro_batch = divmod(observation_id, 2)
        if micro_batch == 0:
            _write(process, "DAG_CONTINUE\n")
            continue
        if layer + 1 == args.num_layers:
            _write(process, "DAG_CLOSE\n")
            continue

        previous = builder.layer_results[layer]
        current: IncrementalLayerResult = builder.build_next_layer({
            0: controller.budgets_for("attention"),
            1: controller.budgets_for("moe"),
        })
        current_keys = tuple(
            key
            for key in current.new_task_keys
            if key not in current.deferred_final_keys
        )
        batch = emitter.append_tasks(
            builder.graph,
            batch_id=f"layer{layer}_tail_layer{layer + 1}_body",
            layer=layer + 1,
            task_keys=previous.deferred_final_keys + current_keys,
            observations=probeep_weight_dispatch_observations(
                builder.graph, layer + 1
            ),
        )
        _write(process, batch.protocol)

    returncode = process.wait(timeout=mode.timeout_seconds)
    log_text = "".join(log_lines)
    log_path = simulation_dir / "htsim.log"
    log_path.write_text(log_text, encoding="utf-8")
    require(returncode == 0, f"HTSim returned {returncode}")
    require(sent_initial, "dynamic controller handshake never occurred")
    require(
        len(records) == args.num_layers * NUM_MICROBATCHES,
        "not every Dispatch observation was consumed",
    )
    require(log_text.count("DAG_SUMMARY") == 1, "expected one DAG summary")
    require(
        log_text.count("DAG_APPEND_ACK") == args.num_layers,
        "append batch count must equal layer count",
    )
    require("DAG_CONTROL_CLOSED" in log_text, "dynamic DAG was not closed")
    require((metrics_dir / "link_info.csv").is_file(), "link inventory missing")
    require((metrics_dir / "link_load_1ms.csv").is_file(), "link samples missing")

    metadata = builder.metadata()
    metadata["dynamic_dag"]["controller_updates"] = [
        record["controller_update"] for record in records
    ]
    metadata["dynamic_dag"]["observation_count"] = len(records)
    metadata["hardware"] = hardware
    _write_observations(run_dir, records)
    combine_records = _combine_telemetry(
        builder=builder,
        emitter=emitter,
        starts=starts,
        dones=dones,
    )
    _write_combine_telemetry(run_dir, combine_records)
    runtime_validation = _validate_runtime(
        builder=builder,
        emitter=emitter,
        starts=starts,
        dones=dones,
        dispatch_records=records,
        combine_records=combine_records,
        require_remote_migration=args.full,
    )
    metadata["dynamic_dag"]["runtime_validation"] = runtime_validation
    emitter.finalize(graph_name=model.name, metadata=metadata)
    summary_match = re.search(r"DAG_SUMMARY .*makespan_us=([0-9.eE+-]+)", log_text)
    require(summary_match is not None, "DAG summary lacks makespan")
    report = (
        "# ProbeEP 动态 DAG 测试\n\n"
        f"- 模式：`{mode.name}`，{mode.tokens_per_rank} tokens/rank。\n"
        f"- 层数：{args.num_layers}；Dispatch observations：{len(records)}。\n"
        "- HTSim 进程数：1；EventList 和网络/CC 状态未重置。\n"
        f"- append 批次：{args.num_layers}；`DAG_SUMMARY` 数量：1。\n"
        f"- makespan：{float(summary_match.group(1)):.3f} us。\n"
        "- MB0/MB1 分别更新 Attention/MoE budget；Combine 不进入控制器。\n"
        "- 上一层 MB1 Final 与下一层主体同批提交，不回写已执行 task。\n"
        f"- source-rail Weight->Dispatch checks：{runtime_validation['source_rail_weight_before_dispatch_checks']}。\n"
        f"- 跨机 expert replicas：{runtime_validation['remote_replica_count']}。\n"
        f"- 权重 TX/RX/RDMA 字节守恒检查：{runtime_validation['weight_byte_conservation_checks']}。\n"
        f"- 跨层 overlap 边界：{runtime_validation['cross_layer_overlap_boundaries']}；同 rank compute 重叠：0。\n"
        f"- 跨层 Attention 零等待检查：{runtime_validation['cross_layer_attention_release_checks']}；最大 gap：{runtime_validation['cross_layer_attention_release_gap_max_us']:.6f} us。\n"
    )
    (run_dir / "测试报告.md").write_text(report, encoding="utf-8")
    (run_dir / "配置.json").write_text(
        json.dumps(
            {
                "mode": asdict(mode),
                "num_layers": args.num_layers,
                "gate_layer_map": layer_map,
                "hardware": hardware,
                "htsim_process_count": 1,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    if not args.skip_visualizations:
        run_visualizations(
            run_dir, workload_dir, log_path, "probeep_dynamic", mode,
            args.num_layers,
        )
        _write_dashboard(
            run_dir / "probeep_dynamic_dashboard.html",
            records,
            combine_records,
            mode.name,
            args.num_layers,
        )
        _package(run_dir, args.num_layers)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
