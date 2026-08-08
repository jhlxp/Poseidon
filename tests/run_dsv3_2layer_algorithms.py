#!/usr/bin/env python3
"""Run four DSV3 algorithms in smoke or explicit full mode and compare them."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYSRC = ROOT / "pysrc"
SIM_DIR = ROOT / "htsim" / "sim"
BUILD_DIR = SIM_DIR / "build-mprail"
BINARY = BUILD_DIR / "datacenter" / "htsim_uec"
TIMELINE = ROOT / "visualization" / "dag_timeline.py"
LINK_LOAD = ROOT / "visualization" / "mprail_link_load.py"
GATE_LOAD = ROOT / "visualization" / "gate_load_profile.py"
COMPARISON = ROOT / "visualization" / "dsv3_algorithm_comparison.py"
COMPUTE_CONFIG = (
    PYSRC / "compute_profiles" / "H100_DSV3_EP32_compute_4096tpr.json"
)
ALGORITHMS = ("nccl", "deepep", "eplb", "moonep")
sys.path.insert(0, str(PYSRC))
sys.path.insert(0, str(ROOT))

from moe_dag import (  # noqa: E402
    JsonComputeCostModel,
    ModelSpec,
    Placement,
    emit_workload,
    make_contiguous_expert_placement,
)
from moe_dag.models import (  # noqa: E402
    TransformerWorkloadConfig,
    build_transformer_workload,
)
from workload.gate import GATE_PROVIDER_NAMES, create_gate_provider  # noqa: E402


DEFAULT_RAW_PLACEMENT = (
    ROOT
    / "workload"
    / "raw_data"
    / "ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json"
)


@dataclass(frozen=True)
class RunMode:
    name: str
    tokens_per_rank: int
    chunk_tokens: int
    link_sample_us: int
    simulation_end_us: int
    timeout_seconds: int
    compute_config: Path | None


@dataclass(frozen=True)
class GateRun:
    provider: str
    seed: int
    rank_alpha: float | None
    local_alpha: float
    target_rank_imbalance: float
    fast_skew: float
    raw_placement_json: Path
    raw_csv_pattern: str
    layer_map: tuple[int, ...] | None


def mode_config(full: bool) -> RunMode:
    if full:
        return RunMode(
            name="full",
            tokens_per_rank=4096,
            chunk_tokens=4096,
            link_sample_us=100,
            simulation_end_us=1_000_000,
            timeout_seconds=1200,
            compute_config=COMPUTE_CONFIG,
        )
    return RunMode(
        name="smoke",
        tokens_per_rank=2,
        chunk_tokens=32,
        link_sample_us=1,
        simulation_end_us=2000,
        timeout_seconds=180,
        compute_config=None,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use 4096 tokens/rank. Without this flag the run is a 2-token smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel algorithm processes.",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHMS),
        help="Comma-separated subset of nccl,deepep,eplb,moonep.",
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
        "--gate-raw-placement-json", type=Path, default=DEFAULT_RAW_PLACEMENT
    )
    parser.add_argument("--gate-raw-csv-pattern", default="decode_{rank}.csv")
    parser.add_argument(
        "--gate-layer-map",
        help="Comma-separated raw layers; needs two entries for this test.",
    )
    parser.add_argument(
        "--moonep-replicas-per-rank",
        type=int,
        default=2,
        help="MoonEP temporary replica capacity; skewed Gate inputs may need more than 2.",
    )
    parser.add_argument(
        "--simulation-end-us",
        type=int,
        help="Override the mode's HTSim end time without changing token count.",
    )
    return parser.parse_args()


def build_simulator(run_dir: Path) -> None:
    commands = (
        [
            "cmake",
            "-S",
            str(SIM_DIR),
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "htsim_uec",
            "-j4",
        ],
    )
    chunks: list[str] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
        chunks.append("$ " + " ".join(command) + "\n" + completed.stdout)
        if completed.returncode != 0:
            (run_dir / "构建.log").write_text(
                "\n".join(chunks), encoding="utf-8"
            )
            raise RuntimeError(f"HTSim build failed with {completed.returncode}")
    (run_dir / "构建.log").write_text("\n".join(chunks), encoding="utf-8")


def execute_htsim(case_dir: Path, workload_dir: Path, mode: RunMode) -> Path:
    simulation_dir = case_dir / "simulation"
    metrics_dir = simulation_dir / "output_metrics"
    metrics_dir.mkdir(parents=True)
    log_path = simulation_dir / "htsim.log"
    command = [
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
        "-end", str(mode.simulation_end_us),
        "-strat", "ecmp_host",
        "-sender_cc_only",
        "-sender_cc_algo", "nscc",
        "-tm", str(workload_dir / "nodes.cm"),
        "-dag", str(workload_dir / "workload.dag"),
        "-o", str(simulation_dir / "htsim.dat"),
    ]
    env = os.environ.copy()
    env["HTSIM_LINK_LOAD_SAMPLE"] = "1"
    env["HTSIM_LINK_LOAD_SAMPLE_US"] = str(mode.link_sample_us)
    (simulation_dir / "命令.txt").write_text(
        " ".join(command)
        + f"\nHTSIM_LINK_LOAD_SAMPLE=1\nHTSIM_LINK_LOAD_SAMPLE_US={mode.link_sample_us}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        command,
        cwd=simulation_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=mode.timeout_seconds,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"HTSim returned {completed.returncode}")
    require((metrics_dir / "link_info.csv").is_file(), "link inventory is missing")
    require(
        (metrics_dir / "link_load_1ms.csv").is_file(),
        "link load samples are missing",
    )
    return log_path


def run_visualizations(
    case_dir: Path,
    workload_dir: Path,
    log_path: Path,
    algorithm: str,
    mode: RunMode,
) -> None:
    timeline_dir = case_dir / "timeline"
    timeline_command = [
        sys.executable,
        str(TIMELINE),
        "--workload-dir", str(workload_dir),
        "--htsim-log", str(log_path),
        "--output-dir", str(timeline_dir),
        "--gpus-per-server", "8",
        "--ranks", "0-31",
        "--title", f"{algorithm.upper()} / DSV3 2-layer / {mode.name}",
    ]
    timeline_result = subprocess.run(
        timeline_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    (case_dir / "timeline.log").write_text(
        timeline_result.stdout, encoding="utf-8"
    )
    require(timeline_result.returncode == 0, "timeline generation failed")

    link_dir = case_dir / "link_load"
    link_command = [
        sys.executable,
        str(LINK_LOAD),
        "--metrics-dir", str(case_dir / "simulation" / "output_metrics"),
        "--output-dir", str(link_dir),
        "--planes", "1",
        "--title", f"{algorithm.upper()} / DSV3 2-layer / {mode.name}",
    ]
    link_result = subprocess.run(
        link_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    (case_dir / "link_load.log").write_text(
        link_result.stdout, encoding="utf-8"
    )
    require(link_result.returncode == 0, "link-load visualization failed")

    gate_dir = case_dir / "gate_load"
    gate_command = [
        sys.executable,
        str(GATE_LOAD),
        "--workload-dir", str(workload_dir),
        "--output-dir", str(gate_dir),
        "--title", f"{algorithm.upper()} / Gate load / {mode.name}",
    ]
    gate_result = subprocess.run(
        gate_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    (case_dir / "gate_load.log").write_text(
        gate_result.stdout, encoding="utf-8"
    )
    require(gate_result.returncode == 0, "Gate-load visualization failed")


def validate_case(
    case_dir: Path,
    workload_dir: Path,
    log_path: Path,
    algorithm: str,
    mode: RunMode,
) -> dict[str, object]:
    manifest = json.loads(
        (workload_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_map = json.loads(
        (workload_dir / "task_map.json").read_text(encoding="utf-8")
    )
    log = log_path.read_text(encoding="utf-8")
    require(manifest["metadata"]["algorithm"] == algorithm, "algorithm mismatch")
    require(
        manifest["metadata"]["tokens_per_rank"] == mode.tokens_per_rank,
        "tokens_per_rank mismatch",
    )
    require(
        manifest["metadata"]["chunk_tokens"] == mode.chunk_tokens,
        "chunk_tokens mismatch",
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "HTSim did not complete the DAG",
    )
    require(
        "queue_size_bytes 531200" in log,
        "standard DSV3 run did not use the 128-packet queue",
    )
    require(
        "MPRAIL_ECN enabled=yes threshold_bytes=16600 "
        "queue_size_bytes=531200" in log,
        "standard DSV3 run has unexpected ECN/queue settings",
    )
    records = task_map["tasks"]
    transfers = [record for record in records if record["kind"] == "transfer"]
    require(bool(transfers), "workload has no transfers")
    require(
        any(
            record["src_rank"] // 8 != record["dst_rank"] // 8
            for record in transfers
        ),
        "workload has no cross-server transfer",
    )
    server_forward = [record for record in transfers if record["route_spec"]]
    hierarchical_legs = {
        record["metadata"].get("hierarchical_leg")
        for record in transfers
        if record["metadata"].get("hierarchical_leg")
    }
    if algorithm == "nccl":
        require(not server_forward, "NCCL must not use server_forward")
        require(not hierarchical_legs, "NCCL must use direct rank flows")
    else:
        require(not server_forward, f"{algorithm} must use explicit hierarchy legs")
        require(
            {
                "dispatch_fabric",
                "dispatch_local",
                "combine_local_reduce",
                "combine_fabric",
            }
            <= hierarchical_legs,
            f"{algorithm} is missing hierarchical transfer legs",
        )
        for invocation in manifest["metadata"]["micro_batch_algorithms"]:
            require(
                invocation["route_count"]
                > invocation["unique_token_payload_count"]
                > invocation["unique_server_payload_count"],
                f"{algorithm} did not exercise rank/server payload deduplication",
            )
    if mode.name == "full":
        compute_cost = manifest["metadata"]["compute_cost"]
        require(
            Path(compute_cost["config_path"]) == COMPUTE_CONFIG.resolve(),
            "full run did not use the 4096tpr compute config",
        )
        require(
            compute_cost["selected_source"] == "theoretical",
            "unexpected full compute source",
        )
        require(
            compute_cost["model"] == "json_linear_per_token_v2",
            "full run did not use the per-token compute schema",
        )
        for record in records:
            if record["kind"] != "compute":
                continue
            metadata = record["metadata"]
            token_count = metadata.get("compute_token_count")
            us_per_token = metadata.get("compute_us_per_token")
            require(
                isinstance(token_count, int) and token_count > 0,
                f"compute task {record['key']} has no token_count",
            )
            require(
                isinstance(us_per_token, (int, float)) and us_per_token > 0,
                f"compute task {record['key']} has no per-token cost",
            )
            require(
                math.isclose(
                    record["duration_us"],
                    token_count * us_per_token,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"compute task {record['key']} duration is not token-scaled",
            )
    starts = {
        int(task): float(time)
        for task, time in re.findall(
            r"^DAG_TASK_START task=(\d+).*time_us=([0-9.eE+-]+)$",
            log,
            re.MULTILINE,
        )
    }
    dones = {
        int(task): float(time)
        for task, time in re.findall(
            r"^DAG_TASK_DONE task=(\d+).*time_us=([0-9.eE+-]+)$",
            log,
            re.MULTILINE,
        )
    }
    by_key = {record["key"]: record for record in records}
    attention = by_key["mb0.layer1.attention.rank0"]
    require(
        not any(
            predecessor.startswith("mb1.layer0.")
            for predecessor in attention["predecessors"]
        ),
        f"{algorithm} reintroduced a layer drain before MB0 layer1 Attention",
    )

    def touches_rank0(record: dict[str, object]) -> bool:
        if record["src_rank"] == 0 or record["dst_rank"] == 0:
            return True
        route_spec = record.get("route_spec")
        if not isinstance(route_spec, str):
            return False
        match = re.fullmatch(
            r"server_forward src_relay:(\d+) dst_relay:(\d+)",
            route_spec,
        )
        return match is not None and 0 in (
            int(match.group(1)),
            int(match.group(2)),
        )

    layer0_mb1_combine = [
        record
        for record in records
        if record["metadata"].get("stream_phase_id") == "mb1.layer0.combine"
        and touches_rank0(record)
    ]
    require(layer0_mb1_combine, f"{algorithm} has no GPU0 layer0 MB1 Combine")
    combine_interval = (
        min(starts[record["task_id"]] for record in layer0_mb1_combine),
        max(dones[record["task_id"]] for record in layer0_mb1_combine),
    )
    attention_interval = (
        starts[attention["task_id"]],
        dones[attention["task_id"]],
    )
    cross_layer_overlap_us = max(
        0.0,
        min(combine_interval[1], attention_interval[1])
        - max(combine_interval[0], attention_interval[0]),
    )
    require(
        cross_layer_overlap_us > 0,
        f"{algorithm} has no L1 MB0 Attention / L0 MB1 Combine overlap",
    )
    timeline_summary = json.loads(
        (case_dir / "timeline" / "dag_timeline_summary.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        timeline_summary["selected_rank_compute_network_overlap_sum_us"] > 0,
        "timeline has no compute/network overlap",
    )
    require(
        (case_dir / "link_load" / "mprail_link_load_by_layer.png").is_file(),
        "link-load image is missing",
    )
    gate_summary = json.loads(
        (case_dir / "gate_load" / "gate_load_profile_summary.json").read_text(
            encoding="utf-8"
        )
    )
    require(len(gate_summary["records"]) == 4, "Gate profile record count mismatch")
    gate_digests = [
        record["gate"]["assignment_digest_sha256"]
        for record in gate_summary["records"]
    ]
    return {
        "algorithm": algorithm,
        "status": "passed",
        "case_dir": str(case_dir),
        "task_count": manifest["task_count"],
        "transfer_task_count": manifest["transfer_task_count"],
        "logical_transfer_bytes": sum(
            manifest["transfer_bytes_by_payload"].values()
        ),
        "hierarchical_legs": sorted(hierarchical_legs),
        "makespan_us": timeline_summary["makespan_us"],
        "overlap_us": timeline_summary[
            "selected_rank_compute_network_overlap_sum_us"
        ],
        "cross_layer_overlap_us": cross_layer_overlap_us,
        "gate_provider": manifest["metadata"]["routing_provider"]["name"],
        "gate_assignment_digests": gate_digests,
    }


def run_algorithm(
    root_dir: Path,
    algorithm: str,
    mode: RunMode,
    gate: GateRun,
    moonep_replicas_per_rank: int,
) -> dict[str, object]:
    case_dir = root_dir / "algorithms" / algorithm
    case_dir.mkdir(parents=True)
    try:
        model = ModelSpec(
            name=f"dsv3_2layer_{algorithm}_{mode.name}",
            hidden=7168,
            ffn_hidden=2048,
            num_attention_heads=128,
            num_kv_heads=128,
            head_dim=128,
            num_experts=256,
            topk=8,
            sequence_length=4096,
            num_layers=2,
            micro_batches=2,
        )
        placement = Placement(
            32,
            8,
            make_contiguous_expert_placement(model.num_experts, 32),
        )
        cost_model = (
            JsonComputeCostModel.from_path(mode.compute_config)
            if mode.compute_config is not None
            else None
        )
        result = build_transformer_workload(
            TransformerWorkloadConfig(
                model=model,
                placement=placement,
                tokens_per_rank=mode.tokens_per_rank,
                algorithm=algorithm,
                chunk_tokens=mode.chunk_tokens,
                replicas_per_rank=moonep_replicas_per_rank,
                token_padding=128,
                gate_provider=create_gate_provider(
                    gate.provider,
                    seed=gate.seed,
                    rank_alpha=gate.rank_alpha,
                    local_alpha=gate.local_alpha,
                    target_rank_imbalance=gate.target_rank_imbalance,
                    fast_skew=gate.fast_skew,
                    raw_placement_json=gate.raw_placement_json,
                    raw_csv_pattern=gate.raw_csv_pattern,
                    layer_map=gate.layer_map,
                ),
                dispatch_dtype="fp8",
                combine_dtype="bf16",
                weight_dtype="bf16",
            ),
            cost_model=cost_model,
        )
        workload_dir = case_dir / "workload"
        emit_workload(result.graph, workload_dir, metadata=result.metadata)
        log_path = execute_htsim(case_dir, workload_dir, mode)
        run_visualizations(case_dir, workload_dir, log_path, algorithm, mode)
        summary = validate_case(
            case_dir, workload_dir, log_path, algorithm, mode
        )
        (case_dir / "结果.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception as exc:
        failure = {
            "algorithm": algorithm,
            "status": "failed",
            "case_dir": str(case_dir),
            "error": str(exc),
        }
        (case_dir / "结果.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


def write_report(
    run_dir: Path,
    mode: RunMode,
    results: list[dict[str, object]],
    comparison_member: str | None,
    visualization_zip: Path | None,
) -> None:
    passed = sum(item["status"] == "passed" for item in results)
    lines = [
        "# DSV3 两层四算法测试报告",
        "",
        f"- 模式：`{mode.name}`",
        f"- tokens/rank/microbatch：{mode.tokens_per_rank}",
        f"- chunk_tokens：{mode.chunk_tokens}",
        f"- link sample：{mode.link_sample_us} us",
        f"- 通过：{passed}/{len(results)}",
        f"- ZIP 内总览 HTML：`{comparison_member or '未生成'}`",
        f"- 可视化 ZIP：`{visualization_zip.name if visualization_zip else '未生成'}`",
        "",
        "| algorithm | status | makespan us | tasks | transfer bytes | overlap us | cross-layer us |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        makespan = item.get("makespan_us")
        overlap = item.get("overlap_us")
        cross_layer = item.get("cross_layer_overlap_us")
        lines.append(
            f"| {item['algorithm']} | {item['status']} | "
            f"{'-' if makespan is None else f'{float(makespan):.9g}'} | "
            f"{item.get('task_count', '-')} | "
            f"{item.get('logical_transfer_bytes', '-')} | "
            f"{'-' if overlap is None else f'{float(overlap):.9g}'} | "
            f"{'-' if cross_layer is None else f'{float(cross_layer):.9g}'} |"
        )
    (run_dir / "测试报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": asdict(mode),
                "passed": passed,
                "total": len(results),
                "comparison_html_in_zip": comparison_member,
                "visualization_zip": (
                    str(visualization_zip) if visualization_zip else None
                ),
                "algorithms": results,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.moonep_replicas_per_rank < 0:
        raise SystemExit("--moonep-replicas-per-rank must be non-negative")
    algorithms = tuple(
        value.strip() for value in args.algorithms.split(",") if value.strip()
    )
    if not algorithms or len(algorithms) != len(set(algorithms)):
        raise SystemExit("--algorithms must contain unique names")
    unsupported = set(algorithms) - set(ALGORITHMS)
    if unsupported:
        raise SystemExit(f"unsupported algorithms: {sorted(unsupported)}")

    mode = mode_config(args.full)
    if args.simulation_end_us is not None:
        if args.simulation_end_us <= 0:
            raise SystemExit("--simulation-end-us must be positive")
        mode = replace(mode, simulation_end_us=args.simulation_end_us)
    layer_map = None
    if args.gate_layer_map:
        try:
            layer_map = tuple(
                int(value.strip()) for value in args.gate_layer_map.split(",")
            )
        except ValueError as exc:
            raise SystemExit("--gate-layer-map must contain integers") from exc
        if len(layer_map) != 2:
            raise SystemExit("--gate-layer-map needs exactly two entries")
    gate = GateRun(
        provider=args.gate_provider,
        seed=args.gate_seed,
        rank_alpha=args.gate_rank_alpha,
        local_alpha=args.gate_local_alpha,
        target_rank_imbalance=args.gate_target_rank_imbalance,
        fast_skew=args.gate_fast_skew,
        raw_placement_json=args.gate_raw_placement_json.resolve(),
        raw_csv_pattern=args.gate_raw_csv_pattern,
        layer_map=layer_map,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        ROOT
        / "test_logs"
        / f"run_{timestamp}_dsv3_2layer_4algo_{mode.name}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "配置.json").write_text(
        json.dumps(
            {
                "mode": asdict(mode),
                "algorithms": algorithms,
                "workers": min(args.workers, len(algorithms)),
                "gate": asdict(gate),
                "moonep_replicas_per_rank": args.moonep_replicas_per_rank,
                "topology": {
                    "ranks": 32,
                    "servers": 4,
                    "gpus_per_server": 8,
                    "planes": 1,
                    "leaf": 8,
                    "spine": 4,
                    "rdma_gbps": 400,
                    "mtu_bytes": 4150,
                    "queue_packets": 128,
                    "queue_bytes": 531200,
                    "ecn_low_bytes": 16600,
                    "ecn_high_bytes": 53950,
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        build_simulator(run_dir)
    except Exception as exc:
        write_report(
            run_dir,
            mode,
            [
                {"algorithm": algorithm, "status": "failed", "error": str(exc)}
                for algorithm in algorithms
            ],
            None,
            None,
        )
        print(f"build failed: {exc}")
        print(f"log directory: {run_dir}")
        return 1

    by_algorithm: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(algorithms))
    ) as executor:
        futures = {
            executor.submit(
                run_algorithm,
                run_dir,
                algorithm,
                mode,
                gate,
                args.moonep_replicas_per_rank,
            ): algorithm
            for algorithm in algorithms
        }
        for future in as_completed(futures):
            algorithm = futures[future]
            try:
                by_algorithm[algorithm] = future.result()
                print(f"[{algorithm}] passed")
            except Exception as exc:
                by_algorithm[algorithm] = {
                    "algorithm": algorithm,
                    "status": "failed",
                    "case_dir": str(run_dir / "algorithms" / algorithm),
                    "error": str(exc),
                }
                print(f"[{algorithm}] failed: {exc}")

    results = [by_algorithm[algorithm] for algorithm in algorithms]
    if all(item["status"] == "passed" for item in results):
        reference = results[0]["gate_assignment_digests"]
        require(
            all(item["gate_assignment_digests"] == reference for item in results),
            "algorithms did not consume identical Gate assignments",
        )
    comparison_path: Path | None = None
    comparison_member: str | None = None
    visualization_zip: Path | None = None
    if all(item["status"] == "passed" for item in results):
        comparison_path = run_dir / "dsv3_algorithm_comparison.html"
        visualization_zip = run_dir / "dsv3_visualization_bundle.zip"
        command = [
            sys.executable,
            str(COMPARISON),
            "--output", str(comparison_path),
            "--zip-output", str(visualization_zip),
        ]
        for algorithm in algorithms:
            command.extend(
                [
                    "--case",
                    f"{algorithm}={run_dir / 'algorithms' / algorithm}",
                ]
            )
        command.extend(
            [
                "--title",
                f"DSV3 2-layer / EP32 / {mode.name} / 4 algorithms",
            ]
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        (run_dir / "comparison.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0 or not visualization_zip.is_file():
            comparison_path = None
            visualization_zip = None
            results.append(
                {
                    "algorithm": "comparison_html",
                    "status": "failed",
                    "error": completed.stdout.strip(),
                }
            )
        else:
            try:
                with zipfile.ZipFile(visualization_zip) as archive:
                    require(archive.testzip() is None, "visualization ZIP is corrupt")
                    members = archive.namelist()
                    require(
                        "dsv3_algorithm_comparison.html" in members,
                        "visualization ZIP is missing the dashboard",
                    )
                    require(
                        sum(name.endswith(".html") for name in members)
                        == 3 * len(algorithms) + 1,
                        "visualization ZIP has an unexpected HTML inventory",
                    )
                    for algorithm in algorithms:
                        require(
                            f"algorithms/{algorithm}/algorithm_dashboard.html"
                            in members,
                            f"visualization ZIP is missing {algorithm} dashboard",
                        )
                    require(
                        not any(
                            "/workload/" in name
                            or "/simulation/" in name
                            or "output_metrics" in name
                            for name in members
                        ),
                        "visualization ZIP contains simulation artifacts",
                    )
            except (AssertionError, OSError, zipfile.BadZipFile) as exc:
                comparison_path = None
                visualization_zip = None
                results.append(
                    {
                        "algorithm": "visualization_zip",
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    if comparison_path is not None and visualization_zip is not None:
        comparison_member = comparison_path.name
        comparison_path.unlink()
        for algorithm in algorithms:
            (run_dir / "algorithms" / algorithm / "timeline" /
             "dag_gpu_timeline.html").unlink()
            (run_dir / "algorithms" / algorithm / "gate_load" /
             "gate_load_profile.html").unlink()
            (run_dir / "algorithms" / algorithm /
             "algorithm_dashboard.html").unlink()
        require(
            not any(run_dir.rglob("*.html")),
            "successful run left loose HTML files outside the ZIP",
        )

    write_report(run_dir, mode, results, comparison_member, visualization_zip)
    passed = sum(item["status"] == "passed" for item in results)
    print(f"DSV3 algorithm tests: {passed}/{len(results)} passed ({mode.name})")
    print(f"log directory: {run_dir}")
    if visualization_zip:
        print(f"visualization ZIP: {visualization_zip}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
