#!/usr/bin/env python3
"""Validate the Python MoE workload generator and one HTSim execution."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYSRC = ROOT / "pysrc"
sys.path.insert(0, str(PYSRC))

from moe_dag import (  # noqa: E402
    ComputeEstimate,
    H100CostModel,
    JsonComputeCostModel,
    ModelSpec,
    MoEInvocation,
    Placement,
    RoutingAssignment,
    TaskGraph,
    ValidationError,
    emit_workload,
)
from moe_dag.algorithms import (  # noqa: E402
    DeepEPBuilder,
    DeepEPConfig,
    EPLBBuilder,
    EPLBConfig,
    MoonEPBuilder,
    MoonEPConfig,
    NCCLBuilder,
    NCCLConfig,
    TokenPayloadPolicy,
    plan_hierarchical_placement,
    plan_token_payloads,
)
from moe_dag.models import (  # noqa: E402
    TransformerWorkloadConfig,
    build_transformer_workload,
)


SIM_DIR = ROOT / "htsim" / "sim"
BUILD_DIR = SIM_DIR / "build-mprail"
BINARY = BUILD_DIR / "datacenter" / "htsim_uec"


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def fixed_compute(duration_us: float) -> ComputeEstimate:
    return ComputeEstimate(
        operation_flops=max(1, round(duration_us * 989e6)),
        duration_us=duration_us,
        overlaps_communication=False,
        available_sms=132,
        peak_flops_per_second=989e12,
        source="test_fixed_duration",
    )


class FixedOperationCostModel:
    communication_sms = 20
    total_sms = 132

    _durations_us = {
        "attention": 10.0,
        "router_projection": 1.0,
        "expert_ffn": 5.0,
        "combine_reduce": 0.01,
        "combine_final_reduce": 0.01,
        "per_server_planning_proxy": 0.5,
    }

    def estimate(
        self,
        operation_flops: int,
        *,
        operation: str,
        overlaps_communication: bool = False,
    ) -> ComputeEstimate:
        duration_us = self._durations_us[operation]
        available_sms = 112 if overlaps_communication else 132
        return ComputeEstimate(
            operation_flops=operation_flops,
            duration_us=duration_us,
            overlaps_communication=overlaps_communication,
            available_sms=available_sms,
            peak_flops_per_second=operation_flops / duration_us * 1e6,
            source=f"test_fixed_operation:{operation}",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "model": "test_fixed_operation_duration",
            "communication_sms": self.communication_sms,
            "total_sms": self.total_sms,
            "durations_us": self._durations_us,
        }


class Suite:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.results: list[CaseResult] = []

    def run(self, name: str, case: Callable[[], str]) -> None:
        try:
            detail = case()
            self.results.append(CaseResult(name, "passed", detail))
        except Exception as exc:
            self.results.append(CaseResult(name, "failed", str(exc)))


def cost_model_case() -> str:
    cost = H100CostModel()
    require(cost.total_sms == 132, "H100 total SM 应为 132")
    require(cost.overlap_sms == 112, "通信预留 20 SM 后应剩 112 SM")
    require(
        math.isclose(cost.overlap_peak_flops_per_second / 1e12, 839.151515, rel_tol=1e-6),
        "overlap BF16 峰值错误",
    )
    estimate = cost.estimate(989_000_000_000, overlaps_communication=False)
    require(math.isclose(estimate.duration_us, 1000.0), "FLOP 到时间换算错误")
    return "132 SM；通信预留 20 SM；dense BF16 overlap 峰值 839.15 TFLOP/s"


def json_compute_cost_case(run_dir: Path) -> str:
    config_path = (
        PYSRC
        / "compute_profiles"
        / "H100_DSV3_EP32_compute_4096tpr.json"
    )
    theoretical = JsonComputeCostModel.from_path(config_path)
    estimate = theoretical.estimate(
        1_000_000,
        operation="attention",
        overlaps_communication=True,
    )
    require(
        math.isclose(estimate.duration_us, 2188.73965337108),
        "JSON theoretical_us 没有成为固定 compute_us",
    )
    require(estimate.available_sms == 112, "JSON cost model 的 overlap SM 错误")
    require(
        theoretical.manifest()["selected_source"] == "theoretical",
        "JSON cost manifest 没有记录 theoretical 选择",
    )
    profile_payload = json.loads(config_path.read_text(encoding="utf-8"))
    profile_payload["selected_source"] = "profiled"
    profile_payload["modules"]["attention"]["profiled_us"] = 7.5
    profiled_path = run_dir / "generated" / "profiled_compute_test.json"
    profiled_path.parent.mkdir(parents=True, exist_ok=True)
    profiled_path.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    profiled = JsonComputeCostModel.from_path(profiled_path)
    profiled_estimate = profiled.estimate(
        1_000_000,
        operation="attention",
    )
    require(
        profiled_estimate.duration_us == 7.5,
        "JSON profiled_us 没有成为固定 compute_us",
    )
    try:
        JsonComputeCostModel.from_path(
            config_path, selected_source="profiled"
        ).estimate(1_000_000, operation="expert_ffn")
    except ValidationError as exc:
        require("profiled_us is null" in str(exc), "空 profiling 报错信息错误")
    else:
        raise AssertionError("选中的 profiled_us 为空时应拒绝生成")
    return "JSON theoretical/profiled 二选一正确；空 profiling 和缺失值不会静默回退"


def emitter_case(run_dir: Path) -> str:
    graph = TaskGraph("barrier_lowering", 2)
    graph.add_compute("root.a", 0, fixed_compute(10))
    graph.add_compute("root.b", 1, fixed_compute(20))
    graph.add_compute(
        "join", 0, fixed_compute(5), predecessors={"root.a", "root.b"}
    )
    emitted = emit_workload(graph, run_dir / "generated" / "barrier_lowering")
    lines = [
        line
        for line in emitted.dag_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    require(len(lines) == 3, "DAG 应有 3 个 task")
    require(lines[2].endswith("| 0 1"), "join 没有依赖两个前驱 barrier")
    require(
        emitted.matrix_path.read_text(encoding="utf-8")
        == "Nodes 2\nConnections 0\n",
        "空 CM 格式错误",
    )
    return "task dependency 被稳定降低为 3 task/3 barrier，join 前驱为 {0,1}"


def token_payload_policy_case() -> str:
    assignments = (
        RoutingAssignment(0, 7, 0, 0),
        RoutingAssignment(0, 7, 1, 1),
    )
    destinations = {0: 9, 1: 9}
    deduplicated = plan_token_payloads(
        assignments,
        lambda assignment: destinations[assignment.expert_id],
        TokenPayloadPolicy(),
    )
    expanded = plan_token_payloads(
        assignments,
        lambda assignment: destinations[assignment.expert_id],
        TokenPayloadPolicy(deduplicate=False, scope="none"),
    )
    require(len(deduplicated[(0, 9)]) == 1, "默认策略没有按 destination rank 去重")
    require(
        len(deduplicated[(0, 9)][0].routes) == 2,
        "去重 payload 没有保留两条 expert route metadata",
    )
    require(len(expanded[(0, 9)]) == 2, "关闭去重后没有保留 top-k multiplicity")
    return "共享 policy 默认生成 1 个 payload；关闭去重时保留 2 个 route payload"


def eplb_official_example_case() -> str:
    loads = (
        (90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86),
        (20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27),
    )
    expected_physical_to_logical = (
        (5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1),
        (7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1),
    )
    for layer_loads, expected in zip(loads, expected_physical_to_logical):
        plan = plan_hierarchical_placement(
            layer_loads,
            num_physical_experts=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=8,
        )
        require(
            plan.physical_to_logical == expected,
            "EPLB phy2log 与官方 README 示例不一致",
        )
        require(
            sum(plan.logical_count) == 16,
            "EPLB logical replica count 总数错误",
        )
        require(
            all(len(replicas) == plan.logical_count[expert]
                for expert, replicas in enumerate(plan.logical_to_physical)),
            "EPLB log2phy 与 logcnt 不一致",
        )
    return "两层官方示例的 32 个 phy2log 元素逐项一致，log2phy/logcnt 自洽"


def eplb_case(run_dir: Path) -> str:
    estimated_loads = (16.0, 1.0, 8.0, 1.0, 4.0, 1.0, 2.0, 1.0)
    placement = Placement(32, 8, tuple(range(8)))
    assignments: list[RoutingAssignment] = []
    token_id = 0
    for expert, route_count in enumerate(estimated_loads):
        for _ in range(int(route_count)):
            assignments.append(RoutingAssignment(0, token_id, 0, expert))
            token_id += 1
    invocation = MoEInvocation(
        "eplb_hot",
        placement,
        (token_id,) + (0,) * 31,
        64,
        128,
        1,
        "fp8",
        "bf16",
        "bf16",
        tuple(assignments),
    )

    graph = TaskGraph("eplb_deepep_steady_state", 32)
    result = EPLBBuilder(
        H100CostModel(),
        EPLBConfig(
            num_physical_experts=32,
            num_groups=4,
            chunk_tokens=4,
            estimated_loads=estimated_loads,
            load_source="test_explicit_snapshot",
        ),
    ).build(graph, invocation)
    graph.validate()

    require(result.metadata["logical_count"] == (7, 1, 7, 1, 6, 2, 5, 3),
            "EPLB 热点 expert replica 数量错误")
    require(set(result.metadata["physical_to_rank"]) == set(range(32)),
            "每 rank 一个 physical expert slot 的映射错误")
    require(max(result.metadata["route_count_by_rank"].values()) == 3,
            "round-robin selector 没有摊开热点 route")
    require(result.metadata["weight_migration_modeled"] is False,
            "稳态 EPLB 错误生成 placement migration")
    require(
        not any(task.payload_kind == "expert_weight_prefetch" for task in graph.tasks),
        "稳态 EPLB 不应生成 MoonEP invocation-level prefetch",
    )
    require(
        result.metadata["hierarchical_transfer"]["task_count_by_leg"].get(
            "dispatch_fabric", 0
        ) > 0,
        "EPLB 跨服务器 token flow 没有复用 DeepEP 分层传输",
    )

    deepep_graph = TaskGraph("eplb_baseline_deepep", 32)
    deepep_result = DeepEPBuilder(
        H100CostModel(), DeepEPConfig(chunk_tokens=4)
    ).build(deepep_graph, invocation)
    deepep_load = max(
        task.metadata["real_token_routes"]
        for task in deepep_graph.tasks
        if task.metadata.get("operation") == "expert_ffn"
    )
    require(deepep_load == 16, "DeepEP baseline 热点 rank route count 错误")
    require(
        max(result.metadata["route_count_by_rank"].values()) < deepep_load,
        "EPLB placement 没有降低该测试快照的最大 rank route count",
    )
    require(
        result.metadata["route_count"]
        >= result.metadata["unique_token_payload_count"]
        >= result.metadata["unique_server_payload_count"],
        "EPLB 分层去冗余计数关系错误",
    )

    emit_workload(
        graph,
        run_dir / "generated" / "eplb_deepep_steady_state",
        metadata=result.metadata,
    )
    return "热点 rank routes 从 16 降至 3；稳态无迁移，并复用 DeepEP 分层去冗余"


def deepep_destination_forward_case(run_dir: Path) -> str:
    placement = Placement(32, 8, (9, 9, 10))
    assignments = tuple(
        RoutingAssignment(0, token, slot, slot)
        for token in range(3)
        for slot in range(3)
    )
    invocation = MoEInvocation(
        "deepep_exact",
        placement,
        (3,) + (0,) * 31,
        128,
        256,
        3,
        "fp8",
        "bf16",
        "bf16",
        assignments,
    )
    graph = TaskGraph("deepep_destination_forward", 32)
    result = DeepEPBuilder(H100CostModel(), DeepEPConfig(chunk_tokens=1)).build(
        graph, invocation
    )
    graph.validate()

    tasks = graph.tasks
    dispatch = [
        task for task in tasks if task.payload_kind == "dispatch_hidden"
    ]
    combine = [
        task for task in tasks if task.payload_kind == "combine_partial"
    ]
    legs = result.metadata["hierarchical_transfer"]["task_count_by_leg"]
    require(legs["dispatch_fabric"] == 3, "每 token 应只有一条 dispatch RDMA")
    require(legs["dispatch_local"] == 6, "dispatch 本地 fanout 数量错误")
    require(legs["combine_local_reduce"] == 6, "combine 本地汇聚数量错误")
    require(legs["combine_fabric"] == 3, "每 token 应只有一条 combine RDMA")
    require(len(dispatch) == 9, "dispatch 分层 task 数量错误")
    require(len(combine) == 9, "combine 分层 task 数量错误")
    require(
        sum(
            task.transfer_bytes
            for task in dispatch
            if task.metadata["hierarchical_leg"] == "dispatch_fabric"
        ) == 3 * 128,
        "跨机 dispatch 没有按目标服务器去冗余",
    )
    require(
        sum(
            task.transfer_bytes
            for task in combine
            if task.metadata["hierarchical_leg"] == "combine_fabric"
        ) == 3 * 128 * 2,
        "跨机 combine 没有先在专家服务器内汇聚",
    )
    require(
        all(
            task.src_rank == 0
            and task.dst_rank == 8
            and task.route_spec is None
            for task in dispatch
            if task.metadata["hierarchical_leg"] == "dispatch_fabric"
        ),
        "dispatch RDMA 没有到达同 index relay rank 8",
    )
    require(
        all(
            task.src_rank == 8
            and task.dst_rank == 0
            and task.route_spec is None
            for task in combine
            if task.metadata["hierarchical_leg"] == "combine_fabric"
        ),
        "combine RDMA 没有从专家服务器 relay 返回 origin",
    )
    require(
        graph.task("deepep_exact.expert.rank9").metadata["real_token_routes"] == 6
        and graph.task("deepep_exact.expert.rank10").metadata["real_token_routes"] == 3,
        "去冗余错误减少了 expert route 计算",
    )
    require(result.metadata["route_count"] == 9, "route_count 汇总错误")
    require(result.metadata["unique_token_payload_count"] == 6, "rank payload 汇总错误")
    require(result.metadata["unique_server_payload_count"] == 3,
            "server payload 汇总错误")
    require(result.metadata["deduplicated_route_count"] == 3, "去重 route 汇总错误")
    require(result.metadata["scaleout_deduplicated_route_count"] == 6,
            "scale-out 去冗余汇总错误")

    emitted = emit_workload(
        graph,
        run_dir / "generated" / "deepep_destination_forward",
        metadata=result.metadata,
    )
    require(emitted.dag_path.exists(), "DeepEP DAG 未生成")
    dag = emitted.dag_path.read_text(encoding="utf-8")
    require("server_forward" not in dag, "分层 task 不应再封装 server_forward")
    return "9 expert routes -> 6 rank payloads -> 3 scale-out payloads；combine 先本地汇聚"


def deepep_cli_case(run_dir: Path) -> str:
    output_dir = run_dir / "generated" / "deepep_cli"
    command = [
        sys.executable,
        str(PYSRC / "generate_moe_dag.py"),
        "--output", str(output_dir),
        "--algorithm", "deepep",
        "--num-ranks", "8",
        "--gpus-per-server", "4",
        "--num-experts", "8",
        "--tokens-per-rank", "2",
        "--topk", "2",
        "--hidden", "128",
        "--ffn-hidden", "256",
        "--sequence-length", "16",
        "--micro-batches", "1",
        "--chunk-tokens", "1",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    (run_dir / "deepep_direct_cli.log").write_text(
        "$ " + " ".join(command) + "\n" + completed.stdout,
        encoding="utf-8",
    )
    require(completed.returncode == 0, f"CLI 返回码 {completed.returncode}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    task_map = json.loads((output_dir / "task_map.json").read_text(encoding="utf-8"))
    require(manifest["metadata"]["algorithm"] == "deepep", "CLI 算法记录错误")
    payloads = {
        task["payload_kind"]
        for task in task_map["tasks"]
        if task["kind"] == "transfer"
    }
    require(payloads == {"dispatch_hidden", "combine_partial"},
            f"DeepEP 出现非核心 payload: {sorted(payloads)}")
    hierarchical_legs = {
        task["metadata"].get("hierarchical_leg")
        for task in task_map["tasks"]
        if task["kind"] == "transfer"
    }
    require(
        {"dispatch_fabric", "dispatch_local", "combine_local_reduce", "combine_fabric"}
        <= hierarchical_legs,
        "CLI 生成物缺少 DeepEP 分层传输 leg",
    )
    return "CLI 生成 DeepEP 核心 workload，并输出显式 RDMA/NVLink 分层 task"


def algorithm_cli_case(run_dir: Path) -> str:
    for algorithm in ("nccl", "eplb", "moonep"):
        output_dir = run_dir / "generated" / f"{algorithm}_cli"
        command = [
            sys.executable,
            str(PYSRC / "generate_moe_dag.py"),
            "--output", str(output_dir),
            "--algorithm", algorithm,
            "--num-ranks", "8",
            "--gpus-per-server", "4",
            "--num-experts", "8",
            "--tokens-per-rank", "2",
            "--topk", "2",
            "--hidden", "128",
            "--ffn-hidden", "256",
            "--sequence-length", "16",
            "--micro-batches", "1",
            "--chunk-tokens", "2",
            "--replicas-per-rank", "1",
            "--token-padding", "1",
        ]
        if algorithm == "eplb":
            command.extend([
                "--eplb-num-physical-experts", "16",
                "--eplb-num-groups", "2",
                "--eplb-loads", "20,1,8,1,4,1,2,1",
            ])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        (run_dir / f"{algorithm}_cli.log").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout,
            encoding="utf-8",
        )
        require(completed.returncode == 0,
                f"{algorithm} CLI 返回码 {completed.returncode}")
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        require(manifest["metadata"]["algorithm"] == algorithm,
                f"{algorithm} CLI manifest 算法记录错误")
        if algorithm == "eplb":
            algorithm_metadata = manifest["metadata"]["micro_batch_algorithms"][0]
            require(
                algorithm_metadata["load_source"] == "cli_explicit_snapshot",
                "EPLB CLI 没有记录显式 load snapshot 来源",
            )
            require(
                algorithm_metadata["weight_migration_modeled"] is False,
                "EPLB CLI 稳态 workload 错误包含 weight migration",
            )
    return "CLI 成功生成 nccl、eplb 与 moonep workload，manifest 算法名正确"


def nccl_case(run_dir: Path) -> str:
    placement = Placement(32, 8, (9, 9))
    assignments = tuple(
        RoutingAssignment(0, token, slot, slot)
        for token in range(3)
        for slot in range(2)
    )
    invocation = MoEInvocation(
        "nccl_exact",
        placement,
        (3,) + (0,) * 31,
        128,
        256,
        2,
        "fp8",
        "bf16",
        "bf16",
        assignments,
    )
    graph = TaskGraph("nccl_rank_direct", 32)
    result = NCCLBuilder(H100CostModel(), NCCLConfig(chunk_routes=1)).build(
        graph, invocation
    )
    graph.validate()

    dispatch = [
        task for task in graph.tasks if task.payload_kind == "dispatch_hidden"
    ]
    combine = [
        task for task in graph.tasks if task.payload_kind == "combine_partial"
    ]
    require(len(dispatch) == 6 and len(combine) == 6,
            "NCCL 没有保留 6 条 top-k route multiplicity")
    require(sum(task.transfer_bytes for task in dispatch) == 6 * 128,
            "NCCL dispatch 字节没有按 route count 计算")
    require(sum(task.transfer_bytes for task in combine) == 6 * 128 * 2,
            "NCCL combine 字节没有按 route count 计算")
    require(all(task.route_spec is None for task in dispatch + combine),
            "NCCL 错误使用了 server_forward")
    require(all(task.src_rank == 0 and task.dst_rank == 9 for task in dispatch),
            "NCCL dispatch 没有直达真实 dst rank")
    require(result.metadata["route_count"] == 6, "NCCL route_count 汇总错误")
    require(result.metadata["token_payload_count"] == 6,
            "NCCL 错误执行了 payload 去冗余")

    emitted = emit_workload(
        graph,
        run_dir / "generated" / "nccl_rank_direct",
        metadata=result.metadata,
    )
    require("server_forward" not in emitted.dag_path.read_text(encoding="utf-8"),
            "NCCL DAG 出现 server_forward")
    return "6 条 top-k routes 生成 6 份 direct dispatch/combine payload，无去重和 relay"


def moonep_case(run_dir: Path) -> str:
    placement = Placement(4, 4, (0, 1, 2, 3))
    assignments = tuple(
        RoutingAssignment(src, token, 0, 0)
        for src in range(4)
        for token in range(2)
    )
    invocation = MoEInvocation(
        "moonep_hot",
        placement,
        (2, 2, 2, 2),
        64,
        128,
        1,
        "fp8",
        "bf16",
        "bf16",
        assignments,
    )
    graph = TaskGraph("moonep_hot_expert", 4)
    result = MoonEPBuilder(
        H100CostModel(),
        MoonEPConfig(replicas_per_rank=1, token_padding=2, chunk_tokens=2),
    ).build(graph, invocation)
    graph.validate()
    real_routes = result.metadata["real_routes_by_rank"]
    require(set(real_routes.values()) == {2}, "MoonEP 没有把 real routes 均衡到每个 rank")
    prefetch = [
        task for task in graph.tasks if task.payload_kind == "expert_weight_prefetch"
    ]
    require(len(prefetch) == 3, "hot expert 应复制到其余 3 个 rank")
    require(
        sum(task.transfer_bytes for task in prefetch)
        == 3 * invocation.expert_weight_bytes,
        "expert gate/up/down 权重预取字节错误",
    )
    emit_workload(
        graph,
        run_dir / "generated" / "moonep_hot_expert",
        metadata=result.metadata,
    )
    return "hot expert 复制到 3 个 rank；每 rank 2 real routes；权重预取字节完整"


def moonep_scaleout_case(run_dir: Path) -> str:
    placement = Placement(32, 8, (9, 17))
    assignments = tuple(
        [RoutingAssignment(0, token, 0, 0) for token in range(16)]
        + [RoutingAssignment(1, token, 0, 1) for token in range(8)]
    )
    invocation = MoEInvocation(
        "moonep_scaleout",
        placement,
        (16, 8) + (0,) * 30,
        64,
        128,
        1,
        "fp8",
        "bf16",
        "bf16",
        assignments,
    )
    graph = TaskGraph("moonep_deepep_scaleout", 32)
    result = MoonEPBuilder(
        H100CostModel(),
        MoonEPConfig(replicas_per_rank=1, token_padding=1, chunk_tokens=4),
    ).build(graph, invocation)
    graph.validate()

    real_routes = result.metadata["real_routes_by_rank"]
    require({real_routes[rank] for rank in range(8, 16)} == {2},
            "server 1 没有独立均衡为每 rank 2 routes")
    require({real_routes[rank] for rank in range(16, 24)} == {1},
            "server 2 没有独立均衡为每 rank 1 route")
    require(all(real_routes[rank] == 0 for rank in (*range(0, 8), *range(24, 32))),
            "无目标 expert 的服务器错误获得了 execution routes")
    require(result.metadata["routes_by_server"] == {0: 0, 1: 16, 2: 8, 3: 0},
            "per-server route 汇总错误")

    replicas = result.metadata["replicas"]
    require(len(replicas) == 14, "两个 hot experts 应各创建 7 个本地 replica")
    require(
        all(
            placement.rank_server(item["home_rank"])
            == placement.rank_server(item["execution_rank"])
            == item["home_server"]
            for item in replicas
        ),
        "MoonEP replica 跨越了 expert home server",
    )
    prefetch = [
        task for task in graph.tasks if task.payload_kind == "expert_weight_prefetch"
    ]
    require(len(prefetch) == 14, "expert weight prefetch flow 数量错误")
    require(all(task.route_spec is None for task in prefetch),
            "本地 expert prefetch 错误使用 server_forward")
    require(
        all(
            placement.rank_server(task.src_rank) == placement.rank_server(task.dst_rank)
            for task in prefetch
        ),
        "expert prefetch flow 跨服务器",
    )
    require(
        sum(task.transfer_bytes for task in prefetch)
        == 14 * invocation.expert_weight_bytes,
        "expert prefetch 总字节错误",
    )

    token_flows = [
        task
        for task in graph.tasks
        if task.payload_kind in {"dispatch_hidden", "combine_partial"}
    ]
    leg_counts = result.metadata["hierarchical_transfer"]["task_count_by_leg"]
    require(
        len(token_flows) == sum(leg_counts.values()),
        "MoonEP 分层 token flow 数量错误",
    )
    require(all(task.route_spec is None for task in token_flows),
            "MoonEP 分层 token flow 错误封装为 server_forward")
    require(leg_counts.get("dispatch_fabric", 0) > 0,
            "MoonEP 缺少 DeepEP dispatch RDMA")
    require(leg_counts.get("combine_fabric", 0) > 0,
            "MoonEP 缺少 DeepEP combine RDMA")
    require(
        graph.task("moonep_scaleout.expert.rank8").duration_us
        > graph.task("moonep_scaleout.expert.rank16").duration_us,
        "不同服务器 route 负载没有产生不同 compute_us",
    )

    emitted = emit_workload(
        graph,
        run_dir / "generated" / "moonep_deepep_scaleout",
        metadata=result.metadata,
    )
    require("server_forward" not in emitted.dag_path.read_text(encoding="utf-8"),
            "MoonEP DAG 错误保留 server_forward 封装")
    return "server1 每 rank 2 routes、server2 每 rank 1 route；14 个本地 replica flows，并复用 DeepEP 分层传输"


def model_pipeline_case(run_dir: Path) -> str:
    model = ModelSpec(
        name="two_microbatch_block",
        hidden=128,
        ffn_hidden=256,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=128,
        num_experts=8,
        topk=2,
        sequence_length=16,
        micro_batches=2,
    )
    placement = Placement(8, 8, tuple(range(8)))
    result = build_transformer_workload(
        TransformerWorkloadConfig(
            model=model,
            placement=placement,
            tokens_per_rank=4,
            algorithm="deepep",
            chunk_tokens=2,
        )
    )
    attention_1 = result.graph.task("mb1.attention.rank0")
    require(
        attention_1.predecessors == {"mb0.router.rank0"},
        "Attention 2 应在同卡 compute stream 的 Router 1 后启动",
    )
    require(
        attention_1.overlaps_communication and attention_1.available_sms == 112,
        "与 Dispatch 1 重叠的 Attention 2 应按 112 SM 计算",
    )
    dispatch_0 = next(
        task
        for task in result.graph.tasks
        if task.key.startswith("mb0.moe.dispatch") and task.kind == "transfer"
    )
    require(
        all(key.startswith("mb0.router.") for key in dispatch_0.predecessors),
        "Dispatch 1 应只等待 Router 1 集合",
    )
    dispatch_1 = next(
        task
        for task in result.graph.tasks
        if task.key.startswith("mb1.moe.dispatch")
        and task.src_rank == 0
    )
    require(
        any(key.startswith("mb0.moe.dispatch") for key in dispatch_1.predecessors),
        "Dispatch 2 没有等待同端点 comm stream 的 Dispatch 1 phase",
    )
    expert_0 = result.graph.task("mb0.moe.expert.rank0")
    expert_1 = result.graph.task("mb1.moe.expert.rank0")
    reduce_0 = result.graph.task("mb0.moe.combine_reduce.rank0")
    reduce_1 = result.graph.task("mb1.moe.combine_reduce.rank0")
    require(
        "mb1.router.rank0" in expert_0.predecessors,
        "Expert 1 没有排在 Attention/Router 2 之后",
    )
    require(
        "mb0.moe.expert.rank0" in expert_1.predecessors,
        "Expert 2 没有在单 compute stream 上等待 Expert 1",
    )
    require(
        "mb1.moe.expert.rank0" in reduce_0.predecessors,
        "Reduce 1 没有排在 Expert 2 之后",
    )
    require(
        "mb0.moe.combine_reduce.rank0" in reduce_1.predecessors,
        "Reduce 2 没有在单 compute stream 上等待 Reduce 1",
    )
    compute_sequence = [
        result.graph.task(key).metadata["stream_sequence"]
        for key in (
            "mb0.attention.rank0",
            "mb0.router.rank0",
            "mb1.attention.rank0",
            "mb1.router.rank0",
            "mb0.moe.expert.rank0",
            "mb1.moe.expert.rank0",
            "mb0.moe.combine_reduce.rank0",
            "mb1.moe.combine_reduce.rank0",
        )
    ]
    require(compute_sequence == list(range(8)), "compute stream 顺序编号错误")
    dispatch_phase = [
        task
        for task in result.graph.tasks
        if task.metadata.get("stream_phase_id") == "mb0.dispatch"
    ]
    require(len(dispatch_phase) > 1, "Dispatch 1 没有多个并行 flow")
    dispatch_keys = {task.key for task in dispatch_phase}
    for task in dispatch_phase:
        intra_phase = task.predecessors & dispatch_keys
        if task.metadata.get("hierarchical_leg") == "dispatch_local":
            require(
                all(
                    result.graph.task(key).metadata.get("hierarchical_leg")
                    == "dispatch_fabric"
                    for key in intra_phase
                ),
                "dispatch local leg 存在非 fabric 的错误串行依赖",
            )
        else:
            require(not intra_phase, "独立 dispatch flow 被错误串行化")
    emitted = emit_workload(
        result.graph,
        run_dir / "generated" / "two_microbatch_block",
        metadata=result.metadata,
    )
    manifest = json.loads(emitted.manifest_path.read_text(encoding="utf-8"))
    require(manifest["metadata"]["scope"]["dynamic_gpu_resource_scheduling"] is False,
            "manifest 没有声明 compute resource 模型边界")
    require(
        manifest["metadata"]["stream_schedule"]["model"]
        == "per_rank_two_stream_double_buffered_v1",
        "manifest 没有记录 two-stream schedule",
    )
    task_map = json.loads(emitted.task_map_path.read_text(encoding="utf-8"))
    require(
        {task["logical_stream"] for task in task_map["tasks"]}
        == {"compute", "communication"},
        "task_map 没有明确两类 logical stream",
    )
    return "每 GPU 一条 compute/comm stream；D0||A1、D1||E0、C0||E1，独立 flow 并行"


def relay_stream_participant_case() -> str:
    model = ModelSpec(
        name="relay_stream_participant",
        hidden=128,
        ffn_hidden=256,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=128,
        num_experts=8,
        topk=2,
        sequence_length=16,
        micro_batches=2,
    )
    result = build_transformer_workload(
        TransformerWorkloadConfig(
            model=model,
            placement=Placement(8, 4, tuple(range(8))),
            tokens_per_rank=4,
            algorithm="deepep",
            chunk_tokens=2,
        )
    )
    dispatch_0 = next(
        task
        for task in result.graph.tasks
        if task.key.startswith("mb0.moe.dispatch")
        and task.metadata.get("hierarchical_leg") == "dispatch_local"
        and isinstance(task.metadata.get("relay_rank"), int)
        and task.metadata.get("relay_rank") != task.dst_rank
    )
    relay = dispatch_0.metadata["relay_rank"]
    require(isinstance(relay, int), "dispatch local leg 缺少 relay rank")
    require(
        any(
            result.graph.task(key).metadata.get("hierarchical_leg")
            == "dispatch_fabric"
            for key in dispatch_0.predecessors
        ),
        "dispatch local leg 没有等待对应 fabric leg",
    )
    expert = result.graph.task(f"mb0.moe.expert.rank{dispatch_0.dst_rank}")
    require(
        dispatch_0.key in expert.predecessors,
        "目标 GPU 的 Expert 没有等待其 Dispatch local leg",
    )
    dispatch_1 = next(
        task
        for task in result.graph.tasks
        if task.key.startswith("mb1.moe.dispatch")
        and (
            task.src_rank == relay
            or task.dst_rank == relay
            or task.metadata.get("relay_rank") == relay
        )
    )
    require(
        dispatch_0.key in dispatch_1.predecessors,
        "relay GPU 的 comm stream 没有串行两个 Dispatch phase",
    )
    return f"relay GPU{relay} 的显式 fabric/local legs 被计入 communication stream"


def multi_layer_group_schedule_case() -> str:
    model = ModelSpec(
        name="two_layer_four_microbatch",
        hidden=128,
        ffn_hidden=256,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=128,
        num_experts=8,
        topk=2,
        sequence_length=16,
        num_layers=2,
        micro_batches=4,
    )
    result = build_transformer_workload(
        TransformerWorkloadConfig(
            model=model,
            placement=Placement(8, 8, tuple(range(8))),
            tokens_per_rank=2,
            algorithm="deepep",
            chunk_tokens=2,
        )
    )
    graph = result.graph
    require(
        result.metadata["model"]["num_layers"] == 2,
        "manifest metadata 没有记录两层模型",
    )
    require(
        result.metadata["stream_schedule"]["layer_count"] == 2,
        "stream schedule 没有记录两层 lowering",
    )
    require(
        result.metadata["stream_schedule"]["schedule_order"]
        == "group_then_layer_wavefront",
        "stream schedule 没有使用跨层 wavefront",
    )
    require(
        len(result.metadata["micro_batch_algorithms"]) == 8,
        "4 microbatch x 2 layer 应产生 8 次 MoE invocation",
    )

    layer1_attention = graph.task("mb0.layer1.attention.rank0")
    require(
        "mb0.layer0.moe.combine_reduce.rank0" in layer1_attention.predecessors,
        "第二层 Attention 没有等待同 microbatch 第一层输出",
    )
    require(
        "mb1.layer0.moe.combine_reduce.rank0"
        not in layer1_attention.predecessors,
        "MB0 第二层 Attention 错误等待 MB1 第一层尾部",
    )
    rank0_compute = sorted(
        (task for task in graph.tasks if task.rank == 0),
        key=lambda task: task.metadata["stream_sequence"],
    )
    rank0_order = [task.key for task in rank0_compute]
    wavefront = (
        "mb0.layer0.moe.combine_reduce.rank0",
        "mb0.layer1.attention.rank0",
        "mb1.layer0.moe.combine_reduce.rank0",
        "mb1.layer1.attention.rank0",
    )
    require(
        [rank0_order.index(key) for key in wavefront]
        == sorted(rank0_order.index(key) for key in wavefront),
        "compute stream 跨层 wavefront 顺序错误",
    )
    next_group = graph.task("mb2.layer0.attention.rank0")
    previous_group_terminals = {
        f"mb{micro_batch}.layer1.moe.combine_reduce.rank{rank}"
        for micro_batch in (0, 1)
        for rank in range(8)
    }
    require(
        previous_group_terminals <= next_group.predecessors,
        "第二个 double-buffer group 没有等待前一组全局 drain",
    )
    require(
        not next_group.overlaps_communication and next_group.available_sms == 132,
        "每组第一个 microbatch 的 Attention 应使用 132 SM",
    )
    require(
        graph.task("mb3.layer0.attention.rank0").available_sms == 112,
        "每组第二个 microbatch 的 Attention 应使用 112 SM",
    )
    layer_phase_ids = {
        task.metadata.get("stream_phase_id")
        for task in graph.tasks
        if task.kind == "transfer"
    }
    require(
        "mb0.layer0.dispatch" in layer_phase_ids
        and "mb1.layer1.combine" in layer_phase_ids,
        "多层 communication phase ID 缺少 layer 坐标",
    )
    return (
        "2 层 x 4 microbatch 按 pair 分组；同 MB 跨层依赖、"
        "MB0 wavefront 和组间全局 drain 正确"
    )


def build_simulator(run_dir: Path) -> None:
    commands = [
        ["cmake", "-S", str(SIM_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(BUILD_DIR), "--target", "htsim_uec", "-j4"],
    ]
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
            (run_dir / "构建.log").write_text("\n".join(chunks), encoding="utf-8")
            raise RuntimeError(f"HTSim 构建失败，返回码 {completed.returncode}")
    (run_dir / "构建.log").write_text("\n".join(chunks), encoding="utf-8")


def execute_htsim(
    run_dir: Path, name: str, matrix_path: Path, dag_path: Path
) -> str:
    case_dir = run_dir / "cases" / name
    (case_dir / "output_metrics").mkdir(parents=True, exist_ok=True)
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
        "-q", "32",
        "-end", "2000",
        "-strat", "ecmp_host",
        "-tm", str(matrix_path),
        "-dag", str(dag_path),
        "-o", str(case_dir / "htsim.dat"),
    ]
    (case_dir / "命令.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    completed = subprocess.run(
        command,
        cwd=case_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    (case_dir / "htsim.log").write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"HTSim 返回码 {completed.returncode}")
    return completed.stdout


def htsim_case(run_dir: Path) -> str:
    graph = TaskGraph("htsim_generated_overlap", 8)
    graph.add_compute("attention.0", 0, fixed_compute(10))
    graph.add_transfer(
        "dispatch.0",
        0,
        4,
        65536,
        "dispatch_hidden",
        "dispatch:rank:0",
        predecessors={"attention.0"},
    )
    graph.add_compute(
        "attention.1", 0, fixed_compute(30), predecessors={"attention.0"}
    )
    graph.add_compute(
        "expert.0", 4, fixed_compute(5), predecessors={"dispatch.0"}
    )
    emitted = emit_workload(graph, run_dir / "generated" / "htsim_overlap")
    log = execute_htsim(
        run_dir,
        "htsim_generated_dag",
        emitted.matrix_path,
        emitted.dag_path,
    )
    require("DAG_SUMMARY tasks=4 barriers=4" in log, "生成 DAG 未完整结束")
    starts = {
        int(task): float(time)
        for task, time in re.findall(
            r"^DAG_TASK_START task=(\d+).*time_us=([0-9.]+)$",
            log,
            re.MULTILINE,
        )
    }
    require(starts[2] == 10 and starts[3] == 10,
            "Dispatch 与 Attention 2 没有在 Attention 1 后同时启动")
    require(starts[4] >= starts[2], "Expert 在 Dispatch 前启动")
    return "HTSim 完成 4 task/4 barrier；10us 时 Dispatch 与下一 Attention 同时启动"


def htsim_two_stream_model_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "two_microbatch_block"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    task_map = json.loads((generated / "task_map.json").read_text(encoding="utf-8"))
    log = execute_htsim(
        run_dir,
        "htsim_two_microbatch_two_stream",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "双 stream 两 microbatch DAG 未完整结束",
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
    require(len(starts) == len(dones) == manifest["task_count"],
            "双 stream HTSim task 时间戳不完整")

    records = task_map["tasks"]
    rank0_compute = sorted(
        (record for record in records if record["rank"] == 0),
        key=lambda record: record["metadata"]["stream_sequence"],
    )
    for previous, current in zip(rank0_compute, rank0_compute[1:]):
        require(
            dones[previous["task_id"]] <= starts[current["task_id"]],
            f"GPU0 compute stream overlap: {previous['key']} / {current['key']}",
        )

    def phase_interval(phase_id: str) -> tuple[float, float]:
        phase = [
            record
            for record in records
            if record["metadata"].get("stream_phase_id") == phase_id
            and (record["src_rank"] == 0 or record["dst_rank"] == 0)
        ]
        require(bool(phase), f"GPU0 缺少 phase {phase_id}")
        return (
            min(starts[record["task_id"]] for record in phase),
            max(dones[record["task_id"]] for record in phase),
        )

    phase_ids = ("mb0.dispatch", "mb1.dispatch", "mb0.combine", "mb1.combine")
    phase_intervals = [phase_interval(phase_id) for phase_id in phase_ids]
    for previous, current in zip(phase_intervals, phase_intervals[1:]):
        require(previous[1] <= current[0], "GPU0 communication phases 发生重叠")

    by_key = {record["key"]: record for record in records}

    def compute_interval(key: str) -> tuple[float, float]:
        task_id = by_key[key]["task_id"]
        return starts[task_id], dones[task_id]

    def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
        return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))

    windows = (
        (phase_intervals[0], compute_interval("mb1.attention.rank0"), "D0||A1"),
        (phase_intervals[1], compute_interval("mb0.moe.expert.rank0"), "D1||E0"),
        (phase_intervals[2], compute_interval("mb1.moe.expert.rank0"), "C0||E1"),
        (
            phase_intervals[3],
            compute_interval("mb0.moe.combine_reduce.rank0"),
            "C1||R0",
        ),
    )
    for communication, compute, name in windows:
        require(overlap(communication, compute) > 0, f"缺少预期 overlap {name}")

    dispatch0 = [
        record
        for record in records
        if record["metadata"].get("stream_phase_id") == "mb0.dispatch"
        and record["src_rank"] == 0
    ]
    require(len(dispatch0) > 1, "GPU0 Dispatch 0 flow 数不足")
    require(
        len({starts[record["task_id"]] for record in dispatch0}) == 1,
        "同一 Dispatch phase 内 flow 没有并行启动",
    )
    return (
        "HTSim 完成 two-stream DAG；GPU0 compute/comm 各自串行，phase 内 flow 并行，"
        "D0||A1、D1||E0、C0||E1、C1||R0 均实际重叠"
    )


def htsim_cross_layer_wavefront_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "two_layer_wavefront"
    model = ModelSpec(
        name="two_layer_wavefront",
        hidden=128,
        ffn_hidden=256,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=128,
        num_experts=16,
        topk=2,
        sequence_length=16,
        num_layers=2,
        micro_batches=2,
    )
    result = build_transformer_workload(
        TransformerWorkloadConfig(
            model=model,
            placement=Placement(16, 8, tuple(range(16))),
            tokens_per_rank=16,
            algorithm="deepep",
            chunk_tokens=16,
        ),
        cost_model=FixedOperationCostModel(),
    )
    emitted = emit_workload(result.graph, generated, metadata=result.metadata)
    manifest = json.loads(emitted.manifest_path.read_text(encoding="utf-8"))
    task_map = json.loads(emitted.task_map_path.read_text(encoding="utf-8"))
    log = execute_htsim(
        run_dir,
        "htsim_cross_layer_wavefront",
        emitted.matrix_path,
        emitted.dag_path,
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "跨层 wavefront DAG 未完整结束",
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
    records = task_map["tasks"]
    require(len(starts) == len(dones) == len(records), "跨层 task 时间戳不完整")

    rank0_compute = sorted(
        (record for record in records if record["rank"] == 0),
        key=lambda record: record["metadata"]["stream_sequence"],
    )
    for previous, current in zip(rank0_compute, rank0_compute[1:]):
        require(
            dones[previous["task_id"]] <= starts[current["task_id"]],
            f"GPU0 compute stream overlap: {previous['key']} / {current['key']}",
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
        return match is not None and 0 in (int(match.group(1)), int(match.group(2)))

    def phase_interval(phase_id: str) -> tuple[float, float]:
        phase = [
            record
            for record in records
            if record["metadata"].get("stream_phase_id") == phase_id
            and touches_rank0(record)
        ]
        require(bool(phase), f"GPU0 缺少 phase {phase_id}")
        return (
            min(starts[record["task_id"]] for record in phase),
            max(dones[record["task_id"]] for record in phase),
        )

    by_key = {record["key"]: record for record in records}
    attention = by_key["mb0.layer1.attention.rank0"]
    attention_interval = (
        starts[attention["task_id"]],
        dones[attention["task_id"]],
    )
    layer0_mb1_combine = phase_interval("mb1.layer0.combine")
    overlap_us = max(
        0.0,
        min(attention_interval[1], layer0_mb1_combine[1])
        - max(attention_interval[0], layer0_mb1_combine[0]),
    )
    require(
        overlap_us > 0,
        "没有实际出现 L1 MB0 Attention || L0 MB1 Combine",
    )
    layer1_mb0_dispatch = phase_interval("mb0.layer1.dispatch")
    require(
        layer0_mb1_combine[1] <= layer1_mb0_dispatch[0],
        "单 communication stream 错误重叠了跨层通信 phase",
    )
    return (
        "HTSim 实际观察到 L1 MB0 Attention || L0 MB1 Combine；"
        f"overlap={overlap_us:.6g}us，compute/comm stream 各自串行"
    )


def htsim_deepep_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "deepep_destination_forward"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    task_count = manifest["task_count"]
    barrier_count = manifest["barrier_count"]
    log = execute_htsim(
        run_dir,
        "htsim_deepep_destination_forward",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={task_count} barriers={barrier_count}" in log,
        "DeepEP 生成 DAG 未完整结束",
    )
    network_tasks = sum(
        metadata["hierarchical_transfer"]["task_count_by_leg"].values()
    )
    require(
        len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == network_tasks,
        "DeepEP 分层 network task 没有全部完成",
    )
    require("SERVER_FORWARD_BEGIN" not in log,
            "DeepEP 显式分层 task 错误进入 server_forward 状态机")
    require("scope=same_rail" in log, "DeepEP RDMA leg 没有进入 fabric")
    require("scope=same_server" in log, "DeepEP NVLink leg 没有进入本地 FullMesh")
    return (
        f"HTSim 完成 DeepEP 分层全图：{task_count} tasks/"
        f"{barrier_count} barriers，{network_tasks} 个显式 RDMA/NVLink task"
    )


def htsim_nccl_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "nccl_rank_direct"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    log = execute_htsim(
        run_dir,
        "htsim_nccl_rank_direct",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "NCCL 生成 DAG 未完整结束",
    )
    require("SERVER_FORWARD_BEGIN" not in log, "NCCL 运行时进入 server_forward")
    require(
        len(re.findall(r"MPRAIL_FLOW flow=\d+ src=0 dst=9 scope=cross_rail", log))
        == 6,
        "NCCL dispatch 没有按真实 0->9 跨 rail 直达",
    )
    require(len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == 12,
            "NCCL 12 个 network tasks 没有全部完成")
    return "HTSim 完成 NCCL direct DAG；0->9 经过 cross-rail，未使用 server_forward"


def htsim_moonep_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "moonep_deepep_scaleout"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    log = execute_htsim(
        run_dir,
        "htsim_moonep_deepep_scaleout",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "MoonEP 生成 DAG 未完整结束",
    )
    network_tasks = sum(
        task["kind"] == "transfer"
        for task in json.loads(
            (generated / "task_map.json").read_text(encoding="utf-8")
        )["tasks"]
    )
    require("SERVER_FORWARD_BEGIN" not in log,
            "MoonEP 显式分层 task 错误进入 server_forward")
    require(len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == network_tasks,
            "MoonEP prefetch 与分层 token tasks 没有全部完成")
    require("scope=same_server" in log, "MoonEP 本地 expert prefetch 未走 FullMesh")
    return f"HTSim 完成 MoonEP EP32 DAG；14 个本地 prefetch 和 {network_tasks - 14} 个分层 token tasks 全部完成"


def htsim_eplb_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "eplb_deepep_steady_state"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    log = execute_htsim(
        run_dir,
        "htsim_eplb_deepep_steady_state",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={manifest['task_count']} "
        f"barriers={manifest['barrier_count']}" in log,
        "EPLB 生成 DAG 未完整结束",
    )
    network_tasks = sum(
        task["kind"] == "transfer"
        for task in json.loads(
            (generated / "task_map.json").read_text(encoding="utf-8")
        )["tasks"]
    )
    require(len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == network_tasks,
            "EPLB network tasks 没有全部完成")
    require("SERVER_FORWARD_BEGIN" not in log,
            "EPLB 显式分层 task 错误进入 server_forward")
    require("expert_weight_prefetch" not in log,
            "稳态 EPLB 运行时出现 weight prefetch")
    return (
        f"HTSim 完成 EPLB 稳态 DAG：{network_tasks} 个分层 network tasks"
    )


def write_report(run_dir: Path, results: list[CaseResult]) -> None:
    passed = sum(result.status == "passed" for result in results)
    lines = [
        "# MoE DAG 生成器测试报告",
        "",
        f"- 通过：{passed}/{len(results)}",
        f"- 失败：{len(results) - passed}/{len(results)}",
        "",
        "| 测试 | 状态 | 验证内容 |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {result.name} | {result.status} | {result.detail} |"
        for result in results
    )
    lines.extend(
        [
            "",
            "## 范围",
            "",
            "已验证字节计算、显式 chunk task、flow 完成依赖、barrier lowering、",
            "NCCL direct/no-dedup、DeepEP 两级去冗余/分层传输、EPLB hierarchical placement/",
            "steady-state transport、MoonEP per-server replica/DeepEP scale-out 组合以及 HTSim 加载执行。",
            "逻辑双 stream 已降低为 DAG edges；不测试运行时 CUDA stream scheduler、",
            "单 flow 包进度事件、动态 SM、HBM 竞争或 kernel profiling。",
        ]
    )
    (run_dir / "测试报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"passed": passed, "total": len(results), "cases": [asdict(item) for item in results]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_workload_generator"
    run_dir.mkdir(parents=True)
    suite = Suite(run_dir)
    suite.run("h100_cost_model", cost_model_case)
    suite.run("json_compute_cost_model", lambda: json_compute_cost_case(run_dir))
    suite.run("barrier_emitter", lambda: emitter_case(run_dir))
    suite.run("token_payload_policy", token_payload_policy_case)
    suite.run("eplb_official_example", eplb_official_example_case)
    suite.run("eplb_steady_state", lambda: eplb_case(run_dir))
    suite.run("deepep_destination_forward", lambda: deepep_destination_forward_case(run_dir))
    suite.run("deepep_cli", lambda: deepep_cli_case(run_dir))
    suite.run("algorithm_cli", lambda: algorithm_cli_case(run_dir))
    suite.run("nccl_rank_direct", lambda: nccl_case(run_dir))
    suite.run("moonep_balanced_replica", lambda: moonep_case(run_dir))
    suite.run("moonep_deepep_scaleout", lambda: moonep_scaleout_case(run_dir))
    suite.run("two_microbatch_overlap", lambda: model_pipeline_case(run_dir))
    suite.run("relay_stream_participant", relay_stream_participant_case)
    suite.run("multi_layer_group_schedule", multi_layer_group_schedule_case)
    try:
        build_simulator(run_dir)
        suite.run("htsim_generated_dag", lambda: htsim_case(run_dir))
        suite.run(
            "htsim_two_microbatch_two_stream",
            lambda: htsim_two_stream_model_case(run_dir),
        )
        suite.run(
            "htsim_cross_layer_wavefront",
            lambda: htsim_cross_layer_wavefront_case(run_dir),
        )
        suite.run("htsim_deepep_destination_forward", lambda: htsim_deepep_case(run_dir))
        suite.run("htsim_nccl_rank_direct", lambda: htsim_nccl_case(run_dir))
        suite.run("htsim_eplb_deepep_steady_state", lambda: htsim_eplb_case(run_dir))
        suite.run("htsim_moonep_deepep_scaleout", lambda: htsim_moonep_case(run_dir))
    except Exception as exc:
        suite.results.append(CaseResult("htsim_generated_dag", "failed", str(exc)))
    write_report(run_dir, suite.results)
    passed = sum(result.status == "passed" for result in suite.results)
    print(f"workload generator tests: {passed}/{len(suite.results)} passed")
    print(f"log directory: {run_dir}")
    for result in suite.results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    return 0 if passed == len(suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
