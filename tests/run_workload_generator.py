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
    ModelSpec,
    MoEInvocation,
    Placement,
    RoutingAssignment,
    TaskGraph,
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
    require(result.metadata["server_forward_task_count"] > 0,
            "EPLB 跨服务器 token flow 没有复用 DeepEP 目标端转发")

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
        result.metadata["unique_token_payload_count"]
        == deepep_result.metadata["unique_token_payload_count"],
        "EPLB 与 DeepEP 的 destination-rank 去重口径不一致",
    )

    emit_workload(
        graph,
        run_dir / "generated" / "eplb_deepep_steady_state",
        metadata=result.metadata,
    )
    return "热点 rank routes 从 16 降至 3；稳态无迁移，并复用 DeepEP 去重/目标端转发"


def deepep_destination_forward_case(run_dir: Path) -> str:
    placement = Placement(32, 8, (9, 9))
    assignments = tuple(
        RoutingAssignment(0, token, slot, slot)
        for token in range(3)
        for slot in range(2)
    )
    invocation = MoEInvocation(
        "deepep_exact",
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
    require(len(dispatch) == 3, "3 tokens 应生成 3 个 dispatch chunks")
    require(len(combine) == 3, "3 tokens 应生成 3 个 combine chunks")
    require(
        sum(task.transfer_bytes for task in dispatch) == 3 * 128,
        "同一 token 在目标 rank 选择两个 expert 时 dispatch payload 被重复",
    )
    require(
        sum(task.transfer_bytes for task in combine) == 3 * 128 * 2,
        "combine partial 字节错误",
    )
    require(
        all(
            task.src_rank == 0
            and task.dst_rank == 9
            and task.route_spec == "server_forward src_relay:0 dst_relay:8"
            for task in dispatch
        ),
        "dispatch 没有使用真实端点和目标侧同 index relay",
    )
    require(
        all(
            task.src_rank == 9
            and task.dst_rank == 0
            and task.route_spec == "server_forward src_relay:9 dst_relay:1"
            for task in combine
        ),
        "combine 没有使用反向目标侧转发",
    )
    expert = graph.task("deepep_exact.expert.rank9")
    require(expert.metadata["real_token_routes"] == 6, "去重错误减少了 expert routes")
    require(result.metadata["route_count"] == 6, "route_count 汇总错误")
    require(result.metadata["unique_token_payload_count"] == 3, "payload 去重汇总错误")
    require(result.metadata["deduplicated_route_count"] == 3, "去重 route 汇总错误")
    require(result.metadata["server_forward_task_count"] == 6, "转发 task 汇总错误")

    emitted = emit_workload(
        graph,
        run_dir / "generated" / "deepep_destination_forward",
        metadata=result.metadata,
    )
    require(emitted.dag_path.exists(), "DeepEP DAG 未生成")
    dag = emitted.dag_path.read_text(encoding="utf-8")
    require(dag.count("server_forward src_relay:0 dst_relay:8") == 3,
            "DAG dispatch route 数量错误")
    require(dag.count("server_forward src_relay:9 dst_relay:1") == 3,
            "DAG combine route 数量错误")
    return "destination-rank 去重保留 6 expert routes；6 个跨服 task 均使用目标端转发"


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
    require(
        any(
            task["route_spec"] and task["route_spec"].startswith("server_forward ")
            for task in task_map["tasks"]
            if task["kind"] == "transfer"
        ),
        "CLI 生成物没有目标端转发 route",
    )
    return "CLI 生成 DeepEP 核心 workload，并输出 destination-side server_forward"


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
    require(len(token_flows) == 32, "MoonEP token dispatch/combine flow 数量错误")
    require(all(task.route_spec and task.route_spec.startswith("server_forward ")
                for task in token_flows),
            "MoonEP 跨服务器 token flow 没有复用 DeepEP transport")
    require(result.metadata["server_forward_task_count"] == 32,
            "MoonEP server_forward task 汇总错误")
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
    require(emitted.dag_path.read_text(encoding="utf-8").count("server_forward") == 32,
            "MoonEP DAG 的 DeepEP route 数量错误")
    return "server1 每 rank 2 routes、server2 每 rank 1 route；14 个本地 replica flows、32 个 DeepEP token flows"


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
        if task.key.startswith("mb1.moe.dispatch.src0.dst1")
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
    require(
        all(not (task.predecessors & dispatch_keys) for task in dispatch_phase),
        "同一 Dispatch phase 内的 flow 被错误串行化",
    )
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
    return "每 GPU 一条 compute/comm stream；D0||A1、D1||E0、C0||E1，phase 内 flow 并行"


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
        and task.route_spec
        and task.metadata.get("dst_relay") != task.dst_rank
    )
    relay = dispatch_0.metadata["dst_relay"]
    require(isinstance(relay, int), "server_forward 缺少 dst relay")
    expert = result.graph.task(f"mb0.moe.expert.rank{relay}")
    require(
        dispatch_0.key in expert.predecessors,
        "relay GPU 的 Expert 没有等待其 Dispatch communication phase",
    )
    dispatch_1 = next(
        task
        for task in result.graph.tasks
        if task.key.startswith("mb1.moe.dispatch")
        and (
            task.src_rank == relay
            or task.dst_rank == relay
            or task.metadata.get("dst_relay") == relay
        )
    )
    require(
        dispatch_0.key in dispatch_1.predecessors,
        "relay GPU 的 comm stream 没有串行两个 Dispatch phase",
    )
    return f"dst relay GPU{relay} 被计入 compute/comm event 和 phase stream tail"


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
        "-mprail_l1_eps_per_plane", "8",
        "-mprail_l0_l1_links_per_spine", "1",
        "-linkspeed", "400000",
        "-local_linkspeed", "3200000",
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


def htsim_deepep_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "deepep_destination_forward"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
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
    require(
        len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == 6,
        "DeepEP 6 个逻辑 network task 没有全部完成",
    )
    begins = re.findall(r"^SERVER_FORWARD_BEGIN .* phases=(\d+)$", log, re.MULTILINE)
    require(begins == ["2"] * 6, f"DeepEP task 不是两个 subflow: {begins}")
    require("phase=src_local" not in log, "DeepEP 错误启动了 source-local phase")
    require(
        len(re.findall(r"^SERVER_FORWARD_PHASE_DONE .* phase=fabric ", log, re.MULTILINE))
        == 6,
        "DeepEP fabric subflow 完成数量错误",
    )
    require(
        len(re.findall(r"^SERVER_FORWARD_PHASE_DONE .* phase=dst_local ", log, re.MULTILINE))
        == 6,
        "DeepEP destination-local subflow 完成数量错误",
    )
    return (
        f"HTSim 完成 DeepEP 目标端转发全图：{task_count} tasks/"
        f"{barrier_count} barriers，6 个逻辑 task 各执行 2 个 subflow"
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
    require(len(re.findall(r"^SERVER_FORWARD_BEGIN", log, re.MULTILINE)) == 32,
            "MoonEP 32 个跨服务器 token tasks 未全部进入 DeepEP transport")
    require("phase=src_local" not in log,
            "MoonEP/DeepEP 目标端转发错误启动 src_local")
    require(len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == 46,
            "MoonEP 14 prefetch + 32 token tasks 没有全部完成")
    require("scope=same_server" in log, "MoonEP 本地 expert prefetch 未走 FullMesh")
    return "HTSim 完成 MoonEP EP32 DAG；14 个本地 prefetch 和 32 个 DeepEP token tasks 全部完成"


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
    require(
        len(re.findall(r"^SERVER_FORWARD_BEGIN", log, re.MULTILINE))
        == metadata["server_forward_task_count"],
        "EPLB DeepEP server_forward 数量与 manifest 不一致",
    )
    require("expert_weight_prefetch" not in log,
            "稳态 EPLB 运行时出现 weight prefetch")
    return (
        f"HTSim 完成 EPLB 稳态 DAG：{network_tasks} 个 network tasks，"
        f"{metadata['server_forward_task_count']} 个目标端转发"
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
            "NCCL direct/no-dedup、DeepEP 目标端转发、EPLB hierarchical placement/",
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
    try:
        build_simulator(run_dir)
        suite.run("htsim_generated_dag", lambda: htsim_case(run_dir))
        suite.run(
            "htsim_two_microbatch_two_stream",
            lambda: htsim_two_stream_model_case(run_dir),
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
