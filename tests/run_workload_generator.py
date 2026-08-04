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
    MoonEPBuilder,
    MoonEPConfig,
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


def deepep_hybrid_case(run_dir: Path) -> str:
    placement = Placement(8, 4, (4, 5))
    assignments = tuple(
        RoutingAssignment(0, token, slot, slot)
        for token in range(3)
        for slot in range(2)
    )
    invocation = MoEInvocation(
        "deepep_exact",
        placement,
        (3, 0, 0, 0, 0, 0, 0, 0),
        128,
        256,
        2,
        "fp8",
        "bf16",
        "bf16",
        assignments,
    )
    graph = TaskGraph("deepep_hybrid_exact", 8)
    result = DeepEPBuilder(
        H100CostModel(), DeepEPConfig(mode="hybrid", chunk_tokens=1)
    ).build(graph, invocation)
    graph.validate()

    tasks = graph.tasks
    fabric_dispatch = [
        task for task in tasks if task.payload_kind == "dispatch_fabric"
    ]
    local_fanout = [
        task for task in tasks if task.payload_kind == "dispatch_local_fanout"
    ]
    fabric_combine = [
        task for task in tasks if task.payload_kind == "combine_fabric_partial"
    ]
    require(len(fabric_dispatch) == 3, "3 tokens 应生成 3 个 fabric dispatch chunks")
    require(len(local_fanout) == 3, "relay->rank5 应生成 3 个 local chunks")
    require(len(fabric_combine) == 3, "combine 每个 token/server 应只返回一个 partial")
    require(
        sum(task.transfer_bytes for task in fabric_dispatch) == 3 * 128,
        "同一 token 在目标 server 选择两个 expert 时 fabric dispatch 被重复",
    )
    require(
        sum(task.transfer_bytes for task in fabric_combine) == 3 * 128 * 2,
        "combine fabric partial 字节错误",
    )
    for fanout in local_fanout:
        require(len(fanout.predecessors) == 1, "每个 local chunk 必须只等对应 fabric chunk")
        predecessor = graph.task(next(iter(fanout.predecessors)))
        require(predecessor.chunk_id == fanout.chunk_id, "chunk pipeline 对应关系错误")

    emitted = emit_workload(
        graph,
        run_dir / "generated" / "deepep_hybrid_exact",
        metadata=result.metadata,
    )
    require(emitted.dag_path.exists(), "DeepEP DAG 未生成")
    return "跨节点 dispatch/combine 按 destination server 去重；3 个 chunk 独立推进"


def deepep_direct_cli_case(run_dir: Path) -> str:
    output_dir = run_dir / "generated" / "deepep_direct_cli"
    command = [
        sys.executable,
        str(PYSRC / "generate_moe_dag.py"),
        "--output", str(output_dir),
        "--algorithm", "deepep-direct",
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
    require(manifest["metadata"]["algorithm"] == "deepep-direct", "CLI 算法记录错误")
    payloads = {
        task["payload_kind"]
        for task in task_map["tasks"]
        if task["kind"] == "transfer"
    }
    require(payloads == {"dispatch_hidden", "combine_partial"},
            f"direct 模式出现非 direct payload: {sorted(payloads)}")
    return "CLI 生成 direct workload；传输仅含 dispatch_hidden/combine_partial"


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
    placement = Placement(8, 4, tuple(range(8)))
    result = build_transformer_workload(
        TransformerWorkloadConfig(
            model=model,
            placement=placement,
            tokens_per_rank=4,
            algorithm="deepep-hybrid",
            chunk_tokens=2,
        )
    )
    attention_1 = result.graph.task("mb1.attention.rank0")
    require(
        attention_1.predecessors == {"mb0.attention.rank0"},
        "Attention 2 不应依赖 Dispatch/Expert 1",
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
    emitted = emit_workload(
        result.graph,
        run_dir / "generated" / "two_microbatch_block",
        metadata=result.metadata,
    )
    manifest = json.loads(emitted.manifest_path.read_text(encoding="utf-8"))
    require(manifest["metadata"]["scope"]["dynamic_gpu_resource_scheduling"] is False,
            "manifest 没有声明 compute resource 模型边界")
    return "Attention 2 与 Dispatch 1 可并行；未引入跨 microbatch stage barrier"


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
        "-mprail_planes", "8",
        "-mprail_gpus_per_server", "4",
        "-mprail_servers_per_rail", "1",
        "-mprail_l1_eps_per_plane", "2",
        "-mprail_l0_l1_links_per_spine", "2",
        "-linkspeed", "100000",
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


def htsim_deepep_case(run_dir: Path) -> str:
    generated = run_dir / "generated" / "deepep_hybrid_exact"
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    task_count = manifest["task_count"]
    barrier_count = manifest["barrier_count"]
    log = execute_htsim(
        run_dir,
        "htsim_deepep_hybrid",
        generated / "nodes.cm",
        generated / "workload.dag",
    )
    require(
        f"DAG_SUMMARY tasks={task_count} barriers={barrier_count}" in log,
        "DeepEP 生成 DAG 未完整结束",
    )
    require(
        len(re.findall(r"^DAG_NETWORK_DONE", log, re.MULTILINE)) == 12,
        "DeepEP 12 个 network chunk 没有全部完成",
    )
    return f"HTSim 完成 DeepEP hybrid 全图：{task_count} tasks/{barrier_count} barriers"


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
            "DeepEP hybrid/direct、MoonEP 首版 planner 以及 HTSim 加载执行。",
            "不测试单 flow 包进度事件、CUDA stream、动态 SM、HBM 竞争或 kernel profiling。",
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
    suite.run("deepep_hybrid_exact_bytes", lambda: deepep_hybrid_case(run_dir))
    suite.run("deepep_direct_cli", lambda: deepep_direct_cli_case(run_dir))
    suite.run("moonep_balanced_replica", lambda: moonep_case(run_dir))
    suite.run("two_microbatch_overlap", lambda: model_pipeline_case(run_dir))
    try:
        build_simulator(run_dir)
        suite.run("htsim_generated_dag", lambda: htsim_case(run_dir))
        suite.run("htsim_deepep_hybrid", lambda: htsim_deepep_case(run_dir))
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
