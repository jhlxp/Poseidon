#!/usr/bin/env python3
"""Test DAG compute/network timeline parsing, overlap accounting, and outputs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visualization.dag_timeline import (  # noqa: E402
    TaskDefinition,
    TaskEvent,
    read_execution_log,
    read_task_map,
    summarize_ranks,
)


PLOTTER = ROOT / "visualization" / "dag_timeline.py"
COMPARISON = ROOT / "visualization" / "dsv3_algorithm_comparison.py"


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def task_record(
    task_id: int,
    key: str,
    kind: str,
    *,
    rank: int | None = None,
    src_rank: int | None = None,
    dst_rank: int | None = None,
    duration_us: float = 0.0,
    transfer_bytes: int = 0,
    operation: str | None = None,
    payload_kind: str | None = None,
    predecessors: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "available_sms": 112 if kind == "compute" else None,
        "barrier_group": None,
        "barrier_id": task_id - 1,
        "chunk_id": 0 if kind == "transfer" else None,
        "communication_phase_id": "synthetic:dispatch" if kind == "transfer" else None,
        "dst_rank": dst_rank,
        "duration_us": duration_us,
        "key": key,
        "kind": kind,
        "metadata": ({"operation": operation} if operation else {}),
        "operation_flops": 1 if kind == "compute" else 0,
        "overlaps_communication": kind == "compute" and rank == 0,
        "payload_kind": payload_kind,
        "peak_flops_per_second": 1.0 if kind == "compute" else None,
        "predecessor_barrier_ids": list(range(len(predecessors))),
        "predecessors": list(predecessors),
        "rank": rank,
        "route_spec": None,
        "src_rank": src_rank,
        "task_id": task_id,
        "transfer_bytes": transfer_bytes,
    }


def build_inputs(input_dir: Path) -> tuple[Path, Path, Path]:
    workload_dir = input_dir / "workload"
    workload_dir.mkdir(parents=True)
    manifest = {
        "format_version": 1,
        "graph": "synthetic_overlap",
        "num_ranks": 2,
        "task_count": 4,
        "barrier_count": 4,
        "compute_task_count": 3,
        "transfer_task_count": 1,
        "transfer_bytes_by_payload": {"dispatch_hidden": 1_000_000},
        "metadata": {},
    }
    tasks = [
        task_record(
            1,
            "mb0.attention.rank0",
            "compute",
            rank=0,
            duration_us=10.0,
            operation="attention",
        ),
        task_record(
            2,
            "mb0.dispatch.src0.dst1",
            "transfer",
            src_rank=0,
            dst_rank=1,
            transfer_bytes=1_000_000,
            payload_kind="dispatch_hidden",
        ),
        task_record(
            3,
            "mb0.router.rank1",
            "compute",
            rank=1,
            duration_us=4.0,
            operation="router_projection",
        ),
        task_record(
            4,
            "mb0.expert.rank0",
            "compute",
            rank=0,
            duration_us=5.0,
            operation="expert_ffn",
            predecessors=("mb0.attention.rank0",),
        ),
    ]
    manifest_path = workload_dir / "manifest.json"
    task_map_path = workload_dir / "task_map.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    task_map_path.write_text(
        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = input_dir / "htsim.log"
    log_path.write_text(
        "\n".join(
            [
                "DAG_TASK_START task=1 barrier=0 src_rank=0 dst_rank=0 transfer_bytes=0 compute_us=10 time_us=0",
                "DAG_TASK_START task=3 barrier=2 src_rank=1 dst_rank=1 transfer_bytes=0 compute_us=4 time_us=0",
                "DAG_TASK_START task=2 barrier=1 src_rank=0 dst_rank=1 transfer_bytes=1000000 compute_us=0 time_us=2",
                "DAG_TASK_DONE task=3 barrier=2 time_us=4",
                "DAG_TASK_DONE task=2 barrier=1 time_us=8",
                "DAG_TASK_DONE task=1 barrier=0 time_us=10",
                "DAG_TASK_START task=4 barrier=3 src_rank=0 dst_rank=0 transfer_bytes=0 compute_us=5 time_us=10",
                "DAG_TASK_DONE task=4 barrier=3 time_us=15",
                "DAG_SUMMARY tasks=4 barriers=4 makespan_us=15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return workload_dir, task_map_path, log_path


def parser_overlap_case(task_map_path: Path, log_path: Path) -> str:
    tasks = read_task_map(task_map_path, 2)
    events = read_execution_log(log_path, tasks)
    summaries = summarize_ranks(events, 2)
    require(len(events) == 4, "没有解析出全部 4 个 task")
    require(events[1].actual_duration_us == 6.0, "network task FCT 应为 6us")
    require(
        math.isclose(events[1].logical_throughput_gbps or 0.0, 4000 / 3),
        "logical throughput 计算错误",
    )
    require(summaries[0].compute_active_us == 15.0, "GPU0 compute union 应为 15us")
    require(summaries[0].network_active_us == 6.0, "GPU0 network union 应为 6us")
    require(summaries[0].compute_network_overlap_us == 6.0,
            "GPU0 compute/network overlap 应为 6us")
    require(summaries[1].compute_network_overlap_us == 2.0,
            "GPU1 compute/network overlap 应为 2us")
    require(summaries[0].tx_bytes == 1_000_000, "GPU0 TX bytes 错误")
    require(summaries[1].rx_bytes == 1_000_000, "GPU1 RX bytes 错误")
    return "4 task join 完整；network FCT=6us；GPU0/GPU1 overlap=6us/2us"


def invalid_log_case(task_map_path: Path, log_path: Path, input_dir: Path) -> str:
    incomplete = input_dir / "htsim_incomplete.log"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    incomplete.write_text(
        "\n".join(line for line in lines if not line.startswith("DAG_TASK_DONE task=2 "))
        + "\n",
        encoding="utf-8",
    )
    tasks = read_task_map(task_map_path, 2)
    try:
        read_execution_log(incomplete, tasks)
    except ValueError as exc:
        require("missing dones=[2]" in str(exc), "缺失完成事件错误不明确")
        return "缺失 task 2 done event 被确定性拒绝"
    raise AssertionError("不完整 HTSim log 没有报错")


def relay_accounting_case() -> str:
    task = TaskDefinition(
        task_id=1,
        barrier_id=0,
        key="relay.transfer",
        kind="transfer",
        rank=None,
        src_rank=0,
        dst_rank=3,
        transfer_bytes=4096,
        declared_duration_us=0.0,
        payload_kind="dispatch_hidden",
        operation=None,
        communication_phase_id="relay:dispatch",
        route_spec="server_forward src_relay:0 dst_relay:2",
        predecessors=(),
        predecessor_barrier_ids=(),
        overlaps_communication=False,
        metadata={},
    )
    summaries = summarize_ranks((TaskEvent(task, 1.0, 5.0),), 4)
    require(summaries[0].tx_bytes == 4096, "source TX bytes 错误")
    require(
        summaries[2].rx_bytes == 4096 and summaries[2].tx_bytes == 4096,
        "dst relay 应同时计入 fabric RX 和 local TX",
    )
    require(summaries[3].rx_bytes == 4096, "final destination RX bytes 错误")
    return "server_forward dst relay 同时进入 RX/TX lane 和 network active interval"


def cli_case(
    run_dir: Path, workload_dir: Path, log_path: Path, output_dir: Path
) -> str:
    command = [
        sys.executable,
        str(PLOTTER),
        "--workload-dir", str(workload_dir),
        "--htsim-log", str(log_path),
        "--output-dir", str(output_dir),
        "--title", "DAG timeline functional test",
        "--gpus-per-server", "2",
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
    (run_dir / "命令.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (run_dir / "绘图.log").write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"可视化 CLI 返回码 {completed.returncode}")
    require("parsed 4 tasks across 2 ranks; makespan 15 us" in completed.stdout,
            "CLI 汇总不正确")
    return "CLI 成功生成 2 GPU Compute/TX/RX timeline"


def artifacts_case(output_dir: Path) -> str:
    html = (output_dir / "dag_gpu_timeline.html").read_text(encoding="utf-8")
    require("GPU 00" in html and "Network TX" in html and "Network RX" in html,
            "HTML 缺少 GPU 三 lane")
    require("Task 2: mb0.dispatch.src0.dst1" in html,
            "HTML tooltip 缺少 network task")
    require("Bytes: 1000000" in html and "Actual duration: 6 us" in html,
            "HTML tooltip 缺少字节或 FCT")
    require(
        'class="zoom-slider"' in html
        and 'class="scale-readout"' in html
        and "ResizeObserver" in html,
        "HTML 缺少 Fit/水平缩放和动态时间比例",
    )
    require(
        '<details class="details"><summary>Task details (4)</summary>' in html,
        "Task details 没有使用三角折叠控件",
    )

    with (output_dir / "dag_rank_overlap_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summaries = {int(row["rank"]): row for row in csv.DictReader(handle)}
    require(summaries[0]["compute_network_overlap_us"] == "6.0",
            "rank summary overlap 错误")
    require(summaries[1]["rx_bytes"] == "1000000", "rank summary RX bytes 错误")

    overview = json.loads(
        (output_dir / "dag_timeline_summary.json").read_text(encoding="utf-8")
    )
    require(overview["makespan_us"] == 15.0, "overview makespan 错误")
    require(overview["logical_transfer_bytes"] == 1_000_000,
            "overview logical bytes 错误")
    require(overview["selected_rank_compute_network_overlap_sum_us"] == 8.0,
            "overview aggregate overlap 错误")

    with (output_dir / "dag_task_timeline.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        tasks = {int(row["task_id"]): row for row in csv.DictReader(handle)}
    require(tasks[2]["actual_duration_us"] == "6.0", "task CSV FCT 错误")
    require(math.isclose(float(tasks[2]["logical_throughput_gbps"]), 4000 / 3),
            "task CSV logical throughput 错误")
    require(not (output_dir / "dag_gpu_timeline.png").exists(),
            "DAG timeline 不应再生成 PNG")
    return "HTML hover、task CSV、rank overlap 和总览正确，未生成 PNG"


def comparison_case(run_dir: Path, timeline_dir: Path) -> str:
    case_dirs: list[tuple[str, Path]] = []
    for algorithm in ("nccl", "deepep"):
        case_dir = run_dir / "comparison_inputs" / algorithm
        case_timeline = case_dir / "timeline"
        case_link = case_dir / "link_load"
        case_timeline.mkdir(parents=True)
        case_link.mkdir()
        for name in ("dag_gpu_timeline.html", "dag_timeline_summary.json"):
            shutil.copyfile(timeline_dir / name, case_timeline / name)
        (case_link / "mprail_link_load_by_layer.png").write_bytes(b"PNG")
        (case_link / "mprail_link_load_summary.csv").write_text(
            "panel,total_bytes\nall,1\n", encoding="utf-8"
        )
        (case_link / "mprail_endpoint_load_summary.csv").write_text(
            "rank,tx_bytes\n0,1\n", encoding="utf-8"
        )
        case_dirs.append((algorithm, case_dir))

    output = run_dir / "comparison.html"
    command = [sys.executable, str(COMPARISON), "--output", str(output)]
    for algorithm, case_dir in case_dirs:
        command.extend(("--case", f"{algorithm}={case_dir}"))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    require(completed.returncode == 0, "四算法总览 HTML 生成失败")
    html = output.read_text(encoding="utf-8")
    require(html.count('<details class="algorithm"') == 2,
            "算法没有分别使用可折叠分区")
    require(html.count("mprail_link_load_by_layer.png") == 2,
            "算法分区没有引用各自链路负载图")
    require("Expand all" in html and "Collapse all" in html,
            "总览缺少全局展开/收起按钮")
    return "算法分区、iframe timeline、链路负载图和全局折叠按钮正确"


class Suite:
    def __init__(self) -> None:
        self.results: list[CaseResult] = []

    def run(self, name: str, function: Callable[[], str]) -> None:
        try:
            self.results.append(CaseResult(name, "passed", function()))
        except Exception as exc:
            self.results.append(CaseResult(name, "failed", str(exc)))


def write_report(run_dir: Path, results: list[CaseResult]) -> None:
    passed = sum(item.status == "passed" for item in results)
    lines = [
        "# DAG 执行时间线可视化测试报告",
        "",
        "本测试构造两个 GPU、三个 compute task 和一个 network task。network 在",
        "GPU0 compute 的 2-8us 区间内执行，同时与 GPU1 compute 的 2-4us 区间重叠。",
        "测试验证日志 join、实际 FCT、logical throughput、每卡 overlap、HTML",
        "hover、逐 task CSV、逐 rank CSV 和总览 JSON。",
        "",
        f"- 通过：{passed}/{len(results)}",
        f"- 失败：{len(results) - passed}/{len(results)}",
        "",
        "| case | 状态 | 结果 |",
        "|---|---|---|",
    ]
    lines.extend(f"| {item.name} | {item.status} | {item.detail} |" for item in results)
    (run_dir / "测试报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "total": len(results),
                "cases": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_dag_timeline_visualization"
    input_dir = run_dir / "inputs"
    output_dir = run_dir / "visualization"
    output_dir.mkdir(parents=True)
    workload_dir, task_map_path, log_path = build_inputs(input_dir)

    suite = Suite()
    suite.run(
        "parse_and_overlap",
        lambda: parser_overlap_case(task_map_path, log_path),
    )
    suite.run(
        "reject_incomplete_log",
        lambda: invalid_log_case(task_map_path, log_path, input_dir),
    )
    suite.run("relay_lane_accounting", relay_accounting_case)
    suite.run(
        "timeline_cli",
        lambda: cli_case(run_dir, workload_dir, log_path, output_dir),
    )
    suite.run("validate_artifacts", lambda: artifacts_case(output_dir))
    suite.run("comparison_dashboard", lambda: comparison_case(run_dir, output_dir))
    write_report(run_dir, suite.results)

    passed = sum(item.status == "passed" for item in suite.results)
    print(f"DAG timeline visualization tests: {passed}/{len(suite.results)} passed")
    print(f"log directory: {run_dir}")
    for item in suite.results:
        print(f"[{item.status}] {item.name}: {item.detail}")
    return 0 if passed == len(suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
