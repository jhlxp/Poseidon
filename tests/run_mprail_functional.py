#!/usr/bin/env python3
"""Build and run the standalone MpRail functional test suite."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "htsim" / "sim"
BUILD_DIR = SIM_DIR / "build-mprail"
BINARY = BUILD_DIR / "datacenter" / "htsim_uec"


@dataclass
class CaseResult:
    name: str
    status: str
    returncode: int
    detail: str
    command: list[str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def build_simulator(run_dir: Path) -> None:
    commands = [
        ["cmake", "-S", str(SIM_DIR), "-B", str(BUILD_DIR),
         "-DCMAKE_BUILD_TYPE=Release"],
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
            write_text(run_dir / "构建.log", "\n".join(chunks))
            raise RuntimeError(
                f"构建失败，返回码 {completed.returncode}，详见 构建.log"
            )
    write_text(run_dir / "构建.log", "\n".join(chunks))


class Suite:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.inputs_dir = run_dir / "inputs"
        self.cases_dir = run_dir / "cases"
        self.results: list[CaseResult] = []

    def run_case(
        self,
        name: str,
        matrix: str,
        validator: Callable[[str, int], str],
        dag: str | None = None,
        end_us: int = 1000,
        extra_args: list[str] | None = None,
    ) -> None:
        matrix_path = self.inputs_dir / f"{name}.cm"
        write_text(matrix_path, matrix)
        case_dir = self.cases_dir / name
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
            "-end", str(end_us),
            "-strat", "ecmp_host",
            "-tm", str(matrix_path),
            "-o", str(case_dir / "htsim.dat"),
        ]
        if extra_args:
            command.extend(extra_args)
        if dag is not None:
            dag_path = self.inputs_dir / f"{name}.dag"
            write_text(dag_path, dag)
            command.extend(["-dag", str(dag_path)])

        write_text(case_dir / "命令.txt", " ".join(command) + "\n")
        try:
            completed = subprocess.run(
                command,
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                check=False,
            )
            write_text(case_dir / "htsim.log", completed.stdout)
            detail = validator(completed.stdout, completed.returncode)
            self.results.append(CaseResult(
                name, "passed", completed.returncode, detail, command
            ))
        except Exception as exc:
            output = locals().get("completed")
            returncode = output.returncode if output is not None else -1
            if output is None:
                write_text(case_dir / "htsim.log", f"测试驱动异常: {exc}\n")
            self.results.append(CaseResult(
                name, "failed", returncode, str(exc), command
            ))


def assert_success(log: str, returncode: int) -> None:
    require(returncode == 0, f"仿真返回码应为 0，实际为 {returncode}")
    require("Done" in log, "仿真日志没有 Done")
    require(not re.search(r"\b(?:OCS|Oxc|oxc_)\b", log),
            "MpRail 日志出现旧光交换语义")


def validate_same_server(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("scope=same_server" in log, "没有识别同服务器路径")
    link_lines = [line for line in log.splitlines() if line.startswith("MPRAIL_LINK")]
    require(link_lines, "没有创建服务器内部 FullMesh 链路")
    require(all("MPRAIL_L0" not in line and "MPRAIL_L1" not in line
                for line in link_lines), "同服务器流量进入了 L0/L1")
    require(any("speed_gbps=7200" in line for line in link_lines),
            "服务器内部链路没有使用 7200Gbps H100 NVLink 速率")
    require("MPRAIL_LOCAL_INJECTION independent=yes speed_gbps=7200" in log,
            "服务器内部流量没有独立于 RDMA NIC")
    return f"本地链路 {len(link_lines)} 条，速率 7200Gbps，L0/L1 链路 0 条"


def validate_same_rail(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("scope=same_rail" in log, "没有识别同 rail 路径")
    require("MPRAIL_L1" not in log, "同 rail 流量错误进入 L1")
    require("MPRAIL_L0_r0_" in log, "同 rail 流量没有进入 rail 0 的 L0")
    l0_links = len(re.findall(r"^MPRAIL_LINK .*MPRAIL_L0", log, re.MULTILINE))
    return f"rail 0 内 L0 相关链路 {l0_links} 条，L1 相关链路 0 条"


def validate_cross_rail(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("scope=cross_rail" in log, "没有识别跨 rail 路径")
    links = re.findall(r"^MPRAIL_LINK src=(\S+) dst=(\S+)", log, re.MULTILINE)
    require(any("MPRAIL_L0" in src and "MPRAIL_L1" in dst
                for src, dst in links), "缺少 L0 -> L1 链路")
    require(any("MPRAIL_L1" in src and "MPRAIL_L0" in dst
                for src, dst in links), "缺少 L1 -> L0 链路")
    require(not any("MPRAIL_L0" in src and "MPRAIL_L0" in dst
                    for src, dst in links), "检测到 rail 间 L0 直连")
    for src, dst in links:
        if "MPRAIL_L0" in src and "MPRAIL_L1" in dst:
            src_plane = re.search(r"_p(\d+)", src)
            dst_plane = re.search(r"_p(\d+)", dst)
            require(src_plane and dst_plane
                    and src_plane.group(1) == dst_plane.group(1),
                    f"跨 plane 链路: {src} -> {dst}")
        if "MPRAIL_L1" in src and "MPRAIL_L0" in dst:
            src_plane = re.search(r"_p(\d+)", src)
            dst_plane = re.search(r"_p(\d+)", dst)
            require(src_plane and dst_plane
                    and src_plane.group(1) == dst_plane.group(1),
                    f"跨 plane 链路: {src} -> {dst}")
    topology = re.search(
        r"MPRAIL_TOPOLOGY nodes=32 servers=(\d+) rails=(\d+) "
        r"planes=(\d+) l0=(\d+) l1=(\d+)", log
    )
    require(topology is not None, "缺少拓扑计数日志")
    require(tuple(map(int, topology.groups())) == (4, 8, 1, 8, 4),
            f"拓扑计数错误: {topology.groups()}")
    fabric_links = sum(
        ("MPRAIL_L0" in src and "MPRAIL_L1" in dst)
        or ("MPRAIL_L1" in src and "MPRAIL_L0" in dst)
        for src, dst in links
    )
    return (
        "servers=4，rails=8，planes=1，L0=8，L1=4，"
        f"本次物化 L0/L1 有向链路 {fabric_links} 条"
    )


def validate_flow_routing_mode(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("routing_mode flow_ecmp" in log, "全局路由模式不是 flow_ecmp")
    require(re.search(r"routing=flow_ecmp plane=\d+", log) is not None,
            "flow ECMP 没有固定 preferred plane")
    require("ecmp_spines=4 ecmp_bundles=1" in log,
            "flow ECMP 下一跳维度错误")
    return "UEC ECMP：固定 plane 0，L0/L1 使用 4 spines x 1 bundle ECMP"


def validate_spray_routing_mode(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("routing_mode packet_spray_ecmp" in log,
            "全局路由模式不是 packet_spray_ecmp")
    require("routing=packet_spray_ecmp plane=-1" in log,
            "packet spray 仍固定 preferred plane")
    require("ecmp_spines=4 ecmp_bundles=1" in log,
            "packet spray ECMP 下一跳维度错误")
    return "UEC spray：单 plane 内逐包改变 pathid，使用 4 spines x 1 bundle ECMP"


def parse_event_times(log: str, prefix: str) -> dict[int, float]:
    return {
        int(item): float(timestamp)
        for item, timestamp in re.findall(
            rf"^{prefix} (?:task|barrier)=(\d+).*?time_us=([0-9.]+)",
            log,
            re.MULTILINE,
        )
    }


def validate_dag(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("DAG_SUMMARY tasks=6 barriers=2" in log, "DAG 没有完整结束")
    starts = parse_event_times(log, "DAG_TASK_START")
    dones = parse_event_times(log, "DAG_TASK_DONE")
    barrier_starts = parse_event_times(log, "DAG_BARRIER_START")
    barrier_dones = parse_event_times(log, "DAG_BARRIER_DONE")
    require(len(starts) == 6 and len(dones) == 6, "DAG task 日志数量不完整")
    require(len({starts[task] for task in (1, 2, 3, 4)}) == 1,
            "barrier 0 的 task 没有同时启动")
    require(barrier_dones[0] == max(dones[task] for task in (1, 2, 3, 4)),
            "barrier 0 没有等待最后一个 task")
    require(barrier_starts[1] >= barrier_dones[0],
            "barrier 1 在 barrier 0 完成前启动")
    require(all(starts[task] >= barrier_starts[1] for task in (5, 6)),
            "barrier 1 task 启动时刻错误")
    return (
        f"6 tasks/2 barriers；barrier0 {barrier_starts[0]:g}-{barrier_dones[0]:g}us，"
        f"barrier1 {barrier_starts[1]:g}-{barrier_dones[1]:g}us，makespan 40us"
    )


def validate_multiple_predecessors(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("DAG_SUMMARY tasks=3 barriers=3 makespan_us=25" in log,
            "多前驱 DAG 没有完整结束")
    barrier_starts = parse_event_times(log, "DAG_BARRIER_START")
    barrier_dones = parse_event_times(log, "DAG_BARRIER_DONE")
    require(barrier_starts[0] == 0 and barrier_starts[1] == 0,
            "两个 root barrier 没有同时启动")
    require(barrier_dones[0] == 10 and barrier_dones[1] == 20,
            "root barrier 完成时刻错误")
    require(barrier_starts[2] == 20 and barrier_dones[2] == 25,
            "barrier 2 没有等待 barrier 0 和 barrier 1")
    return "barrier2 前驱集合={0,1}，20us 启动，25us 完成"


def validate_independent_barriers(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("DAG_SUMMARY tasks=5 barriers=5 makespan_us=45" in log,
            "独立 barrier DAG 没有完整结束")
    starts = parse_event_times(log, "DAG_TASK_START")
    dones = parse_event_times(log, "DAG_TASK_DONE")
    require(starts[2] == starts[3] == dones[1],
            "Dispatch 1 与 Attention 2 没有在共同前驱后并发启动")
    require(starts[5] == dones[3] == 20,
            "Dispatch 2 被无关的 Dispatch 1 阻塞")
    require(starts[4] == dones[2] == 40,
            "Expert 1 没有只等待 Dispatch 1")
    return "两个分支独立推进：Dispatch2=20us 启动，Expert1=40us 启动"


def validate_local_fabric_independence(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    starts = parse_event_times(log, "DAG_TASK_START")
    dones = parse_event_times(log, "DAG_TASK_DONE")
    require(starts[1] == starts[2] == 0,
            "NVLink 与 RDMA flow 没有同时启动")
    local_duration = dones[1] - starts[1]
    fabric_duration = dones[2] - starts[2]
    transfer_bytes = 16 * 1024 * 1024
    local_gbps = transfer_bytes * 8 / local_duration / 1000
    require(local_gbps > 1000,
            f"本地有效吞吐仍疑似受 400Gbps RDMA NIC 限制: {local_gbps:.1f}Gbps")
    require(local_duration < fabric_duration,
            "7.2Tbps NVLink flow 没有早于 400Gbps RDMA flow 完成")
    return (
        f"NVLink/RDMA 并发启动；本地 {local_duration:.3f}us/"
        f"{local_gbps:.1f}Gbps，fabric {fabric_duration:.3f}us"
    )


def expect_failure(marker: str, expected_code: int | None = None) -> Callable[[str, int], str]:
    def validator(log: str, returncode: int) -> str:
        require(returncode != 0, "非法输入意外成功")
        if expected_code is not None:
            require(returncode == expected_code,
                    f"预期返回码 {expected_code}，实际 {returncode}")
        require(marker in log, f"失败日志缺少诊断: {marker}")
        return f"按预期拒绝，返回码 {returncode}，诊断包含：{marker}"
    return validator


def connection_matrix(connections: list[str]) -> str:
    return "Nodes 32\nConnections " + str(len(connections)) + "\n" \
        + "\n".join(connections) + ("\n" if connections else "")


def write_report(run_dir: Path, results: list[CaseResult], config: dict) -> None:
    passed = sum(result.status == "passed" for result in results)
    summary = {
        "status": "passed" if passed == len(results) else "failed",
        "passed": passed,
        "total": len(results),
        "cases": [asdict(result) for result in results],
    }
    write_text(run_dir / "配置.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    write_text(run_dir / "汇总.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    rows = [
        "# MpRail 功能测试说明",
        "",
        "## 测试目的",
        "",
        "验证服务器内部高速 FullMesh、同 rail L0 转发、跨 rail L0-L1-L0 转发、"
        "plane 隔离、UEC flow/spray ECMP、静态流以及 DAG barrier。",
        "",
        "## 拓扑规模",
        "",
        "32 ranks，8 ranks/server，4 servers，1 plane，8 个 L0 Leaf，4 个 L1 Spine。",
        "每个 Leaf 挂四张同 local-index GPU；RDMA 与 L0/L1 链路均为 400Gbps。",
        "Server-local FullMesh 与每 rank local injection 为独立的 7200Gbps 资源。",
        "",
        "## 测试结果",
        "",
        "| Case | 结果 | 返回码 | 说明 |",
        "|---|---|---:|---|",
    ]
    for result in results:
        status = "通过" if result.status == "passed" else "失败"
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {result.name} | {status} | {result.returncode} | {detail} |")
    rows.extend([
        "",
        "## 汇总",
        "",
        f"共 {len(results)} 个 case，通过 {passed} 个，失败 {len(results) - passed} 个。",
        "每个 case 的输入位于 `inputs/`，完整命令、仿真日志和输出文件位于 `cases/`。",
        "失败现场不会被测试脚本删除。",
        "",
    ])
    write_text(run_dir / "测试说明.md", "\n".join(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="指定日志运行目录；默认创建 test_logs/run_*_mprail")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.run_dir or ROOT / "test_logs" / f"run_{stamp}_mprail").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "binary": str(BINARY),
        "build_dir": str(BUILD_DIR),
        "nodes": 32,
        "planes": 1,
        "gpus_per_server": 8,
        "rail_mapping": "gpu_local_index",
        "l1_eps_per_plane": 4,
        "l0_l1_links_per_spine": 1,
        "external_linkspeed_mbps": 400000,
        "local_linkspeed_mbps": 7200000,
    }
    suite = Suite(run_dir)
    try:
        build_simulator(run_dir)

        suite.run_case(
            "same_server",
            connection_matrix(["0->1 id 1 start 0 size 16384"]),
            validate_same_server,
        )
        suite.run_case(
            "same_rail",
            connection_matrix(["0->8 id 2 start 0 size 16384"]),
            validate_same_rail,
        )
        suite.run_case(
            "cross_rail",
            connection_matrix(["0->9 id 3 start 0 size 16384"]),
            validate_cross_rail,
        )
        suite.run_case(
            "flow_routing_mode",
            connection_matrix(["0->9 id 100 start 0 size 16384"]),
            validate_flow_routing_mode,
            extra_args=["-load_balancing_algo", "ecmp"],
        )
        suite.run_case(
            "spray_routing_mode",
            connection_matrix(["0->9 id 101 start 0 size 16384"]),
            validate_spray_routing_mode,
            extra_args=["-load_balancing_algo", "oblivious"],
        )

        empty_matrix = connection_matrix([])
        suite.run_case(
            "dag_barrier",
            empty_matrix,
            validate_dag,
            dag=(
                "1 0 | 0 8 | 16384 0 | -\n"
                "2 0 | 0 9 | 16384 0 | -\n"
                "3 0 | 0 0 | 0 20 | -\n"
                "4 0 | 1 1 | 0 30 | -\n"
                "5 1 | 8 0 | 16384 0 | 0\n"
                "6 1 | 8 8 | 0 10 | 0\n"
            ),
        )
        suite.run_case(
            "dag_multiple_predecessors",
            empty_matrix,
            validate_multiple_predecessors,
            dag=(
                "1 0 | 0 0 | 0 10 | -\n"
                "2 1 | 1 1 | 0 20 | -\n"
                "3 2 | 2 2 | 0 5 | 0 1\n"
            ),
        )
        suite.run_case(
            "dag_independent_barriers",
            empty_matrix,
            validate_independent_barriers,
            dag=(
                "1 0 | 0 0 | 0 10 | -\n"       # Attention 1
                "2 1 | 0 0 | 0 30 | 0\n"       # Dispatch 1
                "3 2 | 0 0 | 0 10 | 0\n"       # Attention 2
                "4 3 | 0 0 | 0 5 | 1\n"        # Expert 1
                "5 4 | 0 0 | 0 5 | 2\n"        # Dispatch 2
            ),
        )
        suite.run_case(
            "local_fabric_independent_injection",
            empty_matrix,
            validate_local_fabric_independence,
            dag=(
                "1 0 | 0 1 | 16777216 0 | -\n"
                "2 1 | 0 8 | 16777216 0 | -\n"
            ),
            end_us=5000,
        )
        suite.run_case(
            "reject_joint_task",
            empty_matrix,
            expect_failure("exactly one of transfer_bytes or compute_us"),
            dag="1 0 | 0 8 | 1024 10 | -\n",
        )
        suite.run_case(
            "reject_empty_task",
            empty_matrix,
            expect_failure("exactly one of transfer_bytes or compute_us"),
            dag="1 0 | 0 0 | 0 0 | -\n",
        )
        suite.run_case(
            "reject_cycle",
            empty_matrix,
            expect_failure("DAG contains a cycle"),
            dag="1 0 | 0 0 | 0 1 | 1\n2 1 | 1 1 | 0 1 | 0\n",
        )
        suite.run_case(
            "reject_missing_predecessor",
            empty_matrix,
            expect_failure("depends on a barrier that has no tasks"),
            dag="1 0 | 0 0 | 0 1 | 99\n",
        )
        suite.run_case(
            "reject_rank_out_of_range",
            empty_matrix,
            expect_failure("invalid rank: 32"),
            dag="1 0 | 0 32 | 1024 0 | -\n",
        )
        suite.run_case(
            "detect_dag_timeout",
            empty_matrix,
            expect_failure("DAG did not finish before the simulation stopped", 2),
            dag="1 0 | 0 0 | 0 100 | -\n",
            end_us=10,
        )
        suite.run_case(
            "reject_same_rank_network",
            empty_matrix,
            expect_failure("network task requires src_rank != dst_rank"),
            dag="1 0 | 0 0 | 1024 0 | -\n",
        )
        suite.run_case(
            "reject_cross_rank_compute",
            empty_matrix,
            expect_failure("compute task requires src_rank == dst_rank"),
            dag="1 0 | 0 1 | 0 10 | -\n",
        )
        suite.run_case(
            "reject_missing_group_separators",
            empty_matrix,
            expect_failure("malformed DAG task"),
            dag="1 0 0 8 1024 0 -\n",
        )
        suite.run_case(
            "reject_invalid_dimension",
            connection_matrix(["0->8 id 1 start 0 size 1024"]),
            expect_failure(
                "-mprail_gpus_per_server requires a positive integer"
            ),
            extra_args=["-mprail_gpus_per_server", "-1"],
        )
    except Exception as exc:
        suite.results.append(CaseResult(
            "build", "failed", -1, str(exc), []
        ))
    finally:
        write_report(run_dir, suite.results, config)

    print(run_dir)
    return 0 if suite.results and all(
        result.status == "passed" for result in suite.results
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
