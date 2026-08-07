#!/usr/bin/env python3
"""Verify flow-level ECMP and packet-level spray behavior on MpRail."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "htsim" / "sim"
BUILD_DIR = SIM_DIR / "build-mprail"
BINARY = BUILD_DIR / "datacenter" / "htsim_uec"


@dataclass
class ModeResult:
    name: str
    status: str
    simulator_returncode: int
    entropy_values_by_flow: dict[str, list[int]]
    preferred_planes: list[int]
    active_plane_ingress: dict[str, int]
    active_forward_paths: dict[str, int]
    detail: str
    command: list[str]


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
            raise RuntimeError(f"构建失败，返回码 {completed.returncode}")
    write_text(run_dir / "构建.log", "\n".join(chunks))


def connection_matrix(flows: list[str]) -> str:
    return "\n".join([
        "Nodes 32",
        f"Connections {len(flows)}",
        *flows,
        "",
    ])


def parse_entropies(log: str) -> dict[str, list[int]]:
    values: dict[str, set[int]] = {}
    pattern = re.compile(
        r"Uec_(\d+)_(\d+) sending pkt \d+ .*? ev (\d+) in_flight"
    )
    for src, dst, entropy in pattern.findall(log):
        values.setdefault(f"{src}->{dst}", set()).add(int(entropy))
    return {flow: sorted(entropies) for flow, entropies in sorted(values.items())}


def parse_preferred_planes(log: str) -> list[int]:
    return sorted({
        int(plane)
        for plane in re.findall(
            r"scope=cross_rail .*?routing=flow_ecmp plane=(\d+)", log
        )
    })


def parse_active_links(case_dir: Path, pattern: re.Pattern[str]) -> dict[str, int]:
    info_path = case_dir / "output_metrics" / "link_info.csv"
    load_path = case_dir / "output_metrics" / "link_load_1ms.csv"
    link_names: dict[int, str] = {}
    with info_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            link_names[int(row["link_id"])] = row["link_name"]

    totals: dict[int, int] = {}
    with load_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            link_id = int(row["link_id"])
            totals[link_id] = totals.get(link_id, 0) + int(row["bytes"])

    result: dict[str, int] = {}
    for link_id, byte_count in sorted(totals.items()):
        name = link_names.get(link_id, "")
        if byte_count > 128 and pattern.match(name):
            result[f"link_id={link_id} {name}"] = byte_count
    return result


def parse_active_plane_ingress(case_dir: Path) -> dict[str, int]:
    return parse_active_links(
        case_dir,
        re.compile(r"^MPRAIL_HOST_SRC_\d+->MPRAIL_L0_r0_p0\(b\d+\)$"),
    )


def parse_active_forward_paths(case_dir: Path) -> dict[str, int]:
    return parse_active_links(
        case_dir,
        re.compile(r"^MPRAIL_L0_r0_p0->MPRAIL_L1_p0_s\d+\(b0\)$"),
    )


def base_command(matrix_path: Path, case_dir: Path, algorithm: str) -> list[str]:
    return [
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
        "-end", "1000",
        "-strat", "ecmp_host",
        "-load_balancing_algo", algorithm,
        "-tm", str(matrix_path),
        "-o", str(case_dir / "htsim.dat"),
        "-debug",
    ]


def run_mode(
    run_dir: Path,
    name: str,
    algorithm: str,
    matrix: str,
) -> ModeResult:
    case_dir = run_dir / "cases" / name
    matrix_path = run_dir / "inputs" / f"{name}.cm"
    (case_dir / "output_metrics").mkdir(parents=True, exist_ok=True)
    write_text(matrix_path, matrix)
    command = base_command(matrix_path, case_dir, algorithm)
    write_text(case_dir / "命令.txt", " ".join(command) + "\n")

    env = os.environ.copy()
    env["HTSIM_LINK_LOAD_SAMPLE"] = "1"
    env["HTSIM_LINK_LOAD_SAMPLE_US"] = "10"
    completed = subprocess.run(
        command,
        cwd=case_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    write_text(case_dir / "htsim.log", completed.stdout)

    entropies = parse_entropies(completed.stdout)
    preferred_planes = parse_preferred_planes(completed.stdout)
    active_ingress = parse_active_plane_ingress(case_dir)
    active_paths = parse_active_forward_paths(case_dir)
    completed_ok = completed.returncode == 0 and "Done" in completed.stdout

    if name == "flow_level_ecmp":
        stable = entropies and all(len(items) == 1 for items in entropies.values())
        pinned = len(preferred_planes) == 1
        one_physical_path = len(active_ingress) == 1 and len(active_paths) == 1
        supported = completed_ok and stable and pinned and one_physical_path
        detail = (
            f"单流熵数量 {len(entropies.get('0->9', []))}；"
            f"plane 入口 {len(active_ingress)} 条；"
            f"L0->L1 链路 {len(active_paths)} 条"
        )
    elif name == "flow_level_distribution":
        stable = len(entropies) == 8 and all(
            len(items) == 1 for items in entropies.values()
        )
        distributed = preferred_planes == [0] and len(active_paths) > 1
        supported = completed_ok and stable and distributed
        detail = (
            f"8 条流的包内熵均固定={stable}；"
            "固定使用 plane 0；"
            f"实际使用 {len(active_paths)}/8 条 L0->L1 spine 链路"
        )
    else:
        entropy_changes = len(entropies.get("0->9", [])) > 1
        one_plane = len(active_ingress) == 1
        physical_spray = len(active_paths) > 1
        supported = completed_ok and entropy_changes and one_plane and physical_spray
        detail = (
            f"单流报文熵数量 {len(entropies.get('0->9', []))}；"
            "固定使用 plane 0；"
            f"实际使用 {len(active_paths)}/8 条 L0->L1 spine 链路；"
            f"报文级物理分流={physical_spray}"
        )

    return ModeResult(
        name=name,
        status="passed" if supported else "failed",
        simulator_returncode=completed.returncode,
        entropy_values_by_flow=entropies,
        preferred_planes=preferred_planes,
        active_plane_ingress=active_ingress,
        active_forward_paths=active_paths,
        detail=detail,
        command=command,
    )


def write_report(run_dir: Path, results: list[ModeResult]) -> None:
    lines = [
        "# MpRail 路由模式测试",
        "",
        "本测试区分报文携带的路径熵与交换机实际使用的物理链路。",
        "两种模式都使用 `-strat ecmp_host`，只改变端侧多路径算法。",
        "",
        "| 模式 | 参数 | 结果 | 观测 |",
        "|---|---|---|---|",
    ]
    algorithms = {
        "flow_level_ecmp": "`-load_balancing_algo ecmp`",
        "flow_level_distribution": "`-load_balancing_algo ecmp`",
        "spray_packet_ecmp": "`-load_balancing_algo oblivious`",
    }
    for result in results:
        lines.append(
            f"| `{result.name}` | {algorithms[result.name]} | "
            f"`{result.status}` | {result.detail} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        "- `flow_level_ecmp`：单流内部的熵、plane 0 和实际 ECMP 路径必须固定。",
        "- `flow_level_distribution`：不同 flow 必须能分散到不同 Spine ECMP 路径。",
        "- `spray_packet_ecmp`：单流不同报文的熵必须变化，并且必须分散到多条实际链路。",
        "",
        "## 代码路径核验",
        "",
        "MpRail 的 L0/L1 使用普通多下一跳 FIB，统一按 `(flow_id, pathid, switch_salt)` 做 ECMP。",
        "`ecmp` 的 `pathid` 在流内固定；`oblivious` 逐包改变 `pathid`。本测试只有",
        "plane 0，因此两者都在单 plane 内对 8 个 Spine 做 ECMP。",
        "",
        "仿真日志、命令、输入矩阵和链路采样 CSV 位于 `cases/`。",
        "机器可读结果位于 `汇总.json`。",
        "",
    ])
    write_text(run_dir / "测试说明.md", "\n".join(lines))
    write_text(
        run_dir / "汇总.json",
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2)
        + "\n",
    )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_mprail_routing_modes"
    run_dir.mkdir(parents=True)
    build_simulator(run_dir)

    single_flow_matrix = connection_matrix([
        "0->9 id 100 start 0 size 65536",
    ])
    flow_distribution_matrix = connection_matrix([
        f"0->{dst} id {100 + index} start 0 size 16384"
        for index, dst in enumerate((9, 10, 11, 12, 13, 14, 15, 17))
    ])
    spray_matrix = connection_matrix([
        "0->9 id 200 start 0 size 65536",
    ])
    results = [
        run_mode(run_dir, "flow_level_ecmp", "ecmp", single_flow_matrix),
        run_mode(
            run_dir,
            "flow_level_distribution",
            "ecmp",
            flow_distribution_matrix,
        ),
        run_mode(run_dir, "spray_packet_ecmp", "oblivious", spray_matrix),
    ]
    write_report(run_dir, results)

    print(f"测试目录: {run_dir}")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    return 0 if all(result.status == "passed" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
