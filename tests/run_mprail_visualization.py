#!/usr/bin/env python3
"""Test the standalone MpRail link-load visualization pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visualization.mprail_link_load import classify_mprail_link  # noqa: E402


PLOTTER = ROOT / "visualization" / "mprail_link_load.py"


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inputs(input_dir: Path) -> None:
    link_rows = [
        {
            "link_id": 0,
            "link_name": (
                "MPRAIL_LOCAL_MPRAIL_HOST_SRC_0->MPRAIL_HOST_DST_1(b0)"
            ),
            "layer": "unknown",
            "direction": "unknown",
            "src_type": "unknown",
            "src_id": -1,
            "dst_type": "unknown",
            "dst_id": -1,
            "bundle": 0,
            "rate_gbps": 3200,
        },
        {
            "link_id": 1,
            "link_name": "MPRAIL_HOST_SRC_8->MPRAIL_L0_r1_p3(b0)",
            "layer": "unknown",
            "direction": "unknown",
            "src_type": "unknown",
            "src_id": -1,
            "dst_type": "unknown",
            "dst_id": -1,
            "bundle": 0,
            "rate_gbps": 400,
        },
        {
            "link_id": 2,
            "link_name": "MPRAIL_L0_r1_p3->MPRAIL_L1_p3_s2(b1)",
            "layer": "unknown",
            "direction": "unknown",
            "src_type": "unknown",
            "src_id": -1,
            "dst_type": "unknown",
            "dst_id": -1,
            "bundle": 1,
            "rate_gbps": 400,
        },
        {
            "link_id": 3,
            "link_name": "MPRAIL_L1_p3_s2->MPRAIL_L0_r2_p3(b0)",
            "layer": "unknown",
            "direction": "unknown",
            "src_type": "unknown",
            "src_id": -1,
            "dst_type": "unknown",
            "dst_id": -1,
            "bundle": 0,
            "rate_gbps": 400,
        },
        {
            "link_id": 4,
            "link_name": "MPRAIL_L0_r2_p3->MPRAIL_HOST_DST_16(b0)",
            "layer": "unknown",
            "direction": "unknown",
            "src_type": "unknown",
            "src_id": -1,
            "dst_type": "unknown",
            "dst_id": -1,
            "bundle": 0,
            "rate_gbps": 400,
        },
    ]
    write_csv(
        input_dir / "link_info.csv",
        [
            "link_id",
            "link_name",
            "layer",
            "direction",
            "src_type",
            "src_id",
            "dst_type",
            "dst_id",
            "bundle",
            "rate_gbps",
        ],
        link_rows,
    )

    throughput_by_link = {
        0: [(0, 4_000_000, 3200.0, 0), (1, 2_000_000, 1600.0, 0)],
        1: [(0, 500_000, 400.0, 1024), (2, 250_000, 200.0, 2048)],
        2: [(0, 500_000, 400.0, 4096), (2, 125_000, 100.0, 2048)],
        3: [(1, 375_000, 300.0, 1024), (3, 250_000, 200.0, 0)],
        4: [(1, 250_000, 200.0, 0), (3, 125_000, 100.0, 0)],
    }
    load_rows: list[dict[str, object]] = []
    for link_id, values in throughput_by_link.items():
        for bucket, byte_count, throughput, max_queue in values:
            load_rows.append(
                {
                    "time_ms": bucket * 0.01,
                    "bucket": bucket,
                    "link_id": link_id,
                    "bytes": byte_count,
                    "throughput_gbps": throughput,
                    "max_queue_bytes": max_queue,
                }
            )
    write_csv(
        input_dir / "link_load_1ms.csv",
        [
            "time_ms",
            "bucket",
            "link_id",
            "bytes",
            "throughput_gbps",
            "max_queue_bytes",
        ],
        load_rows,
    )


def parsing_case() -> str:
    cases = [
        (0, "MPRAIL_LOCAL_MPRAIL_HOST_SRC_0->MPRAIL_HOST_DST_1(b0)", "server_local"),
        (1, "MPRAIL_HOST_SRC_8->MPRAIL_L0_r1_p3(b0)", "host_l0_up"),
        (2, "MPRAIL_L0_r1_p3->MPRAIL_L1_p3_s2(b1)", "l0_l1_up"),
        (3, "MPRAIL_L1_p3_s2->MPRAIL_L0_r2_p3(b0)", "l1_l0_down"),
        (4, "MPRAIL_L0_r2_p3->MPRAIL_HOST_DST_16(b0)", "l0_host_down"),
    ]
    parsed = [classify_mprail_link(item, name, 400) for item, name, _ in cases]
    require(
        [item.panel for item in parsed] == [panel for _, _, panel in cases],
        "五类链路分类错误",
    )
    require(parsed[1].src_rank == 8 and parsed[1].rail == 1 and parsed[1].plane == 3,
            "Host->L0 坐标解析错误")
    require(parsed[2].spine == 2 and parsed[2].bundle == 1,
            "L0->L1 spine/bundle 解析错误")
    return "五类 MpRail 链路均恢复出正确 panel 和拓扑坐标"


def reject_cross_plane_case() -> str:
    try:
        classify_mprail_link(
            7,
            "MPRAIL_L0_r0_p0->MPRAIL_L1_p1_s0(b0)",
            400,
        )
    except ValueError as exc:
        require("crosses planes" in str(exc), "跨 plane 错误信息不明确")
        return "L0 p0 -> L1 p1 被确定性拒绝"
    raise AssertionError("跨 plane 链路没有被拒绝")


def run_plotter(run_dir: Path, input_dir: Path, output_dir: Path) -> str:
    command = [
        sys.executable,
        str(PLOTTER),
        "--metrics-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--title",
        "MpRail visualization functional test",
        "--planes",
        "8",
        "--dpi",
        "100",
    ]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(run_dir / "matplotlib-cache")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    (run_dir / "命令.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (run_dir / "绘图.log").write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"绘图进程返回码 {completed.returncode}")
    require("parsed 5 MpRail links across 4 time buckets" in completed.stdout,
            "绘图日志的链路或时间桶数量错误")
    return "CLI 成功读取 5 links/4 buckets，按 8 planes 生成四类产物"


def png_case(output_dir: Path) -> str:
    path = output_dir / "mprail_link_load_by_layer.png"
    data = path.read_bytes()
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "输出不是 PNG")
    require(len(data) > 50_000, "PNG 内容异常小")
    width, height = struct.unpack(">II", data[16:24])
    require(width >= 1100 and height >= 900, f"PNG 尺寸异常: {width}x{height}")
    return f"四行/七面板 PNG 为 {width}x{height}，文件 {len(data)} bytes"


def summary_case(output_dir: Path) -> str:
    with (output_dir / "mprail_link_load_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["panel"]: row for row in csv.DictReader(handle)}
    require(len(rows) == 5, "summary 不是五个面板")
    require(all(row["active_links"] == "1" for row in rows.values()),
            "每面板应有一条 active link")
    require(rows["server_local"]["line_rate_gbps"] == "3200.0",
            "server-local line rate 错误")
    require(rows["l0_l1_up"]["line_rate_gbps"] == "400.0",
            "fabric line rate 错误")
    require(rows["server_local"]["total_bytes"] == "6000000",
            "server-local 总字节错误")
    require(rows["l0_l1_up"]["max_queue_bytes"] == "4096",
            "最大队列统计错误")

    with (output_dir / "mprail_endpoint_load_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        endpoints = {row["direction"]: row for row in csv.DictReader(handle)}
    require(endpoints["output"]["configured_planes"] == "8",
            "endpoint summary plane 数错误")
    require(endpoints["output"]["aggregate_line_rate_gbps"] == "3200.0",
            "8x400 Gbps 聚合容量错误")
    require(math.isclose(float(endpoints["output"]["peak_utilization"]), 0.125),
            "output peak utilization 应为 12.5%")
    require(math.isclose(float(endpoints["output"]["peak_headroom"]), 0.875),
            "output peak headroom 应为 87.5%")
    require(math.isclose(float(endpoints["input"]["peak_utilization"]), 0.0625),
            "input peak utilization 应为 6.25%")

    with (output_dir / "mprail_link_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        inventory = list(csv.DictReader(handle))
    require(len(inventory) == 5, "inventory 链路数错误")
    fabric = next(row for row in inventory if row["panel"] == "l0_l1_up")
    require(
        fabric["rail"] == "1"
        and fabric["plane"] == "3"
        and fabric["spine"] == "2"
        and fabric["bundle"] == "1",
        "inventory fabric 坐标错误",
    )
    return (
        "8x400=3200 Gbps；output peak utilization/headroom=12.5%/87.5%；"
        "inventory 恢复 rail1/plane3/spine2/bundle1"
    )


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
        "# MpRail 链路负载可视化测试报告",
        "",
        "本测试使用五类链路和四个 10us 时间桶的固定合成采样，验证名称解析、",
        "跨 plane 校验、四行/七面板 PNG、endpoint utilization/headroom、",
        "分层 summary 和结构化 inventory。",
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
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_mprail_visualization"
    input_dir = run_dir / "inputs" / "output_metrics"
    output_dir = run_dir / "visualization"
    output_dir.mkdir(parents=True)
    build_inputs(input_dir)

    suite = Suite()
    suite.run("parse_five_link_classes", parsing_case)
    suite.run("reject_cross_plane", reject_cross_plane_case)
    suite.run("plotter_cli", lambda: run_plotter(run_dir, input_dir, output_dir))
    suite.run("validate_png", lambda: png_case(output_dir))
    suite.run("validate_summary_inventory", lambda: summary_case(output_dir))
    write_report(run_dir, suite.results)

    passed = sum(item.status == "passed" for item in suite.results)
    print(f"MpRail visualization tests: {passed}/{len(suite.results)} passed")
    print(f"log directory: {run_dir}")
    for item in suite.results:
        print(f"[{item.status}] {item.name}: {item.detail}")
    return 0 if passed == len(suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
