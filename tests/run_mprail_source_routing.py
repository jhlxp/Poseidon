#!/usr/bin/env python3
"""Test MpRail explicit routes and server-internal forwarding."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
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


def connection_matrix(flows: list[str]) -> str:
    return "\n".join([
        "Nodes 16",
        f"Connections {len(flows)}",
        *flows,
        "",
    ])


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


class Suite:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.results: list[CaseResult] = []

    def run_case(
        self,
        name: str,
        matrix: str,
        validator: Callable[[str, int], str],
        *,
        dag: str | None = None,
        strategy: str = "ecmp_host",
        algorithm: str = "ecmp",
    ) -> None:
        case_dir = self.run_dir / "cases" / name
        input_dir = self.run_dir / "inputs"
        (case_dir / "output_metrics").mkdir(parents=True, exist_ok=True)
        matrix_path = input_dir / f"{name}.cm"
        write_text(matrix_path, matrix)
        command = [
            str(BINARY),
            "-topology", "mprail",
            "-mprail_planes", "8",
            "-mprail_gpus_per_server", "4",
            "-mprail_servers_per_rail", "2",
            "-mprail_l1_eps_per_plane", "2",
            "-mprail_l0_l1_links_per_spine", "2",
            "-linkspeed", "100000",
            "-local_linkspeed", "3200000",
            "-local_latency_ns", "50",
            "-hop_latency", "0.1",
            "-switch_latency", "0.02",
            "-q", "32",
            "-end", "1000",
            "-strat", strategy,
            "-load_balancing_algo", algorithm,
            "-tm", str(matrix_path),
            "-o", str(case_dir / "htsim.dat"),
        ]
        if dag is not None:
            dag_path = input_dir / f"{name}.dag"
            write_text(dag_path, dag)
            command.extend(["-dag", str(dag_path)])

        write_text(case_dir / "命令.txt", " ".join(command) + "\n")
        try:
            env = os.environ.copy()
            env["HTSIM_TRACE_FLOW_COMPLETIONS"] = "1"
            env["HTSIM_TRACE_TRIGGERS"] = "1"
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


def explicit_validator(expected_mode: str) -> Callable[[str, int], str]:
    def validate(log: str, returncode: int) -> str:
        assert_success(log, returncode)
        require("MPRAIL_EXPLICIT_FLOW flow=2 src=0 dst=8 plane=3" in log,
                "没有识别指定的显式路径")
        require(expected_mode in log, f"没有进入预期全局模式 {expected_mode}")
        fabric_links = {
            f"{src}->{dst}(b{bundle})"
            for src, dst, bundle in re.findall(
                r"^MPRAIL_LINK src=(MPRAIL_L[01]\S+) "
                r"dst=(MPRAIL_L[01]\S+) bundle=(\d+)",
                log,
                re.MULTILINE,
            )
        }
        expected = {
            "MPRAIL_L0_r0_p3->MPRAIL_L1_p3_s1(b0)",
            "MPRAIL_L1_p3_s1->MPRAIL_L0_r1_p3(b1)",
            "MPRAIL_L0_r1_p3->MPRAIL_L1_p3_s1(b1)",
            "MPRAIL_L1_p3_s1->MPRAIL_L0_r0_p3(b0)",
        }
        require(fabric_links == expected,
                f"显式路径物化链路不精确: {sorted(fabric_links)}")
        require("MPRAIL_L0_r0_p0" not in log and "MPRAIL_L1_p0" not in log,
                "显式 flow 仍创建或使用了其他 plane")
        return "正反向均严格使用 plane 3、spine 1、bundle 0/1"

    return validate


def validate_explicit_same_server(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("MPRAIL_EXPLICIT_FLOW flow=5 src=0 dst=3 "
            "path=explicit rank:0 rank:3" in log,
            "同服务器显式路径未生效")
    require("MPRAIL_L0" not in log and "MPRAIL_L1" not in log,
            "同服务器显式路径进入了交换网络")
    require("speed_gbps=3200" in log, "同服务器显式路径没有使用本地高速链路")
    return "rank:0 rank:3 只使用 3200Gbps server-local FullMesh"


def validate_explicit_same_rail(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("MPRAIL_EXPLICIT_FLOW flow=6 src=0 dst=4 plane=5" in log,
            "同 rail 显式路径未固定到 plane 5")
    require("MPRAIL_L1" not in log, "同 rail 显式路径错误进入 L1")
    require("MPRAIL_L0_r0_p5" in log, "同 rail 显式路径没有经过指定 L0")
    require("MPRAIL_L0_r0_p0" not in log, "同 rail 显式路径创建了其他 plane")
    return "rank:0 -> L0(r0,p5) -> rank:4，不经过 L1"


def phase_events(log: str, event: str) -> list[tuple[str, float]]:
    return [
        (phase, float(timestamp))
        for phase, timestamp in re.findall(
            rf"^SERVER_FORWARD_PHASE_{event} flow=3 phase=(\w+) "
            rf"(?:src=\d+ dst=\d+ )?time_us=([0-9.]+)$",
            log,
            re.MULTILINE,
        )
    ]


def validate_server_forward_cm(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("SERVER_FORWARD_BEGIN flow=3 src=0 src_relay=3 "
            "dst_relay=11 dst=8 bytes=16384 phases=3" in log,
            "三阶段逻辑 flow 元数据错误")
    starts = phase_events(log, "START")
    dones = phase_events(log, "DONE")
    expected_names = ["src_local", "fabric", "dst_local"]
    require([name for name, _ in starts] == expected_names,
            f"phase 启动顺序错误: {starts}")
    require([name for name, _ in dones] == expected_names,
            f"phase 完成顺序错误: {dones}")
    require(starts[1][1] >= dones[0][1] and starts[2][1] >= dones[1][1],
            "后一个 phase 在前一个完成前启动")
    require("scope=same_server" in log,
            "服务器内部 phase 没有走本地 FullMesh")
    require(re.search(r"MPRAIL_FLOW flow=3 src=3 dst=11 scope=cross_rail", log)
            is not None, "fabric phase 没有使用逻辑 flow ID 或 relay 端点")
    final = re.search(r"^SERVER_FORWARD_DONE flow=3 time_us=([0-9.]+)$",
                      log, re.MULTILINE)
    require(final is not None and float(final.group(1)) >= dones[-1][1],
            "逻辑 flow 在最后一个 phase 前完成")
    return "src_local、fabric、dst_local 严格串行且最终完成语义正确"


def validate_server_forward_skip(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    require("phases=1" in log, "两侧 relay 等于端点时没有跳过本地 phase")
    require([name for name, _ in phase_events(log, "START")] == ["fabric"],
            "跳过场景仍启动了本地 phase")
    require("SERVER_FORWARD_DONE flow=3" in log, "单阶段逻辑 flow 未完成")
    return "src_relay=src 且 dst_relay=dst 时只创建 fabric phase"


def validate_server_forward_trigger(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    logical_done = event_time(
        log, r"^SERVER_FORWARD_DONE flow=3 time_us=([0-9.]+)$"
    )
    require("Trigger 1 fired" in log, "末 phase 没有触发 send_done_trigger")
    activation = re.search(r"^Uec_4_12 activate$", log, re.MULTILINE)
    require(activation is not None, "trigger 没有启动后续 flow 4")
    require(activation.start() > log.find("SERVER_FORWARD_DONE flow=3"),
            "后续 flow 在 server_forward 完成日志前激活")
    flow_done = event_time(
        log,
        r"^Flow Uec_4_12 flowId 4 .* finished at ([0-9.]+)",
    )
    require(flow_done > logical_done, "后续 flow 没有在 server_forward 完成后运行")
    return "send_done_trigger 在 dst_local 完成后启动并完成后续 flow"


def validate_dag_explicit(log: str, returncode: int) -> str:
    explicit_validator("routing_mode packet_spray_ecmp")(log, returncode)
    require("DAG_NETWORK_DONE task=2" in log, "显式 DAG network task 未完成")
    require("DAG_SUMMARY tasks=1 barriers=1" in log, "显式 DAG 未正常收敛")
    return "DAG 第五组显式路径生效并完成 barrier"


def event_time(log: str, pattern: str) -> float:
    match = re.search(pattern, log, re.MULTILINE)
    require(match is not None, f"缺少日志: {pattern}")
    return float(match.group(1))


def validate_dag_server_forward(log: str, returncode: int) -> str:
    assert_success(log, returncode)
    validate_server_forward_cm(log, returncode)
    dst_done = event_time(
        log,
        r"^SERVER_FORWARD_PHASE_DONE flow=3 phase=dst_local time_us=([0-9.]+)$",
    )
    network_done = event_time(log, r"^DAG_NETWORK_DONE task=3 time_us=([0-9.]+)$")
    barrier_one_start = event_time(
        log, r"^DAG_BARRIER_START barrier=1 time_us=([0-9.]+)$"
    )
    require(network_done >= dst_done, "DAG task 在 dst_local 完成前通知完成")
    require(barrier_one_start >= network_done, "后继 barrier 在逻辑 flow 完成前启动")
    require("DAG_COMPUTE_DONE task=4" in log, "后继 compute task 未完成")
    return "DAG barrier 等待三阶段最终完成后才释放后继 barrier"


def expect_failure(fragment: str) -> Callable[[str, int], str]:
    def validate(log: str, returncode: int) -> str:
        require(returncode != 0, "非法输入应返回非零状态")
        require(fragment in log, f"错误日志缺少预期信息: {fragment}")
        return f"确定性拒绝，错误包含：{fragment}"

    return validate


def write_report(run_dir: Path, results: list[CaseResult]) -> None:
    passed = sum(result.status == "passed" for result in results)
    summary = {
        "suite": "mprail_source_routing",
        "passed": passed,
        "failed": len(results) - passed,
        "cases": [asdict(result) for result in results],
    }
    write_text(run_dir / "summary.json",
               json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# MpRail 源路由与服务器转发测试",
        "",
        "## 测试内容",
        "",
        "- CM/DAG 完整显式路径及其正反向精确链路。",
        "- 同服务器、同 rail、跨 rail 三种显式路径形状。",
        "- 显式路径覆盖 flow ECMP、oblivious spray 和 ecmp_rr。",
        "- 服务器内部 src_local、fabric、dst_local 严格串行与 phase 跳过。",
        "- server_forward 最后 phase 的 CM trigger 链。",
        "- DAG 后继 barrier 等待最终 dst_local 完成。",
        "- 跨 plane 显式路径、错误 relay、compute route 和缺字段确定性失败。",
        "",
        "## 结果",
        "",
        f"通过 {passed}/{len(results)}，失败 {len(results) - passed}/{len(results)}。",
        "",
        "| 用例 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for result in results:
        lines.append(f"| `{result.name}` | {result.status} | {result.detail} |")
    lines.extend([
        "",
        "每个用例的输入、命令和完整仿真输出位于 `inputs/` 与 `cases/`。",
        "",
    ])
    write_text(run_dir / "测试报告.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_mprail_source_routing"
    run_dir.mkdir(parents=True, exist_ok=False)
    suite = Suite(run_dir)

    explicit_cm = connection_matrix([
        "0->8 id 2 start 0 size 16384 route explicit "
        "rank:0 l0:r0:p3:b0 l1:p3:s1:b1 l0:r1:p3 rank:8"
    ])
    empty_cm = connection_matrix([])
    try:
        if not args.skip_build:
            build_simulator(run_dir)
        suite.run_case(
            "cm_explicit_flow_ecmp",
            explicit_cm,
            explicit_validator("routing_mode flow_ecmp"),
        )
        suite.run_case(
            "cm_explicit_oblivious",
            explicit_cm,
            explicit_validator("routing_mode packet_spray_ecmp"),
            algorithm="oblivious",
        )
        suite.run_case(
            "cm_explicit_ecmp_rr",
            explicit_cm,
            explicit_validator("switch_strategy rr"),
            strategy="ecmp_rr",
            algorithm="oblivious",
        )
        suite.run_case(
            "cm_explicit_same_server",
            connection_matrix([
                "0->3 id 5 start 0 size 16384 route explicit rank:0 rank:3"
            ]),
            validate_explicit_same_server,
            algorithm="oblivious",
        )
        suite.run_case(
            "cm_explicit_same_rail",
            connection_matrix([
                "0->4 id 6 start 0 size 16384 route explicit "
                "rank:0 l0:r0:p5 rank:4"
            ]),
            validate_explicit_same_rail,
            algorithm="oblivious",
        )
        suite.run_case(
            "cm_server_forward_three_phases",
            connection_matrix([
                "0->8 id 3 start 0 size 16384 route server_forward "
                "src_relay:3 dst_relay:11"
            ]),
            validate_server_forward_cm,
        )
        suite.run_case(
            "cm_server_forward_skip_local",
            connection_matrix([
                "0->8 id 3 start 0 size 16384 route server_forward "
                "src_relay:0 dst_relay:8"
            ]),
            validate_server_forward_skip,
        )
        suite.run_case(
            "cm_server_forward_trigger",
            "\n".join([
                "Nodes 16",
                "Connections 2",
                "Triggers 1",
                "0->8 id 3 start 0 size 16384 send_done_trigger 1 "
                "route server_forward src_relay:3 dst_relay:11",
                "4->12 id 4 trigger 1 size 16384",
                "trigger id 1 oneshot",
                "",
            ]),
            validate_server_forward_trigger,
        )
        suite.run_case(
            "dag_explicit",
            empty_cm,
            validate_dag_explicit,
            dag=(
                "2 0 | 0 8 | 16384 0 | - | explicit "
                "rank:0 l0:r0:p3:b0 l1:p3:s1:b1 l0:r1:p3 rank:8\n"
            ),
            algorithm="oblivious",
        )
        suite.run_case(
            "dag_server_forward_barrier",
            empty_cm,
            validate_dag_server_forward,
            dag=(
                "3 0 | 0 8 | 16384 0 | - | server_forward "
                "src_relay:3 dst_relay:11\n"
                "4 1 | 8 8 | 0 10 | 0\n"
            ),
        )
        suite.run_case(
            "reject_explicit_cross_plane",
            connection_matrix([
                "0->8 id 2 start 0 size 16384 route explicit "
                "rank:0 l0:r0:p3:b0 l1:p2:s1:b1 l0:r1:p3 rank:8"
            ]),
            expect_failure("must remain within one valid plane"),
        )
        suite.run_case(
            "reject_wrong_src_relay",
            connection_matrix([
                "0->8 id 3 start 0 size 16384 route server_forward "
                "src_relay:4 dst_relay:11"
            ]),
            expect_failure("src_relay is not on the logical source server"),
        )
        suite.run_case(
            "reject_explicit_wrong_endpoint",
            connection_matrix([
                "0->8 id 2 start 0 size 16384 route explicit "
                "rank:1 l0:r0:p3:b0 l1:p3:s1:b1 l0:r1:p3 rank:8"
            ]),
            expect_failure("endpoint rank does not match the flow"),
        )
        suite.run_case(
            "reject_explicit_bundle_out_of_range",
            connection_matrix([
                "0->8 id 2 start 0 size 16384 route explicit "
                "rank:0 l0:r0:p3:b2 l1:p3:s1:b1 l0:r1:p3 rank:8"
            ]),
            expect_failure("require valid bundle coordinates"),
        )
        suite.run_case(
            "reject_compute_route",
            empty_cm,
            expect_failure("compute task cannot carry a route"),
            dag=(
                "3 0 | 0 0 | 0 10 | - | server_forward "
                "src_relay:0 dst_relay:0\n"
            ),
        )
        suite.run_case(
            "reject_missing_dst_relay",
            connection_matrix([
                "0->8 id 3 start 0 size 16384 route server_forward src_relay:3"
            ]),
            expect_failure("requires src_relay and dst_relay"),
        )
    except Exception as exc:
        suite.results.append(CaseResult("build", "failed", -1, str(exc), []))
    finally:
        write_report(run_dir, suite.results)

    print(run_dir)
    return 0 if suite.results and all(
        result.status == "passed" for result in suite.results
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
