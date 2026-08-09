#!/usr/bin/env python3
"""Validate dynamic DAG append and observation in one HTSim process."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT / "htsim" / "sim" / "build-mprail" / "datacenter" / "htsim_uec"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def send(process: subprocess.Popen[str], lines: list[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("\n".join(lines) + "\n")
    process.stdin.flush()


def main() -> int:
    require(SIMULATOR.is_file(), f"missing simulator: {SIMULATOR}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_dynamic_dag_functional"
    run_dir.mkdir(parents=True)
    matrix = run_dir / "nodes.cm"
    matrix.write_text("Nodes 2\nConnections 0\n", encoding="utf-8")
    log_path = run_dir / "htsim.log"
    command = [
        str(SIMULATOR),
        "-topology", "mprail",
        "-mprail_planes", "1",
        "-mprail_gpus_per_server", "1",
        "-mprail_l1_eps_per_plane", "1",
        "-mprail_l0_l1_links_per_spine", "1",
        "-linkspeed", "400000",
        "-local_linkspeed", "7200000",
        "-q", "128",
        "-end", "100",
        "-tm", str(matrix),
        "-dag_control",
        "-o", str(run_dir / "htsim.dat"),
    ]
    (run_dir / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output: list[str] = []
    sent_initial = False
    for line in process.stdout:
        output.append(line)
        if "DAG_CONTROL_READY" in line:
            require(not sent_initial, "HTSim requested the initial batch twice")
            sent_initial = True
            send(process, [
                "DAG_APPEND_BEGIN batch0",
                "DAG_TASK 1 1 | 0 0 | 0 5 | -",
                "DAG_OBSERVE 0 | 1",
                "DAG_APPEND_END batch0",
            ])
        elif "DAG_OBSERVATION_READY observation=0" in line:
            send(process, [
                "DAG_APPEND_BEGIN batch1",
                "DAG_TASK 2 2 | 0 0 | 0 7 | 1",
                "DAG_OBSERVE 1 | 2",
                "DAG_APPEND_END batch1",
            ])
        elif "DAG_OBSERVATION_READY observation=1" in line:
            send(process, ["DAG_CLOSE"])
    returncode = process.wait(timeout=30)
    text = "".join(output)
    log_path.write_text(text, encoding="utf-8")

    require(returncode == 0, f"HTSim failed with exit code {returncode}")
    require(sent_initial, "HTSim never entered dynamic control mode")
    require(text.count("DAG_APPEND_ACK") == 2, "expected two append ACKs")
    require(text.count("DAG_OBSERVATION_READY") == 2,
            "expected two observations")
    require(text.count("DAG_SUMMARY") == 1, "expected exactly one DAG summary")
    require(
        re.search(r"DAG_APPEND_ACK batch=batch1 .*time_us=5(?:\.0+)?(?:\s|$)", text)
        is not None,
        "second batch was not appended at the first observation time",
    )
    require(
        re.search(r"DAG_TASK_START task=2 .*time_us=5(?:\.0+)?(?:\s|$)", text)
        is not None,
        "second task did not start at 5 us",
    )
    summary = re.search(
        r"DAG_SUMMARY tasks=2 barriers=2 observations=2 batches=2 "
        r"makespan_us=([0-9.]+)",
        text,
    )
    require(summary is not None, "dynamic DAG summary fields are incomplete")
    require(abs(float(summary.group(1)) - 12.0) < 1e-9,
            "dynamic DAG makespan must be 12 us")

    (run_dir / "测试说明.md").write_text(
        "# 动态 DAG 功能测试\n\n"
        "- 只启动 1 个 HTSim 进程。\n"
        "- batch0 在 0 us 执行 5 us compute。\n"
        "- observation 0 在 5 us 触发，batch1 在同一模拟时刻追加。\n"
        "- batch1 执行 7 us compute，最终 makespan 为 12 us。\n"
        "- 日志中只有 1 个 `DAG_SUMMARY`。\n",
        encoding="utf-8",
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
