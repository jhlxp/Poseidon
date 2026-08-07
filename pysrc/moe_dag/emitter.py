from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .graph import Task, TaskGraph


@dataclass(frozen=True)
class EmissionResult:
    output_dir: Path
    dag_path: Path
    matrix_path: Path
    manifest_path: Path
    task_map_path: Path
    report_path: Path
    task_ids: dict[str, int]
    barrier_ids: dict[str, int]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _format_duration_us(duration_us: float) -> str:
    value = f"{duration_us:.9f}".rstrip("0").rstrip(".")
    return value if value else "0"


def emit_workload(
    graph: TaskGraph,
    output_dir: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> EmissionResult:
    graph.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_keys = graph.topological_keys()
    ordered_tasks = [graph.task(key) for key in ordered_keys]
    task_ids = {task.key: index for index, task in enumerate(ordered_tasks, start=1)}

    barrier_keys: list[str] = []
    for task in ordered_tasks:
        if task.barrier_key not in barrier_keys:
            barrier_keys.append(task.barrier_key)
    barrier_ids = {key: index for index, key in enumerate(barrier_keys)}

    lines = [
        "# task_id barrier_id | src_rank dst_rank | transfer_bytes compute_us "
        "| predecessor_barriers [| route_spec]"
    ]
    task_records: list[dict[str, Any]] = []
    for task in ordered_tasks:
        predecessor_barriers = sorted(
            {barrier_ids[graph.task(key).barrier_key] for key in task.predecessors}
        )
        predecessors = (
            " ".join(str(item) for item in predecessor_barriers)
            if predecessor_barriers
            else "-"
        )
        barrier_id = barrier_ids[task.barrier_key]
        if task.kind == "compute":
            assert task.rank is not None
            endpoints = f"{task.rank} {task.rank}"
            operation = f"0 {_format_duration_us(task.duration_us)}"
        else:
            assert task.src_rank is not None and task.dst_rank is not None
            endpoints = f"{task.src_rank} {task.dst_rank}"
            operation = f"{task.transfer_bytes} 0"
        line = (
            f"{task_ids[task.key]} {barrier_id} | {endpoints} | "
            f"{operation} | {predecessors}"
        )
        if task.route_spec:
            line += f" | {task.route_spec}"
        lines.append(line)

        record = asdict(task)
        record.update(
            {
                "task_id": task_ids[task.key],
                "barrier_id": barrier_id,
                "predecessor_barrier_ids": predecessor_barriers,
                "logical_stream": task.metadata.get(
                    "logical_stream",
                    "compute" if task.kind == "compute" else "communication",
                ),
            }
        )
        record["predecessors"] = sorted(task.predecessors)
        task_records.append(record)

    dag_path = output_dir / "workload.dag"
    matrix_path = output_dir / "nodes.cm"
    manifest_path = output_dir / "manifest.json"
    task_map_path = output_dir / "task_map.json"
    report_path = output_dir / "生成报告.md"
    dag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix_path.write_text(
        f"Nodes {graph.num_ranks}\nConnections 0\n", encoding="utf-8"
    )

    kind_counts = Counter(task.kind for task in ordered_tasks)
    payload_bytes = Counter()
    for task in ordered_tasks:
        if task.kind == "transfer":
            payload_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes

    manifest = {
        "format_version": 1,
        "graph": graph.name,
        "num_ranks": graph.num_ranks,
        "task_count": len(ordered_tasks),
        "barrier_count": len(barrier_ids),
        "compute_task_count": kind_counts["compute"],
        "transfer_task_count": kind_counts["transfer"],
        "transfer_bytes_by_payload": dict(sorted(payload_bytes.items())),
        "metadata": metadata or {},
    }
    _write_json(manifest_path, manifest)
    _write_json(task_map_path, {"tasks": task_records})

    report_lines = [
        f"# {graph.name} 生成报告",
        "",
        "## 汇总",
        "",
        f"- rank 数：{graph.num_ranks}",
        f"- task 数：{len(ordered_tasks)}",
        f"- barrier 数：{len(barrier_ids)}",
        f"- compute task：{kind_counts['compute']}",
        f"- transfer task：{kind_counts['transfer']}",
        "",
        "## 传输字节",
        "",
        "| payload_kind | bytes |",
        "|---|---:|",
    ]
    if payload_bytes:
        report_lines.extend(
            f"| {kind} | {count} |" for kind, count in sorted(payload_bytes.items())
        )
    else:
        report_lines.append("| - | 0 |")
    report_lines.extend(
        [
            "",
            "## 模型边界",
            "",
            f"- 算法：`{(metadata or {}).get('algorithm', '未指定')}`",
            "- network task 的完成点是整条 flow/chunk 收到完整 ACK。",
            "- compute task 使用生成时确定的固定 `compute_us`。",
            "- 逻辑 stream 顺序由生成器降低为 predecessor edges；HTSim 不做运行时 stream 调度。",
            "- 不模拟单 flow 包进度、动态 SM、CUDA occupancy 或 HBM 竞争。",
            "",
            "## 产物",
            "",
            "- `workload.dag`：HTSim barrier DAG。",
            "- `nodes.cm`：DAG 模式所需的空 connection matrix。",
            "- `manifest.json`：生成配置与统计。",
            "- `task_map.json`：task ID、barrier ID 与逻辑 task 的映射。",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return EmissionResult(
        output_dir=output_dir,
        dag_path=dag_path,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        task_map_path=task_map_path,
        report_path=report_path,
        task_ids=task_ids,
        barrier_ids=barrier_ids,
    )
