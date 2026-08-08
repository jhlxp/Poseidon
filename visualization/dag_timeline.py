#!/usr/bin/env python3
"""Visualize HTSim DAG execution as per-GPU compute and network timelines."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, dataclass
import gzip
from html import escape
import json
import math
from pathlib import Path
import re
from typing import Iterable

_FLOAT = r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?"
_START_RE = re.compile(
    rf"^DAG_TASK_START task=(?P<task>\d+).* time_us=(?P<time>{_FLOAT})$"
)
_DONE_RE = re.compile(
    rf"^DAG_TASK_DONE task=(?P<task>\d+).* time_us=(?P<time>{_FLOAT})$"
)
_SERVER_FORWARD_RE = re.compile(
    r"^server_forward src_relay:(?P<src>\d+) dst_relay:(?P<dst>\d+)$"
)

COMPUTE_COLORS = {
    "attention": "#2563EB",
    "router_projection": "#C89400",
    "per_server_planning_proxy": "#C89400",
    "expert_ffn": "#16A34A",
    "combine_reduce": "#7C3AED",
    "combine_final_reduce": "#7C3AED",
    "compute": "#64748B",
}
NETWORK_COLORS = {
    "dispatch_hidden": "#EA580C",
    "combine_partial": "#DC2626",
    "expert_weight_prefetch": "#0891B2",
    "expert_weight_scatter": "#0891B2",
    "expert_weight_rdma": "#0891B2",
    "expert_weight_gather": "#0891B2",
    "transfer": "#64748B",
}
LANES = ("Compute", "Network TX", "Network RX")
EXPERT_WEIGHT_PAYLOADS = frozenset(
    {
        "expert_weight_prefetch",
        "expert_weight_scatter",
        "expert_weight_rdma",
        "expert_weight_gather",
    }
)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: int
    barrier_id: int
    key: str
    kind: str
    rank: int | None
    src_rank: int | None
    dst_rank: int | None
    transfer_bytes: int
    declared_duration_us: float
    payload_kind: str | None
    operation: str | None
    communication_phase_id: str | None
    route_spec: str | None
    predecessors: tuple[str, ...]
    predecessor_barrier_ids: tuple[int, ...]
    overlaps_communication: bool
    metadata: dict[str, object]


@dataclass(frozen=True)
class TaskEvent:
    task: TaskDefinition
    start_us: float
    end_us: float

    @property
    def actual_duration_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def logical_throughput_gbps(self) -> float | None:
        if self.task.kind != "transfer" or self.actual_duration_us <= 0:
            return None
        return self.task.transfer_bytes * 8.0 / self.actual_duration_us / 1000.0


@dataclass(frozen=True)
class RankSummary:
    rank: int
    compute_tasks: int
    network_tx_tasks: int
    network_rx_tasks: int
    declared_compute_us: float
    actual_compute_sum_us: float
    compute_active_us: float
    network_active_us: float
    compute_network_overlap_us: float
    overlap_fraction_of_compute: float
    overlap_fraction_of_network: float
    tx_bytes: int
    rx_bytes: int


def _require_int(record: dict[str, object], key: str, context: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {key} must be an integer")
    return value


def _optional_int(record: dict[str, object], key: str, context: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {key} must be an integer or null")
    return value


def read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    num_ranks = payload.get("num_ranks")
    if isinstance(num_ranks, bool) or not isinstance(num_ranks, int) or num_ranks <= 0:
        raise ValueError(f"{path}: num_ranks must be a positive integer")
    task_count = payload.get("task_count")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
    ):
        raise ValueError(f"{path}: task_count must be a positive integer")
    return payload


def read_task_map(path: Path, num_ranks: int) -> dict[int, TaskDefinition]:
    if not path.is_file():
        raise FileNotFoundError(f"missing task map: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: tasks must be a non-empty array")

    result: dict[int, TaskDefinition] = {}
    for index, record in enumerate(records):
        context = f"{path}:tasks[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: task must be an object")
        task_id = _require_int(record, "task_id", context)
        if task_id <= 0 or task_id in result:
            raise ValueError(f"{context}: duplicate or non-positive task_id {task_id}")
        kind = record.get("kind")
        if kind not in {"compute", "transfer"}:
            raise ValueError(f"{context}: unsupported kind {kind!r}")
        key = record.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context}: key must be a non-empty string")
        rank = _optional_int(record, "rank", context)
        src_rank = _optional_int(record, "src_rank", context)
        dst_rank = _optional_int(record, "dst_rank", context)
        if kind == "compute":
            if rank is None or src_rank is not None or dst_rank is not None:
                raise ValueError(f"{context}: invalid compute rank fields")
            ranks = (rank,)
        else:
            if rank is not None or src_rank is None or dst_rank is None:
                raise ValueError(f"{context}: invalid transfer rank fields")
            ranks = (src_rank, dst_rank)
        if any(item < 0 or item >= num_ranks for item in ranks):
            raise ValueError(f"{context}: rank outside [0, {num_ranks})")

        transfer_bytes = _require_int(record, "transfer_bytes", context)
        duration = record.get("duration_us")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ValueError(f"{context}: duration_us must be finite")
        if transfer_bytes < 0 or duration < 0:
            raise ValueError(f"{context}: bytes and duration must be non-negative")
        if kind == "compute" and transfer_bytes != 0:
            raise ValueError(f"{context}: compute task has transfer bytes")
        if kind == "transfer" and transfer_bytes <= 0:
            raise ValueError(f"{context}: transfer task must have positive bytes")

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{context}: metadata must be an object")
        operation = metadata.get("operation")
        if operation is not None and not isinstance(operation, str):
            raise ValueError(f"{context}: metadata.operation must be a string")
        predecessors = record.get("predecessors")
        predecessor_barriers = record.get("predecessor_barrier_ids")
        if not isinstance(predecessors, list) or not all(
            isinstance(item, str) for item in predecessors
        ):
            raise ValueError(f"{context}: predecessors must be a string array")
        if not isinstance(predecessor_barriers, list) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in predecessor_barriers
        ):
            raise ValueError(
                f"{context}: predecessor_barrier_ids must be an integer array"
            )

        result[task_id] = TaskDefinition(
            task_id=task_id,
            barrier_id=_require_int(record, "barrier_id", context),
            key=key,
            kind=kind,
            rank=rank,
            src_rank=src_rank,
            dst_rank=dst_rank,
            transfer_bytes=transfer_bytes,
            declared_duration_us=float(duration),
            payload_kind=(
                str(record["payload_kind"])
                if record.get("payload_kind") is not None
                else None
            ),
            operation=operation,
            communication_phase_id=(
                str(record["communication_phase_id"])
                if record.get("communication_phase_id") is not None
                else None
            ),
            route_spec=(
                str(record["route_spec"])
                if record.get("route_spec") is not None
                else None
            ),
            predecessors=tuple(predecessors),
            predecessor_barrier_ids=tuple(predecessor_barriers),
            overlaps_communication=bool(record.get("overlaps_communication", False)),
            metadata=metadata,
        )
    return result


def read_execution_log(
    path: Path, tasks: dict[int, TaskDefinition]
) -> tuple[TaskEvent, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing HTSim log: {path}")
    starts: dict[int, float] = {}
    dones: dict[int, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _START_RE.fullmatch(line)
        target = starts
        if match is None:
            match = _DONE_RE.fullmatch(line)
            target = dones
        if match is None:
            continue
        task_id = int(match.group("task"))
        if task_id in target:
            raise ValueError(f"{path}:{line_number}: duplicate task event {task_id}")
        target[task_id] = float(match.group("time"))

    expected = set(tasks)
    missing_starts = sorted(expected - set(starts))
    missing_dones = sorted(expected - set(dones))
    unknown = sorted((set(starts) | set(dones)) - expected)
    if missing_starts or missing_dones or unknown:
        raise ValueError(
            "task/log mismatch: "
            f"missing starts={missing_starts[:8]}, "
            f"missing dones={missing_dones[:8]}, unknown={unknown[:8]}"
        )

    events: list[TaskEvent] = []
    for task_id in sorted(tasks):
        start = starts[task_id]
        end = dones[task_id]
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError(f"task {task_id}: invalid interval [{start}, {end}]")
        events.append(TaskEvent(tasks[task_id], start, end))
    return tuple(events)


def parse_rank_selection(value: str | None, num_ranks: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(num_ranks))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("rank selection contains an empty component")
        if "-" in part:
            pieces = part.split("-", 1)
            if not all(piece.isdigit() for piece in pieces):
                raise ValueError(f"invalid rank range: {part}")
            begin, end = map(int, pieces)
            if end < begin:
                raise ValueError(f"descending rank range: {part}")
            selected.update(range(begin, end + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise ValueError(f"invalid rank: {part}")
    if not selected or min(selected) < 0 or max(selected) >= num_ranks:
        raise ValueError(f"selected ranks must be inside [0, {num_ranks})")
    return tuple(sorted(selected))


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total += end - start
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def summarize_ranks(events: tuple[TaskEvent, ...], num_ranks: int) -> tuple[RankSummary, ...]:
    summaries: list[RankSummary] = []
    for rank in range(num_ranks):
        compute = [event for event in events if event.task.rank == rank]
        tx = [event for event in events if rank in _network_tx_ranks(event.task)]
        rx = [event for event in events if rank in _network_rx_ranks(event.task)]
        network_by_id = {event.task.task_id: event for event in (*tx, *rx)}
        compute_intervals = _merge_intervals(
            (event.start_us, event.end_us) for event in compute
        )
        network_intervals = _merge_intervals(
            (event.start_us, event.end_us) for event in network_by_id.values()
        )
        compute_active = _interval_duration(compute_intervals)
        network_active = _interval_duration(network_intervals)
        overlap = _intersection_duration(compute_intervals, network_intervals)
        summaries.append(
            RankSummary(
                rank=rank,
                compute_tasks=len(compute),
                network_tx_tasks=len(tx),
                network_rx_tasks=len(rx),
                declared_compute_us=sum(
                    event.task.declared_duration_us for event in compute
                ),
                actual_compute_sum_us=sum(
                    event.actual_duration_us for event in compute
                ),
                compute_active_us=compute_active,
                network_active_us=network_active,
                compute_network_overlap_us=overlap,
                overlap_fraction_of_compute=(
                    overlap / compute_active if compute_active else 0.0
                ),
                overlap_fraction_of_network=(
                    overlap / network_active if network_active else 0.0
                ),
                tx_bytes=sum(event.task.transfer_bytes for event in tx),
                rx_bytes=sum(event.task.transfer_bytes for event in rx),
            )
        )
    return tuple(summaries)


def _task_color(task: TaskDefinition) -> str:
    if task.kind == "compute":
        return COMPUTE_COLORS.get(task.operation or "compute", COMPUTE_COLORS["compute"])
    return NETWORK_COLORS.get(
        task.payload_kind or "transfer", NETWORK_COLORS["transfer"]
    )


def _format_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:g} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


TASK_DATA_COLUMNS = (
    "task_id",
    "barrier_id",
    "key",
    "kind",
    "rank",
    "src_rank",
    "dst_rank",
    "start_us",
    "end_us",
    "actual_duration_us",
    "declared_compute_us",
    "transfer_bytes",
    "logical_throughput_gbps",
    "operation",
    "payload_kind",
    "communication_phase_id",
    "route_spec",
    "predecessor_count",
    "overlaps_communication",
)


def _json_for_html(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")


def _task_data(events: tuple[TaskEvent, ...]) -> list[list[object]]:
    records: list[list[object]] = []
    for event in events:
        task = event.task
        records.append(
            [
                task.task_id,
                task.barrier_id,
                task.key,
                "c" if task.kind == "compute" else "n",
                task.rank,
                task.src_rank,
                task.dst_rank,
                event.start_us,
                event.end_us,
                event.actual_duration_us,
                task.declared_duration_us,
                task.transfer_bytes,
                event.logical_throughput_gbps,
                task.operation,
                task.payload_kind,
                task.communication_phase_id,
                task.route_spec,
                len(task.predecessors),
                task.overlaps_communication,
            ]
        )
    return records


def _compressed_predecessors(events: tuple[TaskEvent, ...]) -> str:
    task_id_by_key = {event.task.key: event.task.task_id for event in events}
    payload: list[list[object]] = []
    for event in events:
        predecessors = event.task.predecessors
        if not predecessors:
            continue
        references = [
            task_id_by_key.get(predecessor, predecessor)
            for predecessor in predecessors
        ]
        payload.append([event.task.task_id, references])
    compressed = gzip.compress(
        _json_for_html(payload).encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    return base64.b64encode(compressed).decode("ascii")


def _lane_events(
    events: tuple[TaskEvent, ...], rank: int, lane: str
) -> list[TaskEvent]:
    if lane == "Compute":
        return [event for event in events if event.task.rank == rank]
    if lane == "Network TX":
        return [event for event in events if rank in _network_tx_ranks(event.task)]
    if lane == "Network RX":
        return [event for event in events if rank in _network_rx_ranks(event.task)]
    raise ValueError(f"unknown lane: {lane}")


def _network_tx_ranks(task: TaskDefinition) -> set[int]:
    if task.kind != "transfer" or task.src_rank is None or task.dst_rank is None:
        return set()
    match = _SERVER_FORWARD_RE.fullmatch(task.route_spec or "")
    if match is None:
        return {task.src_rank}
    src_relay = int(match.group("src"))
    dst_relay = int(match.group("dst"))
    result = {src_relay}
    if task.src_rank != src_relay:
        result.add(task.src_rank)
    if dst_relay != task.dst_rank:
        result.add(dst_relay)
    return result


def _network_rx_ranks(task: TaskDefinition) -> set[int]:
    if task.kind != "transfer" or task.src_rank is None or task.dst_rank is None:
        return set()
    match = _SERVER_FORWARD_RE.fullmatch(task.route_spec or "")
    if match is None:
        return {task.dst_rank}
    src_relay = int(match.group("src"))
    dst_relay = int(match.group("dst"))
    result = {dst_relay}
    if task.src_rank != src_relay:
        result.add(src_relay)
    if dst_relay != task.dst_rank:
        result.add(task.dst_rank)
    return result


def write_task_csv(path: Path, events: tuple[TaskEvent, ...]) -> None:
    fields = [
        "task_id",
        "barrier_id",
        "key",
        "kind",
        "rank",
        "src_rank",
        "dst_rank",
        "start_us",
        "end_us",
        "actual_duration_us",
        "declared_compute_us",
        "transfer_bytes",
        "logical_throughput_gbps",
        "operation",
        "payload_kind",
        "communication_phase_id",
        "route_spec",
        "predecessors",
        "predecessor_barrier_ids",
        "overlaps_communication",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            task = event.task
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "barrier_id": task.barrier_id,
                    "key": task.key,
                    "kind": task.kind,
                    "rank": "" if task.rank is None else task.rank,
                    "src_rank": "" if task.src_rank is None else task.src_rank,
                    "dst_rank": "" if task.dst_rank is None else task.dst_rank,
                    "start_us": event.start_us,
                    "end_us": event.end_us,
                    "actual_duration_us": event.actual_duration_us,
                    "declared_compute_us": task.declared_duration_us,
                    "transfer_bytes": task.transfer_bytes,
                    "logical_throughput_gbps": (
                        "" if event.logical_throughput_gbps is None
                        else event.logical_throughput_gbps
                    ),
                    "operation": task.operation or "",
                    "payload_kind": task.payload_kind or "",
                    "communication_phase_id": task.communication_phase_id or "",
                    "route_spec": task.route_spec or "",
                    "predecessors": "|".join(task.predecessors),
                    "predecessor_barrier_ids": "|".join(
                        str(item) for item in task.predecessor_barrier_ids
                    ),
                    "overlaps_communication": task.overlaps_communication,
                }
            )


def write_rank_summary(path: Path, summaries: tuple[RankSummary, ...]) -> None:
    fields = list(RankSummary.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def write_html(
    path: Path,
    events: tuple[TaskEvent, ...],
    summaries: tuple[RankSummary, ...],
    selected_ranks: tuple[int, ...],
    *,
    title: str,
    gpus_per_server: int,
    pixels_per_us: float,
) -> None:
    makespan = max(event.end_us for event in events)
    timeline_width = 1400.0
    row_height = 23
    top_height = 35
    rows = [(rank, lane) for rank in selected_ranks for lane in LANES]
    lane_codes = {lane: index for index, lane in enumerate(LANES)}
    svg_height = top_height + len(rows) * row_height + 18
    x_scale = timeline_width / max(makespan, 1e-12)
    svg_parts = [
        f"<svg class='timeline-svg' xmlns='http://www.w3.org/2000/svg' "
        f"width='{timeline_width:.3f}' "
        f"height='{svg_height}' viewBox='0 0 {timeline_width:.3f} {svg_height}'>"
        "<g class='ruler'></g>"
    ]
    for row_index, (rank, lane) in enumerate(rows):
        y = top_height + row_index * row_height
        if rank is not None and (rank // gpus_per_server) % 2:
            svg_parts.append(
                f"<rect class='row-background' x='0' y='{y}' "
                f"width='{timeline_width:.3f}' "
                f"height='{row_height}' fill='#F1F4F7'/>"
            )
        for event in _lane_events(events, rank, lane):
            x = event.start_us * x_scale
            width = max(event.actual_duration_us * x_scale, 1.2)
            svg_parts.append(
                f"<rect class='task' data-task-id='{event.task.task_id}' "
                f"data-lane='{lane_codes[lane]}' "
                f"x='{x:.3f}' y='{y + 3}' width='{width:.3f}' "
                f"height='{row_height - 6}' rx='1.5' fill='{_task_color(event.task)}' "
                "fill-opacity='0.86' stroke='#FFFFFF' stroke-width='0.35'/>"
            )
        if lane == "Network RX":
            svg_parts.append(
                f"<line class='row-separator' x1='0' y1='{y + row_height}' "
                f"x2='{timeline_width:.3f}' "
                f"y2='{y + row_height}' stroke='#9AA5B1' stroke-width='0.8'/>"
            )
    svg_parts.append("</svg>")

    labels = [f"<div class='label-spacer'></div>"]
    for rank, lane in rows:
        labels.append(
            f"<div class='lane-label server-{rank // gpus_per_server % 2}'>"
            f"<strong>GPU {rank:02d}</strong><span>{escape(lane)}</span></div>"
        )

    transfer_bytes = sum(
        event.task.transfer_bytes for event in events if event.task.kind == "transfer"
    )
    compute_count = sum(event.task.kind == "compute" for event in events)
    transfer_count = sum(event.task.kind == "transfer" for event in events)
    cards = (
        ("Makespan", f"{makespan:.6g} us"),
        (
            "Tasks",
            f"{len(events)} ({compute_count} compute / {transfer_count} network)",
        ),
        ("Logical bytes", _format_bytes(transfer_bytes)),
    )
    card_html = "".join(
        f"<div class='card'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in cards
    )
    legend_items = [
        ("Attention", COMPUTE_COLORS["attention"]),
        ("Router", COMPUTE_COLORS["router_projection"]),
        ("Expert", COMPUTE_COLORS["expert_ffn"]),
        ("Reduce", COMPUTE_COLORS["combine_reduce"]),
        ("Dispatch", NETWORK_COLORS["dispatch_hidden"]),
        ("Combine", NETWORK_COLORS["combine_partial"]),
        ("Expert weight", NETWORK_COLORS["expert_weight_rdma"]),
    ]
    legend_html = "".join(
        f"<span><i style='background:{color}'></i>{escape(label)}</span>"
        for label, color in legend_items
    )
    task_data_json = _json_for_html(_task_data(events))
    task_columns_json = _json_for_html(TASK_DATA_COLUMNS)
    predecessor_payload = _compressed_predecessors(events)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
body {{ margin: 0; background: #F6F8FA; color: #17212B; }}
header {{ padding: 20px 24px 14px; background: #FFFFFF; border-bottom: 1px solid #D7DDE5; }}
h1 {{ margin: 0 0 5px; font-size: 22px; font-weight: 650; }}
.sub {{ color: #5E6B78; font-size: 13px; }}
.cards {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px; margin-top: 16px; }}
.card {{ border: 1px solid #D7DDE5; border-radius: 6px; padding: 10px 12px; background: #FBFCFD; }}
.card span {{ display: block; color: #66727F; font-size: 11px; text-transform: uppercase; }}
.card strong {{ display: block; margin-top: 4px; font-size: 15px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 11px 24px; background: #FFFFFF; border-bottom: 1px solid #D7DDE5; font-size: 12px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
.legend i {{ width: 11px; height: 11px; border-radius: 2px; }}
.timeline-toolbar {{ display: flex; align-items: center; gap: 8px; margin: 16px 16px 0; padding: 8px 10px; border: 1px solid #C9D1DA; background: #FFFFFF; }}
.timeline-toolbar button {{ width: 32px; height: 30px; border: 1px solid #B8C1CC; background: #FFFFFF; color: #17212B; font-size: 17px; cursor: pointer; }}
.timeline-toolbar button.fit {{ width: auto; padding: 0 11px; font-size: 12px; }}
.timeline-toolbar input {{ width: min(340px, 40vw); }}
.scale-readout {{ margin-left: auto; color: #4F5D6B; font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; }}
.timeline-shell {{ display: grid; grid-template-columns: 152px minmax(0, 1fr); margin: 16px; border: 1px solid #C9D1DA; background: #FFFFFF; }}
.labels {{ border-right: 1px solid #C9D1DA; }}
.label-spacer {{ height: {top_height}px; border-bottom: 1px solid #D7DDE5; }}
.lane-label {{ height: {row_height}px; box-sizing: border-box; padding: 3px 8px; display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #E6EAF0; font-size: 11px; }}
.lane-label.server-1 {{ background: #F1F4F7; }}
.lane-label strong {{ font-size: 11px; }}
.lane-label span {{ color: #697582; font-size: 10px; }}
.viewport {{ overflow: auto; }}
.task:hover {{ fill-opacity: 1; stroke: #111827; stroke-width: 1; }}
.task.selected {{ fill-opacity: 1; stroke: #111827; stroke-width: 1.4; }}
.hover-tooltip {{ position: fixed; z-index: 20; max-width: 440px; padding: 8px 10px; border: 1px solid #9DA8B5; background: #FFFFFF; box-shadow: 0 4px 14px rgba(23, 33, 43, 0.16); pointer-events: none; font-size: 11px; line-height: 1.45; }}
.hover-tooltip strong {{ display: block; margin-bottom: 4px; overflow-wrap: anywhere; }}
.hover-tooltip span {{ display: block; color: #4F5D6B; }}
.task-inspector {{ margin: 0 16px 16px; padding: 12px 14px; border: 1px solid #C9D1DA; background: #FFFFFF; }}
.inspector-head {{ display: flex; align-items: flex-start; gap: 10px; }}
.inspector-head strong {{ overflow-wrap: anywhere; }}
.inspector-head button {{ margin-left: auto; width: 28px; height: 28px; border: 1px solid #B8C1CC; background: #FFFFFF; cursor: pointer; }}
.inspector-grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 8px 16px; margin-top: 10px; font-size: 11px; }}
.inspector-grid div {{ min-width: 0; }}
.inspector-grid span {{ display: block; color: #697582; font-size: 10px; text-transform: uppercase; }}
.inspector-grid strong {{ display: block; margin-top: 2px; overflow-wrap: anywhere; }}
.predecessor-details {{ margin-top: 10px; border-top: 1px solid #D7DDE5; }}
.predecessor-details summary {{ padding: 9px 0 0; cursor: pointer; font-size: 12px; font-weight: 650; }}
.predecessor-list {{ max-height: 260px; overflow: auto; margin-top: 8px; font: 11px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }}
.predecessor-list ol {{ margin: 0; padding-left: 26px; }}
.predecessor-list li {{ overflow-wrap: anywhere; }}
.details {{ margin: 16px; background: #FFFFFF; border: 1px solid #C9D1DA; }}
.details summary {{ padding: 12px 14px; cursor: pointer; font-size: 15px; font-weight: 650; }}
.details-scroll {{ overflow: auto; max-height: 520px; border-top: 1px solid #D7DDE5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
th {{ position: sticky; top: 0; background: #EEF2F6; text-align: left; z-index: 1; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #E4E8ED; white-space: nowrap; }}
td.key {{ max-width: 420px; overflow: hidden; text-overflow: ellipsis; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} .timeline-shell {{ grid-template-columns: 130px minmax(0, 1fr); }} .inspector-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1><div class="sub">HTSim task start/done intervals, logical bytes, FCT and compute/network overlap.</div><div class="cards">{card_html}</div></header>
<div class="legend">{legend_html}</div>
<div class="timeline-panel" data-makespan-us="{makespan:.12g}" data-svg-height="{svg_height}" data-top-height="{top_height}" data-preferred-px-us="{pixels_per_us:.12g}">
<div class="timeline-toolbar"><button type="button" class="zoom-out" title="Zoom out">−</button><input class="zoom-slider" type="range" min="0" max="8" step="0.25" value="0" aria-label="Timeline zoom"><button type="button" class="zoom-in" title="Zoom in">+</button><button type="button" class="fit" title="Fit complete timeline">Fit</button><output class="scale-readout"></output></div>
<section class="timeline-shell"><div class="labels">{''.join(labels)}</div><div class="viewport">{''.join(svg_parts)}</div></section>
</div>
<div class="hover-tooltip" hidden></div>
<section class="task-inspector" hidden><div class="inspector-head"><strong></strong><button type="button" title="Close task inspector">X</button></div><div class="inspector-grid"></div><details class="predecessor-details"><summary></summary><div class="predecessor-list"></div></details></section>
<details class="details"><summary>Task details ({len(events)})</summary><div class="details-scroll"><table><thead><tr><th>ID</th><th>Kind</th><th>Task</th><th>Endpoint</th><th>Category</th><th>Start us</th><th>End us</th><th>FCT us</th><th>Bytes</th><th>Logical Gbps</th></tr></thead><tbody></tbody></table></div></details>
<script type="application/json" id="task-columns">{task_columns_json}</script>
<script type="application/json" id="task-data">{task_data_json}</script>
<script type="text/plain" id="predecessor-data">{predecessor_payload}</script>
<script>
(() => {{
  const panel = document.querySelector('.timeline-panel');
  const viewport = panel.querySelector('.viewport');
  const svg = panel.querySelector('.timeline-svg');
  const slider = panel.querySelector('.zoom-slider');
  const readout = panel.querySelector('.scale-readout');
  const makespan = Number(panel.dataset.makespanUs);
  const svgHeight = Number(panel.dataset.svgHeight);
  const topHeight = Number(panel.dataset.topHeight);
  const columns = JSON.parse(document.getElementById('task-columns').textContent);
  const I = Object.fromEntries(columns.map((name, index) => [name, index]));
  const taskRows = JSON.parse(document.getElementById('task-data').textContent);
  const tasks = new Map(taskRows.map(row => [row[I.task_id], row]));
  const laneNames = {_json_for_html(LANES)};
  const hoverTooltip = document.querySelector('.hover-tooltip');
  const inspector = document.querySelector('.task-inspector');
  const inspectorTitle = inspector.querySelector('.inspector-head strong');
  const inspectorGrid = inspector.querySelector('.inspector-grid');
  const predecessorDetails = inspector.querySelector('.predecessor-details');
  const predecessorSummary = predecessorDetails.querySelector('summary');
  const predecessorList = predecessorDetails.querySelector('.predecessor-list');
  const taskDetails = document.querySelector('.details');
  let scale = 1;
  let selectedBar = null;
  let predecessorMapPromise = null;

  function escapeHtml(value) {{
    return String(value ?? '').replace(/[&<>"']/g, character => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }})[character]);
  }}

  function number(value) {{
    return value === null ? '' : String(Number(Number(value).toPrecision(7)));
  }}

  function kind(row) {{
    return row[I.kind] === 'c' ? 'compute' : 'transfer';
  }}

  function endpoint(row) {{
    return row[I.kind] === 'c'
      ? `GPU ${{row[I.rank]}}`
      : `GPU ${{row[I.src_rank]}} -> GPU ${{row[I.dst_rank]}}`;
  }}

  function category(row) {{
    return row[I.kind] === 'c' ? row[I.operation] : row[I.payload_kind];
  }}

  function formatBytes(value) {{
    let amount = Number(value);
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let unit = 0;
    while (amount >= 1024 && unit < units.length - 1) {{ amount /= 1024; unit += 1; }}
    return `${{Number(amount.toPrecision(6))}} ${{units[unit]}}`;
  }}

  function summaryHtml(row, lane) {{
    const lines = [
      `<strong>Task ${{row[I.task_id]}}: ${{escapeHtml(row[I.key])}}</strong>`,
      `<span>${{escapeHtml(lane)}} · ${{number(row[I.start_us])}}-${{number(row[I.end_us])}} us · FCT ${{number(row[I.actual_duration_us])}} us</span>`,
      `<span>${{escapeHtml(endpoint(row))}} · ${{escapeHtml(category(row) || kind(row))}} · predecessors ${{row[I.predecessor_count]}}</span>`,
    ];
    if (row[I.kind] === 'n') lines.push(`<span>${{formatBytes(row[I.transfer_bytes])}} · ${{number(row[I.logical_throughput_gbps])}} Gbps</span>`);
    return lines.join('');
  }}

  function inspectorFields(row, lane) {{
    const fields = [
      ['Lane', lane],
      ['Kind', kind(row)],
      ['Endpoint', endpoint(row)],
      ['Category', category(row) || ''],
      ['Start', `${{number(row[I.start_us])}} us`],
      ['End', `${{number(row[I.end_us])}} us`],
      ['FCT', `${{number(row[I.actual_duration_us])}} us`],
      ['Barrier', row[I.barrier_id]],
    ];
    if (row[I.kind] === 'c') {{
      fields.push(['Declared compute', `${{number(row[I.declared_compute_us])}} us`]);
      fields.push(['Communication overlap', row[I.overlaps_communication]]);
    }} else {{
      fields.push(['Bytes', `${{row[I.transfer_bytes]}} (${{formatBytes(row[I.transfer_bytes])}})`]);
      fields.push(['Logical throughput', `${{number(row[I.logical_throughput_gbps])}} Gbps`]);
      fields.push(['Route', row[I.route_spec] || 'dynamic']);
      fields.push(['Phase', row[I.communication_phase_id] || '']);
    }}
    return fields.map(([label, value]) => `<div><span>${{escapeHtml(label)}}</span><strong>${{escapeHtml(value)}}</strong></div>`).join('');
  }}

  async function loadPredecessorMap() {{
    if (predecessorMapPromise) return predecessorMapPromise;
    predecessorMapPromise = (async () => {{
      if (!('DecompressionStream' in window)) throw new Error('Browser does not support gzip decompression');
      const encoded = document.getElementById('predecessor-data').textContent.trim();
      const binary = atob(encoded);
      const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const payload = await new Response(stream).json();
      return new Map(payload);
    }})();
    return predecessorMapPromise;
  }}

  async function renderPredecessors(taskId) {{
    const expected = String(taskId);
    predecessorList.textContent = 'Loading...';
    try {{
      const predecessorMap = await loadPredecessorMap();
      if (inspector.dataset.taskId !== expected) return;
      const references = predecessorMap.get(taskId) || [];
      if (!references.length) {{ predecessorList.textContent = 'None'; return; }}
      const names = references.map(reference =>
        typeof reference === 'number' && tasks.has(reference)
          ? tasks.get(reference)[I.key]
          : String(reference)
      );
      predecessorList.innerHTML = `<ol>${{names.map(name => `<li>${{escapeHtml(name)}}</li>`).join('')}}</ol>`;
    }} catch (error) {{
      predecessorList.textContent = String(error);
    }}
  }}

  function selectTask(bar) {{
    if (selectedBar) selectedBar.classList.remove('selected');
    selectedBar = bar;
    selectedBar.classList.add('selected');
    const row = tasks.get(Number(bar.dataset.taskId));
    const lane = laneNames[Number(bar.dataset.lane)];
    inspector.dataset.taskId = String(row[I.task_id]);
    inspectorTitle.textContent = `Task ${{row[I.task_id]}}: ${{row[I.key]}}`;
    inspectorGrid.innerHTML = inspectorFields(row, lane);
    predecessorSummary.textContent = `Predecessors (${{row[I.predecessor_count]}})`;
    predecessorDetails.open = false;
    predecessorList.textContent = '';
    inspector.hidden = false;
  }}

  function renderTaskTable() {{
    if (taskDetails.dataset.rendered) return;
    taskDetails.dataset.rendered = 'true';
    taskDetails.querySelector('tbody').innerHTML = taskRows.map(row => `
      <tr><td>${{row[I.task_id]}}</td><td>${{kind(row)}}</td><td class="key">${{escapeHtml(row[I.key])}}</td>
      <td>${{escapeHtml(endpoint(row))}}</td><td>${{escapeHtml(category(row) || '')}}</td>
      <td>${{number(row[I.start_us])}}</td><td>${{number(row[I.end_us])}}</td><td>${{number(row[I.actual_duration_us])}}</td>
      <td>${{row[I.transfer_bytes] || ''}}</td><td>${{number(row[I.logical_throughput_gbps])}}</td></tr>`).join('');
  }}

  function niceStep(raw) {{
    const power = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-12))));
    const value = raw / power;
    const nice = value <= 1 ? 1 : value <= 2 ? 2 : value <= 5 ? 5 : 10;
    return nice * power;
  }}

  function drawRuler(width) {{
    const ruler = svg.querySelector('.ruler');
    const step = niceStep(120 / scale);
    const parts = [];
    for (let time = 0; time <= makespan + step * 0.001; time += step) {{
      const x = time * scale;
      parts.push(`<line x1="${{x}}" y1="${{topHeight}}" x2="${{x}}" y2="${{svgHeight}}" stroke="#C6CDD6" stroke-width="0.7"/><text x="${{x + 3}}" y="18" fill="#4B5563" font-size="11">${{Number(time.toPrecision(7))}} us</text>`);
    }}
    ruler.innerHTML = parts.join('');
  }}

  function applyZoom(level, preserveCenter = true) {{
    const oldScale = scale;
    const centerTime = (viewport.scrollLeft + viewport.clientWidth / 2) / oldScale;
    const fitScale = viewport.clientWidth / Math.max(makespan, 1e-12);
    scale = fitScale * Math.pow(2, level);
    const width = Math.max(viewport.clientWidth, makespan * scale);
    svg.setAttribute('width', String(width));
    svg.setAttribute('viewBox', `0 0 ${{width}} ${{svgHeight}}`);
    svg.querySelectorAll('.row-background').forEach(item => item.setAttribute('width', String(width)));
    svg.querySelectorAll('.row-separator').forEach(item => item.setAttribute('x2', String(width)));
    svg.querySelectorAll('.task').forEach(item => {{
      const row = tasks.get(Number(item.dataset.taskId));
      item.setAttribute('x', String(row[I.start_us] * scale));
      item.setAttribute('width', String(Math.max(row[I.actual_duration_us] * scale, 1.2)));
    }});
    drawRuler(width);
    readout.value = `100 px = ${{(100 / scale).toPrecision(6)}} us · visible ${{(viewport.clientWidth / scale).toPrecision(6)}} us · total ${{makespan.toPrecision(7)}} us`;
    if (preserveCenter && level > 0) {{
      viewport.scrollLeft = Math.max(0, centerTime * scale - viewport.clientWidth / 2);
    }} else {{
      viewport.scrollLeft = 0;
    }}
  }}

  slider.addEventListener('input', () => applyZoom(Number(slider.value)));
  panel.querySelector('.zoom-out').addEventListener('click', () => {{
    slider.value = String(Math.max(Number(slider.min), Number(slider.value) - 1));
    applyZoom(Number(slider.value));
  }});
  panel.querySelector('.zoom-in').addEventListener('click', () => {{
    slider.value = String(Math.min(Number(slider.max), Number(slider.value) + 1));
    applyZoom(Number(slider.value));
  }});
  panel.querySelector('.fit').addEventListener('click', () => {{
    slider.value = '0';
    applyZoom(0, false);
  }});
  viewport.addEventListener('wheel', event => {{
    if (!event.ctrlKey) return;
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.25 : -0.25;
    slider.value = String(Math.max(Number(slider.min), Math.min(Number(slider.max), Number(slider.value) + delta)));
    applyZoom(Number(slider.value));
  }}, {{ passive: false }});
  svg.addEventListener('pointermove', event => {{
    const bar = event.target.closest && event.target.closest('.task');
    if (!bar) {{ hoverTooltip.hidden = true; return; }}
    const row = tasks.get(Number(bar.dataset.taskId));
    hoverTooltip.innerHTML = summaryHtml(row, laneNames[Number(bar.dataset.lane)]);
    hoverTooltip.hidden = false;
    const margin = 12;
    const bounds = hoverTooltip.getBoundingClientRect();
    hoverTooltip.style.left = `${{Math.min(event.clientX + margin, window.innerWidth - bounds.width - margin)}}px`;
    hoverTooltip.style.top = `${{Math.min(event.clientY + margin, window.innerHeight - bounds.height - margin)}}px`;
  }});
  svg.addEventListener('pointerleave', () => {{ hoverTooltip.hidden = true; }});
  svg.addEventListener('click', event => {{
    const bar = event.target.closest && event.target.closest('.task');
    if (bar) selectTask(bar);
  }});
  inspector.querySelector('.inspector-head button').addEventListener('click', () => {{
    inspector.hidden = true;
    if (selectedBar) selectedBar.classList.remove('selected');
    selectedBar = null;
  }});
  predecessorDetails.addEventListener('toggle', () => {{
    if (predecessorDetails.open) renderPredecessors(Number(inspector.dataset.taskId));
  }});
  taskDetails.addEventListener('toggle', () => {{
    if (taskDetails.open) renderTaskTable();
  }});
  new ResizeObserver(() => applyZoom(Number(slider.value), Number(slider.value) > 0)).observe(viewport);
  applyZoom(0, false);
}})();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_overview_json(
    path: Path,
    events: tuple[TaskEvent, ...],
    summaries: tuple[RankSummary, ...],
    selected_ranks: tuple[int, ...],
    *,
    workload_dir: Path,
    log_path: Path,
) -> None:
    compute_events = [event for event in events if event.task.kind == "compute"]
    expert_weight_events = [
        event
        for event in events
        if event.task.payload_kind in EXPERT_WEIGHT_PAYLOADS
    ]
    network_events = [event for event in events if event.task.kind == "transfer"]
    payload_bytes: dict[str, int] = {}
    for event in network_events:
        key = event.task.payload_kind or "transfer"
        payload_bytes[key] = payload_bytes.get(key, 0) + event.task.transfer_bytes
    payload_fct: dict[str, float] = {}
    for event in network_events:
        key = event.task.payload_kind or "transfer"
        payload_fct[key] = payload_fct.get(key, 0.0) + event.actual_duration_us
    selected = [summaries[rank] for rank in selected_ranks]
    payload = {
        "format_version": 1,
        "source": {
            "workload_dir": str(workload_dir),
            "htsim_log": str(log_path),
        },
        "makespan_us": max(event.end_us for event in events),
        "task_count": len(events),
        "compute_task_count": len(compute_events),
        "expert_weight_task_count": len(expert_weight_events),
        "expert_weight_bytes": sum(
            event.task.transfer_bytes for event in expert_weight_events
        ),
        "expert_weight_active_us": _interval_duration(
            _merge_intervals(
                (event.start_us, event.end_us)
                for event in expert_weight_events
            )
        ),
        "expert_weight_fraction_of_makespan": (
            _interval_duration(
                _merge_intervals(
                    (event.start_us, event.end_us)
                    for event in expert_weight_events
                )
            )
            / max(event.end_us for event in events)
            if expert_weight_events
            else 0.0
        ),
        "network_task_count": len(network_events),
        "logical_transfer_bytes": sum(
            event.task.transfer_bytes for event in network_events
        ),
        "logical_transfer_bytes_by_payload": dict(sorted(payload_bytes.items())),
        "network_task_fct_sum_us_by_payload": dict(sorted(payload_fct.items())),
        "selected_ranks": selected_ranks,
        "selected_rank_compute_active_sum_us": sum(
            item.compute_active_us for item in selected
        ),
        "selected_rank_network_active_sum_us": sum(
            item.network_active_us for item in selected
        ),
        "selected_rank_compute_network_overlap_sum_us": sum(
            item.compute_network_overlap_us for item in selected
        ),
        "semantics": {
            "task_interval": "DAG_TASK_START to DAG_TASK_DONE",
            "network_duration": "logical HTSim task FCT including server_forward phases",
            "logical_throughput": "transfer_bytes divided by logical task FCT",
            "rank_network_activity": "union of logical transfers touching rank as src or dst",
            "overlap": "intersection of per-rank compute and network interval unions",
            "resource_model_warning": (
                "timeline shows DAG concurrency; HTSim does not dynamically schedule GPU SMs"
            ),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.json and task_map.json.",
    )
    parser.add_argument("--htsim-log", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <htsim-log-dir>/dag_timeline.",
    )
    parser.add_argument("--title", default="DAG Compute and Communication Timeline")
    parser.add_argument("--gpus-per-server", type=int, default=8)
    parser.add_argument(
        "--ranks",
        help="Optional comma-separated ranks/ranges, for example 0-7,16,24-31.",
    )
    parser.add_argument(
        "--pixels-per-us",
        type=float,
        default=16.0,
        help="Interactive HTML horizontal scale; the minimum canvas is 1400px.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpus_per_server <= 0:
        raise SystemExit("--gpus-per-server must be positive")
    if not math.isfinite(args.pixels_per_us) or args.pixels_per_us <= 0:
        raise SystemExit("--pixels-per-us must be positive and finite")

    workload_dir = args.workload_dir.resolve()
    log_path = args.htsim_log.resolve()
    output_dir = (args.output_dir or log_path.parent / "dag_timeline").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = read_manifest(workload_dir / "manifest.json")
        num_ranks = int(manifest["num_ranks"])
        tasks = read_task_map(workload_dir / "task_map.json", num_ranks)
        if len(tasks) != manifest["task_count"]:
            raise ValueError(
                "manifest/task-map mismatch: "
                f"manifest={manifest['task_count']}, task_map={len(tasks)}"
            )
        events = read_execution_log(log_path, tasks)
        selected_ranks = parse_rank_selection(args.ranks, num_ranks)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid DAG timeline input: {exc}") from exc

    summaries = summarize_ranks(events, num_ranks)
    html_path = output_dir / "dag_gpu_timeline.html"
    write_html(
        html_path,
        events,
        summaries,
        selected_ranks,
        title=args.title,
        gpus_per_server=args.gpus_per_server,
        pixels_per_us=args.pixels_per_us,
    )
    write_task_csv(output_dir / "dag_task_timeline.csv", events)
    write_rank_summary(output_dir / "dag_rank_overlap_summary.csv", summaries)
    write_overview_json(
        output_dir / "dag_timeline_summary.json",
        events,
        summaries,
        selected_ranks,
        workload_dir=workload_dir,
        log_path=log_path,
    )
    print(
        f"parsed {len(events)} tasks across {num_ranks} ranks; "
        f"makespan {max(event.end_us for event in events):.9g} us"
    )
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
