from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any, Literal

from .cost import ComputeEstimate
from .schema import ValidationError


TaskKind = Literal["compute", "transfer"]


@dataclass
class Task:
    key: str
    kind: TaskKind
    predecessors: set[str] = field(default_factory=set)
    barrier_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    operation_flops: int = 0
    duration_us: float = 0.0
    overlaps_communication: bool = False
    available_sms: int | None = None
    peak_flops_per_second: float | None = None
    src_rank: int | None = None
    dst_rank: int | None = None
    transfer_bytes: int = 0
    payload_kind: str | None = None
    route_spec: str | None = None
    communication_phase_id: str | None = None
    chunk_id: int | None = None

    @property
    def barrier_key(self) -> str:
        return self.barrier_group or f"task:{self.key}"


class TaskGraph:
    def __init__(self, name: str, num_ranks: int) -> None:
        if not name:
            raise ValidationError("graph name must not be empty")
        if num_ranks <= 0:
            raise ValidationError("num_ranks must be positive")
        self.name = name
        self.num_ranks = num_ranks
        self._tasks: dict[str, Task] = {}

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks.values())

    def task(self, key: str) -> Task:
        try:
            return self._tasks[key]
        except KeyError as exc:
            raise ValidationError(f"unknown task: {key}") from exc

    def add_compute(
        self,
        key: str,
        rank: int,
        estimate: ComputeEstimate,
        *,
        predecessors: set[str] | None = None,
        barrier_group: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        self._validate_new_key(key)
        self._validate_rank(rank)
        task = Task(
            key=key,
            kind="compute",
            rank=rank,
            operation_flops=estimate.operation_flops,
            duration_us=estimate.duration_us,
            overlaps_communication=estimate.overlaps_communication,
            available_sms=estimate.available_sms,
            peak_flops_per_second=estimate.peak_flops_per_second,
            predecessors=set(predecessors or ()),
            barrier_group=barrier_group,
            metadata={
                "cost_source": estimate.source,
                "compute_token_count": estimate.token_count,
                "compute_us_per_token": estimate.us_per_token,
                "compute_token_kind": estimate.token_kind,
                **(metadata or {}),
            },
        )
        self._tasks[key] = task
        return task

    def add_transfer(
        self,
        key: str,
        src_rank: int,
        dst_rank: int,
        transfer_bytes: int,
        payload_kind: str,
        communication_phase_id: str,
        *,
        predecessors: set[str] | None = None,
        route_spec: str | None = None,
        chunk_id: int | None = None,
        barrier_group: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        self._validate_new_key(key)
        self._validate_rank(src_rank)
        self._validate_rank(dst_rank)
        if src_rank == dst_rank:
            raise ValidationError("transfer task requires src_rank != dst_rank")
        if transfer_bytes <= 0:
            raise ValidationError("transfer_bytes must be positive")
        if not payload_kind:
            raise ValidationError("payload_kind must not be empty")
        if not communication_phase_id:
            raise ValidationError("communication_phase_id must not be empty")
        if route_spec is not None and "|" in route_spec:
            raise ValidationError("route_spec must not contain '|'")
        if chunk_id is not None and chunk_id < 0:
            raise ValidationError("chunk_id must be non-negative")
        task = Task(
            key=key,
            kind="transfer",
            src_rank=src_rank,
            dst_rank=dst_rank,
            transfer_bytes=transfer_bytes,
            payload_kind=payload_kind,
            route_spec=route_spec,
            communication_phase_id=communication_phase_id,
            chunk_id=chunk_id,
            predecessors=set(predecessors or ()),
            barrier_group=barrier_group,
            metadata=dict(metadata or {}),
        )
        self._tasks[key] = task
        return task

    def add_dependency(self, task_key: str, predecessor_key: str) -> None:
        self.task(task_key).predecessors.add(predecessor_key)

    def validate(self) -> None:
        for task in self._tasks.values():
            missing = task.predecessors - self._tasks.keys()
            if missing:
                raise ValidationError(
                    f"task {task.key} has missing predecessors: {sorted(missing)}"
                )
            if task.key in task.predecessors:
                raise ValidationError(f"task {task.key} depends on itself")
        self._validate_barrier_groups()
        self.topological_keys()

    def topological_keys(self) -> tuple[str, ...]:
        order = {key: index for index, key in enumerate(self._tasks)}
        indegree = {key: len(task.predecessors) for key, task in self._tasks.items()}
        successors: dict[str, list[str]] = {key: [] for key in self._tasks}
        for key, task in self._tasks.items():
            for predecessor in task.predecessors:
                if predecessor in successors:
                    successors[predecessor].append(key)

        ready = [(order[key], key) for key, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        result: list[str] = []
        while ready:
            _, key = heapq.heappop(ready)
            result.append(key)
            for successor in successors[key]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, (order[successor], successor))
        if len(result) != len(self._tasks):
            raise ValidationError("task graph contains a cycle")
        return tuple(result)

    def terminal_keys(self) -> set[str]:
        referenced = {
            predecessor
            for task in self._tasks.values()
            for predecessor in task.predecessors
        }
        return set(self._tasks) - referenced

    def _validate_new_key(self, key: str) -> None:
        if not key:
            raise ValidationError("task key must not be empty")
        if key in self._tasks:
            raise ValidationError(f"duplicate task key: {key}")

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.num_ranks:
            raise ValidationError(
                f"rank {rank} outside configured range [0, {self.num_ranks})"
            )

    def _validate_barrier_groups(self) -> None:
        grouped: dict[str, list[Task]] = {}
        for task in self._tasks.values():
            grouped.setdefault(task.barrier_key, []).append(task)
        for barrier_key, tasks in grouped.items():
            expected = tasks[0].predecessors
            for task in tasks[1:]:
                if task.predecessors != expected:
                    raise ValidationError(
                        f"barrier group {barrier_key} contains tasks with "
                        "different predecessors"
                    )
            task_keys = {task.key for task in tasks}
            if expected & task_keys:
                raise ValidationError(
                    f"barrier group {barrier_key} contains an internal dependency"
                )
