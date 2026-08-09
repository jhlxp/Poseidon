from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .graph import Task, TaskGraph
from .schema import ValidationError


@dataclass(frozen=True)
class DynamicDagBatch:
    batch_id: str
    layer: int
    task_keys: tuple[str, ...]
    observation_barriers: dict[int, tuple[int, ...]]
    protocol: str


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


def _duration_us(value: float) -> str:
    rendered = f"{value:.9f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _execution_signature(task: Task) -> tuple[object, ...]:
    return (
        task.kind,
        tuple(sorted(task.predecessors)),
        task.barrier_key,
        task.rank,
        task.duration_us,
        task.src_rank,
        task.dst_rank,
        task.transfer_bytes,
        task.payload_kind,
        task.route_spec,
        task.communication_phase_id,
        task.chunk_id,
    )


class DynamicDagEmitter:
    """Assign immutable IDs and emit append batches from rebuilt full graphs."""

    def __init__(self, output_dir: Path | str, num_ranks: int) -> None:
        if num_ranks <= 0:
            raise ValidationError("num_ranks must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_ranks = num_ranks
        self.task_ids: dict[str, int] = {}
        self.barrier_ids: dict[str, int] = {}
        self._signatures: dict[str, tuple[object, ...]] = {}
        self._task_records: list[dict[str, Any]] = []
        self._dag_lines = [
            "# Dynamic DAG cumulative snapshot.",
            "# task_id barrier_id | src_rank dst_rank | transfer_bytes compute_us "
            "| predecessor_barriers [| route_spec]",
        ]
        self._batches: list[dict[str, Any]] = []
        self._batch_ids: set[str] = set()
        self._observation_ids: set[int] = set()
        self._next_task_id = 1
        self._next_barrier_id = 0
        (self.output_dir / "nodes.cm").write_text(
            f"Nodes {num_ranks}\nConnections 0\n", encoding="utf-8"
        )

    @property
    def committed_task_keys(self) -> frozenset[str]:
        return frozenset(self.task_ids)

    def assert_committed_unchanged(self, graph: TaskGraph) -> None:
        for key, expected in self._signatures.items():
            try:
                current = graph.task(key)
            except ValidationError as exc:
                raise ValidationError(
                    f"rebuilt graph lost committed task {key}"
                ) from exc
            if _execution_signature(current) != expected:
                raise ValidationError(
                    f"rebuilt graph changed already committed task {key}"
                )

    def append_layer(
        self,
        graph: TaskGraph,
        layer: int,
        observations: dict[int, tuple[str, ...]],
    ) -> DynamicDagBatch:
        selected = tuple(
            key
            for key in graph.topological_keys()
            if graph.task(key).metadata.get("layer") == layer
            and key not in self.task_ids
        )
        return self.append_tasks(
            graph,
            batch_id=f"layer{layer}",
            layer=layer,
            task_keys=selected,
            observations=observations,
        )

    def append_tasks(
        self,
        graph: TaskGraph,
        *,
        batch_id: str,
        layer: int,
        task_keys: tuple[str, ...],
        observations: dict[int, tuple[str, ...]],
    ) -> DynamicDagBatch:
        if (
            not batch_id
            or any(character.isspace() for character in batch_id)
            or "|" in batch_id
        ):
            raise ValidationError("dynamic batch ID must be a non-empty token")
        if batch_id in self._batch_ids:
            raise ValidationError(f"dynamic batch ID is not unique: {batch_id}")
        if graph.num_ranks != self.num_ranks:
            raise ValidationError("dynamic graph rank count changed")
        graph.validate()
        self.assert_committed_unchanged(graph)
        requested = set(task_keys)
        if len(requested) != len(task_keys):
            raise ValidationError(f"dynamic batch {batch_id} repeats task keys")
        selected = tuple(
            key for key in graph.topological_keys() if key in requested
        )
        if len(selected) != len(task_keys):
            missing = requested - set(selected)
            raise ValidationError(
                f"dynamic batch {batch_id} references unknown tasks: {sorted(missing)}"
            )
        already_committed = requested & self.task_ids.keys()
        if already_committed:
            raise ValidationError(
                f"dynamic batch {batch_id} repeats committed tasks: "
                f"{sorted(already_committed)}"
            )
        if not selected:
            raise ValidationError(f"dynamic batch {batch_id} contains no new tasks")
        selected_set = set(selected)
        for key in selected:
            missing = graph.task(key).predecessors - (
                self.task_ids.keys() | selected_set
            )
            if missing:
                raise ValidationError(
                    f"layer {layer} task {key} depends on uncommitted future tasks: "
                    f"{sorted(missing)}"
                )

        barrier_members: dict[str, set[str]] = {}
        for task in graph.tasks:
            barrier_members.setdefault(task.barrier_key, set()).add(task.key)
        for key in selected:
            members = barrier_members[graph.task(key).barrier_key]
            if not members <= selected_set:
                raise ValidationError(
                    f"barrier {graph.task(key).barrier_key} crosses append batches"
                )

        new_observation_ids = set(observations)
        duplicate_observations = new_observation_ids & self._observation_ids
        if duplicate_observations:
            raise ValidationError(
                "dynamic observation IDs are not globally unique: "
                f"{sorted(duplicate_observations)}"
            )
        for observation_id, terminal_keys in observations.items():
            if observation_id < 0 or observation_id > 0xFFFFFFFF:
                raise ValidationError(
                    f"observation ID is outside uint32 range: {observation_id}"
                )
            if not terminal_keys:
                raise ValidationError(
                    f"observation {observation_id} has no terminal tasks"
                )
            if len(set(terminal_keys)) != len(terminal_keys):
                raise ValidationError(
                    f"observation {observation_id} repeats terminal tasks"
                )
            unknown = set(terminal_keys) - (self.task_ids.keys() | selected_set)
            if unknown:
                raise ValidationError(
                    f"observation {observation_id} references uncommitted tasks: "
                    f"{sorted(unknown)}"
                )
            if not (set(terminal_keys) & selected_set):
                raise ValidationError(
                    f"observation {observation_id} has no terminal in the new batch"
                )

        assigned_task_ids: dict[str, int] = {}
        next_task_id = self._next_task_id
        for key in selected:
            assigned_task_ids[key] = next_task_id
            next_task_id += 1
        assigned_barrier_ids: dict[str, int] = {}
        next_barrier_id = self._next_barrier_id
        for key in selected:
            barrier_key = graph.task(key).barrier_key
            if (
                barrier_key not in self.barrier_ids
                and barrier_key not in assigned_barrier_ids
            ):
                assigned_barrier_ids[barrier_key] = next_barrier_id
                next_barrier_id += 1

        all_barrier_ids = {**self.barrier_ids, **assigned_barrier_ids}

        protocol_lines = [f"DAG_APPEND_BEGIN {batch_id}"]
        new_dag_lines: list[str] = []
        new_task_records: list[dict[str, Any]] = []
        new_signatures: dict[str, tuple[object, ...]] = {}
        for key in selected:
            task = graph.task(key)
            predecessor_barriers = sorted({
                all_barrier_ids[graph.task(predecessor).barrier_key]
                for predecessor in task.predecessors
            })
            predecessors = (
                " ".join(str(value) for value in predecessor_barriers)
                if predecessor_barriers else "-"
            )
            barrier_id = all_barrier_ids[task.barrier_key]
            if task.kind == "compute":
                assert task.rank is not None
                endpoints = f"{task.rank} {task.rank}"
                operation = f"0 {_duration_us(task.duration_us)}"
            else:
                assert task.src_rank is not None and task.dst_rank is not None
                endpoints = f"{task.src_rank} {task.dst_rank}"
                operation = f"{task.transfer_bytes} 0"
            dag_line = (
                f"{assigned_task_ids[key]} {barrier_id} | {endpoints} | "
                f"{operation} | {predecessors}"
            )
            if task.route_spec:
                dag_line += f" | {task.route_spec}"
            new_dag_lines.append(dag_line)
            protocol_lines.append(f"DAG_TASK {dag_line}")

            record = asdict(task)
            record.update({
                "task_id": assigned_task_ids[key],
                "barrier_id": barrier_id,
                "predecessor_barrier_ids": predecessor_barriers,
                "logical_stream": task.metadata.get(
                    "logical_stream",
                    "compute" if task.kind == "compute" else "communication",
                ),
                "dynamic_batch_id": batch_id,
            })
            record["predecessors"] = sorted(task.predecessors)
            new_task_records.append(record)
            new_signatures[key] = _execution_signature(task)

        observation_barriers: dict[int, tuple[int, ...]] = {}
        for observation_id, task_keys in sorted(observations.items()):
            barriers = tuple(sorted({
                all_barrier_ids[graph.task(key).barrier_key]
                for key in task_keys
            }))
            observation_barriers[observation_id] = barriers
            protocol_lines.append(
                f"DAG_OBSERVE {observation_id} | "
                + " ".join(str(value) for value in barriers)
            )
        protocol_lines.append(f"DAG_APPEND_END {batch_id}")
        protocol = "\n".join(protocol_lines) + "\n"
        self.task_ids.update(assigned_task_ids)
        self.barrier_ids.update(assigned_barrier_ids)
        self._signatures.update(new_signatures)
        self._task_records.extend(new_task_records)
        self._dag_lines.extend(new_dag_lines)
        self._next_task_id = next_task_id
        self._next_barrier_id = next_barrier_id
        self._batch_ids.add(batch_id)
        self._observation_ids.update(new_observation_ids)
        self._batches.append({
            "batch_id": batch_id,
            "layer": layer,
            "task_count": len(selected),
            "task_ids": [assigned_task_ids[key] for key in selected],
            "observations": {
                str(key): list(value)
                for key, value in observation_barriers.items()
            },
        })
        self._write_snapshots(include_task_map=False)
        return DynamicDagBatch(
            batch_id=batch_id,
            layer=layer,
            task_keys=selected,
            observation_barriers=observation_barriers,
            protocol=protocol,
        )

    def finalize(
        self,
        *,
        graph_name: str,
        metadata: dict[str, Any],
    ) -> None:
        kind_counts = Counter(record["kind"] for record in self._task_records)
        payload_bytes: Counter[str] = Counter()
        for record in self._task_records:
            if record["kind"] == "transfer":
                payload_bytes[record.get("payload_kind") or "unspecified"] += int(
                    record["transfer_bytes"]
                )
        manifest = {
            "format_version": 2,
            "dag_mode": "dynamic_append_v1",
            "graph": graph_name,
            "num_ranks": self.num_ranks,
            "task_count": len(self._task_records),
            "barrier_count": len(self.barrier_ids),
            "compute_task_count": kind_counts["compute"],
            "transfer_task_count": kind_counts["transfer"],
            "transfer_bytes_by_payload": dict(sorted(payload_bytes.items())),
            "dynamic_batches": self._batches,
            "metadata": metadata,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_snapshots(include_task_map=True)

    def _write_snapshots(self, *, include_task_map: bool) -> None:
        (self.output_dir / "workload.dag").write_text(
            "\n".join(self._dag_lines) + "\n", encoding="utf-8"
        )
        if include_task_map:
            (self.output_dir / "task_map.json").write_text(
                json.dumps(
                    _json_ready({"tasks": self._task_records}),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        (self.output_dir / "dynamic_batches.json").write_text(
            json.dumps(
                _json_ready({"batches": self._batches}),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


def probeep_weight_dispatch_observations(
    graph: TaskGraph, layer: int
) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for micro_batch in (0, 1):
        stage_id = f"mb{micro_batch}.layer{layer}.weight_dispatch"
        keys = tuple(
            task.key
            for task in graph.tasks
            if task.metadata.get("communication_stage_id") == stage_id
        )
        if not keys:
            raise ValidationError(f"missing ProbeEP stage {stage_id}")
        result[layer * 2 + micro_batch] = keys
    return result
