from __future__ import annotations

from dataclasses import dataclass
import re

from ..graph import Task, TaskGraph
from ..schema import ValidationError


_FRONT_COMPUTE = {
    "attention",
    "router_projection",
    "per_server_planning_proxy",
}
_EXPERT_COMPUTE = {"expert_ffn"}
_FINAL_COMPUTE = {"combine_reduce", "combine_final_reduce"}
_TRANSFER_PHASE = {
    "expert_weight_prefetch": "prefetch",
    "dispatch_hidden": "dispatch",
    "combine_partial": "combine",
}
_SERVER_FORWARD_RE = re.compile(
    r"^server_forward src_relay:(?P<src>\d+) dst_relay:(?P<dst>\d+)$"
)


@dataclass(frozen=True)
class TwoStreamScheduleResult:
    added_dependencies: int
    compute_task_count: int
    communication_task_count: int
    communication_phase_count: int
    microbatch_group_size: int = 2

    def manifest(self) -> dict[str, object]:
        return {
            "model": "per_rank_two_stream_double_buffered_v1",
            "compute_streams_per_rank": 1,
            "communication_streams_per_rank": 1,
            "microbatch_group_size": self.microbatch_group_size,
            "phase_internal_flows_parallel": True,
            "stream_order_lowering": "predecessor_edges",
            "added_dependencies": self.added_dependencies,
            "compute_task_count": self.compute_task_count,
            "communication_task_count": self.communication_task_count,
            "communication_phase_count": self.communication_phase_count,
        }


def _compute_section(task: Task) -> str:
    operation = task.metadata.get("operation")
    if operation in _FRONT_COMPUTE:
        return "front"
    if operation in _EXPERT_COMPUTE:
        return "expert"
    if operation in _FINAL_COMPUTE:
        return "final"
    raise ValidationError(
        f"two-stream scheduler does not recognize compute operation {operation!r} "
        f"for task {task.key}"
    )


def _transfer_phase(task: Task) -> str:
    try:
        return _TRANSFER_PHASE[task.payload_kind or ""]
    except KeyError as exc:
        raise ValidationError(
            "two-stream scheduler does not recognize transfer payload "
            f"{task.payload_kind!r} for task {task.key}"
        ) from exc


def apply_double_buffered_two_stream_schedule(
    graph: TaskGraph,
    microbatch_task_keys: tuple[tuple[str, ...], ...],
) -> TwoStreamScheduleResult:
    if not microbatch_task_keys:
        raise ValidationError("two-stream scheduler requires at least one microbatch")
    flattened = [key for keys in microbatch_task_keys for key in keys]
    if len(flattened) != len(set(flattened)):
        raise ValidationError("two-stream microbatch fragments overlap")
    if set(flattened) != {task.key for task in graph.tasks}:
        raise ValidationError("two-stream fragments must cover the complete graph")

    fragments = [tuple(graph.task(key) for key in keys) for keys in microbatch_task_keys]
    compute_sections: list[dict[str, dict[int, list[Task]]]] = []
    transfer_phases: list[dict[str, list[Task]]] = []
    for micro_batch, tasks in enumerate(fragments):
        sections = {
            "front": {rank: [] for rank in range(graph.num_ranks)},
            "expert": {rank: [] for rank in range(graph.num_ranks)},
            "final": {rank: [] for rank in range(graph.num_ranks)},
        }
        phases = {"prefetch": [], "dispatch": [], "combine": []}
        for task in tasks:
            task.metadata["micro_batch"] = micro_batch
            if task.kind == "compute":
                assert task.rank is not None
                section = _compute_section(task)
                sections[section][task.rank].append(task)
                task.metadata["logical_stream"] = "compute"
                task.metadata["stream_section"] = section
            else:
                phase = _transfer_phase(task)
                phases[phase].append(task)
                task.metadata["logical_stream"] = "communication"
                task.metadata["stream_phase"] = phase
        compute_sections.append(sections)
        transfer_phases.append(phases)

    added_dependencies = 0

    def add_dependency(task: Task, predecessor: Task) -> None:
        nonlocal added_dependencies
        if task.key == predecessor.key:
            raise ValidationError(f"stream scheduler created self-edge for {task.key}")
        if predecessor.key not in task.predecessors:
            graph.add_dependency(task.key, predecessor.key)
            added_dependencies += 1

    # CUDA events connecting compute producers and communication consumers.
    for micro_batch, (sections, phases) in enumerate(
        zip(compute_sections, transfer_phases)
    ):
        front_tail = {
            rank: tasks[-1]
            for rank, tasks in sections["front"].items()
            if tasks
        }
        experts = {
            rank: tasks[-1]
            for rank, tasks in sections["expert"].items()
            if tasks
        }
        finals = {
            rank: tasks[-1]
            for rank, tasks in sections["final"].items()
            if tasks
        }

        for phase_name in ("prefetch", "dispatch"):
            for task in phases[phase_name]:
                for rank in _participant_ranks(task, graph.num_ranks):
                    if rank in front_tail:
                        add_dependency(task, front_tail[rank])

        dispatch_by_rank = _tasks_touching_ranks(
            phases["dispatch"], graph.num_ranks
        )
        for rank, expert in experts.items():
            for task in dispatch_by_rank[rank]:
                add_dependency(expert, task)

        for task in phases["combine"]:
            for rank in _participant_ranks(task, graph.num_ranks):
                if rank in experts:
                    add_dependency(task, experts[rank])

        combine_by_rank = _tasks_touching_ranks(
            phases["combine"], graph.num_ranks
        )
        for rank, final in finals.items():
            for task in combine_by_rank[rank]:
                add_dependency(final, task)

        for phase_name, tasks in phases.items():
            phase_id = f"mb{micro_batch}.{phase_name}"
            for task in tasks:
                task.metadata["stream_phase_id"] = phase_id

    # One compute stream per rank. Double-buffer each adjacent microbatch pair.
    for rank in range(graph.num_ranks):
        compute_order: list[Task] = []
        for pair_start in range(0, len(fragments), 2):
            pair = compute_sections[pair_start:pair_start + 2]
            for section_name in ("front", "expert", "final"):
                for sections in pair:
                    compute_order.extend(sections[section_name][rank])
        for sequence, task in enumerate(compute_order):
            task.metadata["stream_sequence"] = sequence
        for predecessor, task in zip(compute_order, compute_order[1:]):
            add_dependency(task, predecessor)

    # One communication stream per rank. Flow tasks inside one phase remain parallel.
    phase_order: list[tuple[str, list[Task]]] = []
    for pair_start in range(0, len(fragments), 2):
        pair = transfer_phases[pair_start:pair_start + 2]
        for phases in pair:
            for phase_name in ("prefetch", "dispatch"):
                if phases[phase_name]:
                    phase_order.append((phase_name, phases[phase_name]))
        for phases in pair:
            if phases["combine"]:
                phase_order.append(("combine", phases["combine"]))

    communication_tail: dict[int, list[Task]] = {
        rank: [] for rank in range(graph.num_ranks)
    }
    for sequence, (_, phase_tasks) in enumerate(phase_order):
        for task in phase_tasks:
            task.metadata["communication_stream_sequence"] = sequence
            for rank in _participant_ranks(task, graph.num_ranks):
                for predecessor in communication_tail[rank]:
                    add_dependency(task, predecessor)
        touching = _tasks_touching_ranks(phase_tasks, graph.num_ranks)
        for rank, tasks in touching.items():
            if tasks:
                communication_tail[rank] = list(tasks)

    graph.validate()
    return TwoStreamScheduleResult(
        added_dependencies=added_dependencies,
        compute_task_count=sum(task.kind == "compute" for task in graph.tasks),
        communication_task_count=sum(task.kind == "transfer" for task in graph.tasks),
        communication_phase_count=len(phase_order),
    )


def _tasks_touching_ranks(
    tasks: list[Task], num_ranks: int
) -> dict[int, list[Task]]:
    result = {rank: [] for rank in range(num_ranks)}
    for task in tasks:
        for rank in _participant_ranks(task, num_ranks):
            result[rank].append(task)
    return result


def _participant_ranks(task: Task, num_ranks: int) -> set[int]:
    if task.src_rank is None or task.dst_rank is None:
        raise ValidationError(f"transfer task {task.key} is missing endpoints")
    result = {task.src_rank, task.dst_rank}
    if task.route_spec is None:
        return result
    match = _SERVER_FORWARD_RE.fullmatch(task.route_spec)
    if match is None:
        return result
    result.update((int(match.group("src")), int(match.group("dst"))))
    if min(result) < 0 or max(result) >= num_ranks:
        raise ValidationError(f"task {task.key} has an out-of-range relay rank")
    return result
