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
    "expert_weight_scatter": "prefetch",
    "expert_weight_rdma": "prefetch",
    "expert_weight_gather": "prefetch",
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
    cpu_task_count: int
    cpu_streams_global: int
    communication_task_count: int
    communication_phase_count: int
    layer_count: int
    microbatch_group_size: int = 2

    def manifest(self) -> dict[str, object]:
        return {
            "model": "per_rank_two_stream_double_buffered_v1",
            "compute_streams_per_rank": 1,
            "communication_streams_per_rank": 1,
            "layer_count": self.layer_count,
            "microbatch_group_size": self.microbatch_group_size,
            "microbatch_group_completion": "complete_workload_drain",
            "schedule_order": "group_then_layer_wavefront",
            "phase_internal_flows_parallel": True,
            "stream_order_lowering": "predecessor_edges",
            "added_dependencies": self.added_dependencies,
            "compute_task_count": self.compute_task_count,
            "cpu_task_count": self.cpu_task_count,
            "cpu_streams_global": self.cpu_streams_global,
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

    microbatch_fragments = [
        tuple(graph.task(key) for key in keys) for keys in microbatch_task_keys
    ]
    layer_fragments: dict[tuple[int, int], list[Task]] = {}
    layer_ids: tuple[int, ...] | None = None
    for micro_batch, tasks in enumerate(microbatch_fragments):
        current_layers: set[int] = set()
        for task in tasks:
            layer = task.metadata.get("layer", 0)
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                raise ValidationError(
                    f"task {task.key} has invalid layer metadata {layer!r}"
                )
            current_layers.add(layer)
            layer_fragments.setdefault((micro_batch, layer), []).append(task)
        ordered_layers = tuple(sorted(current_layers))
        if ordered_layers != tuple(range(len(ordered_layers))):
            raise ValidationError(
                f"microbatch {micro_batch} layers must be contiguous from zero"
            )
        if layer_ids is None:
            layer_ids = ordered_layers
        elif layer_ids != ordered_layers:
            raise ValidationError("all microbatches must contain the same layers")
    assert layer_ids is not None

    compute_sections: dict[
        tuple[int, int], dict[str, dict[int, list[Task]]]
    ] = {}
    cpu_sections: dict[tuple[int, int], list[Task]] = {}
    transfer_phases: dict[tuple[int, int], dict[str, list[Task]]] = {}
    for micro_batch in range(len(microbatch_fragments)):
        for layer in layer_ids:
            tasks = layer_fragments[(micro_batch, layer)]
            fragment_key = (micro_batch, layer)
            sections = {
                "front": {rank: [] for rank in range(graph.num_ranks)},
                "expert": {rank: [] for rank in range(graph.num_ranks)},
                "final": {rank: [] for rank in range(graph.num_ranks)},
            }
            phases = {"prefetch": [], "dispatch": [], "combine": []}
            cpu_tasks: list[Task] = []
            for task in tasks:
                task.metadata["micro_batch"] = micro_batch
                task.metadata["layer"] = layer
                if task.kind == "compute":
                    if task.metadata.get("logical_resource") == "cpu":
                        task.metadata["logical_stream"] = "cpu"
                        task.metadata["stream_section"] = "planner"
                        cpu_tasks.append(task)
                        continue
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
            compute_sections[fragment_key] = sections
            cpu_sections[fragment_key] = cpu_tasks
            transfer_phases[fragment_key] = phases

    added_dependencies = 0

    def add_dependency(task: Task, predecessor: Task) -> None:
        nonlocal added_dependencies
        if task.key == predecessor.key:
            raise ValidationError(f"stream scheduler created self-edge for {task.key}")
        if predecessor.key not in task.predecessors:
            graph.add_dependency(task.key, predecessor.key)
            added_dependencies += 1

    # CUDA events connecting compute producers and communication consumers.
    for micro_batch in range(len(microbatch_fragments)):
        for layer in layer_ids:
            sections = compute_sections[(micro_batch, layer)]
            phases = transfer_phases[(micro_batch, layer)]
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
                if len(layer_ids) > 1:
                    phase_id = f"mb{micro_batch}.layer{layer}.{phase_name}"
                for task in tasks:
                    task.metadata["stream_phase_id"] = phase_id

    # One compute stream per rank. At a layer boundary, MB0 enters the next
    # layer before MB1's current-layer final, allowing next Attention to
    # overlap with the tail of MB1 Combine on the communication stream.
    for rank in range(graph.num_ranks):
        compute_order: list[Task] = []
        for pair_start in range(0, len(microbatch_fragments), 2):
            pair = tuple(range(
                pair_start, min(pair_start + 2, len(microbatch_fragments))
            ))
            first_layer = layer_ids[0]
            for micro_batch in pair:
                compute_order.extend(
                    compute_sections[(micro_batch, first_layer)]["front"][rank]
                )
            for layer_index, layer in enumerate(layer_ids):
                for micro_batch in pair:
                    compute_order.extend(
                        compute_sections[(micro_batch, layer)]["expert"][rank]
                    )
                next_layer = (
                    layer_ids[layer_index + 1]
                    if layer_index + 1 < len(layer_ids)
                    else None
                )
                for micro_batch in pair:
                    compute_order.extend(
                        compute_sections[(micro_batch, layer)]["final"][rank]
                    )
                    if next_layer is not None:
                        compute_order.extend(
                            compute_sections[(micro_batch, next_layer)]["front"][rank]
                        )
        for sequence, task in enumerate(compute_order):
            task.metadata["stream_sequence"] = sequence
        for predecessor, task in zip(compute_order, compute_order[1:]):
            add_dependency(task, predecessor)

    # A new pair starts only after the preceding pair has drained on every rank.
    for next_pair_start in range(2, len(microbatch_fragments), 2):
        previous_pair_start = next_pair_start - 2
        previous_pair = range(
            previous_pair_start,
            min(previous_pair_start + 2, len(microbatch_fragments)),
        )
        previous_terminals = [
            task
            for micro_batch in previous_pair
            for tasks in compute_sections[(micro_batch, layer_ids[-1])][
                "final"
            ].values()
            for task in tasks
        ]
        if not previous_terminals:
            raise ValidationError(
                f"microbatch group at {previous_pair_start} has no final tasks"
            )
        for tasks in compute_sections[(next_pair_start, layer_ids[0])][
            "front"
        ].values():
            if not tasks:
                continue
            for predecessor in previous_terminals:
                add_dependency(tasks[0], predecessor)

    # One logical host-CPU stream: planning can overlap accelerator work, but
    # separate planner invocations do not execute concurrently.
    cpu_order: list[Task] = []
    for pair_start in range(0, len(microbatch_fragments), 2):
        pair = range(
            pair_start, min(pair_start + 2, len(microbatch_fragments))
        )
        for layer in layer_ids:
            for micro_batch in pair:
                cpu_order.extend(cpu_sections[(micro_batch, layer)])
    for sequence, task in enumerate(cpu_order):
        task.metadata["cpu_stream_sequence"] = sequence
    for predecessor, task in zip(cpu_order, cpu_order[1:]):
        add_dependency(task, predecessor)

    # One communication stream per rank. Flow tasks inside one phase remain parallel.
    phase_order: list[tuple[str, list[Task]]] = []
    for pair_start in range(0, len(microbatch_fragments), 2):
        pair = range(
            pair_start, min(pair_start + 2, len(microbatch_fragments))
        )
        for layer in layer_ids:
            for micro_batch in pair:
                phases = transfer_phases[(micro_batch, layer)]
                weight_dispatch = phases["prefetch"] + phases["dispatch"]
                is_probeep = any(
                    task.metadata.get("algorithm") == "probeep"
                    for task in weight_dispatch
                )
                if is_probeep and weight_dispatch:
                    phase_order.append(("weight_dispatch", weight_dispatch))
                    stage_id = f"mb{micro_batch}.weight_dispatch"
                    if len(layer_ids) > 1:
                        stage_id = (
                            f"mb{micro_batch}.layer{layer}.weight_dispatch"
                        )
                    for task in weight_dispatch:
                        task.metadata["communication_stage_id"] = stage_id
                else:
                    for phase_name in ("prefetch", "dispatch"):
                        if phases[phase_name]:
                            phase_order.append((phase_name, phases[phase_name]))
            for micro_batch in pair:
                phases = transfer_phases[(micro_batch, layer)]
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
        compute_task_count=sum(
            task.kind == "compute"
            and task.metadata.get("logical_resource") != "cpu"
            for task in graph.tasks
        ),
        cpu_task_count=sum(
            task.kind == "compute"
            and task.metadata.get("logical_resource") == "cpu"
            for task in graph.tasks
        ),
        cpu_streams_global=1 if cpu_order else 0,
        communication_task_count=sum(task.kind == "transfer" for task in graph.tasks),
        communication_phase_count=len(phase_order),
        layer_count=len(layer_ids),
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
