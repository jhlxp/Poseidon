from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schema import MoEInvocation, RoutingAssignment, ValidationError


@dataclass(frozen=True)
class ExpertInstance:
    instance_id: str
    logical_expert: int
    rank: int
    kind: str
    physical_expert: int | None = None
    replica_index: int | None = None

    def manifest(self, load: int) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "logical_expert": self.logical_expert,
            "rank": self.rank,
            "kind": self.kind,
            "physical_expert": self.physical_expert,
            "replica_index": self.replica_index,
            "load": load,
        }


def _imbalance(values: list[int]) -> dict[str, float | int]:
    total = sum(values)
    mean = total / len(values) if values else 0.0
    maximum = max(values, default=0)
    minimum = min(values, default=0)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "max_mean": maximum / mean if mean else 0.0,
    }


def _snapshot(
    invocation: MoEInvocation,
    catalog: tuple[ExpertInstance, ...],
    loads: dict[str, int],
) -> dict[str, object]:
    rank_loads = [0] * invocation.placement.num_ranks
    server_loads = [0] * invocation.placement.num_servers
    logical_loads = [0] * invocation.placement.num_experts
    instances: list[dict[str, object]] = []
    for instance in catalog:
        load = loads.get(instance.instance_id, 0)
        rank_loads[instance.rank] += load
        server_loads[invocation.placement.rank_server(instance.rank)] += load
        logical_loads[instance.logical_expert] += load
        instances.append(instance.manifest(load))
    instance_values = [int(item["load"]) for item in instances]
    return {
        "instances": instances,
        "logical_expert_loads": logical_loads,
        "rank_loads": rank_loads,
        "server_loads": server_loads,
        "rank_imbalance": _imbalance(rank_loads),
        "server_imbalance": _imbalance(server_loads),
        "instance_imbalance": _imbalance(instance_values),
        "total_routes": sum(instance_values),
    }


def baseline_instances(invocation: MoEInvocation) -> tuple[ExpertInstance, ...]:
    return tuple(
        ExpertInstance(
            instance_id=f"logical:{expert}",
            logical_expert=expert,
            rank=invocation.placement.expert_rank(expert),
            kind="logical",
            physical_expert=expert,
            replica_index=0,
        )
        for expert in range(invocation.placement.num_experts)
    )


def build_expert_load_profile(
    invocation: MoEInvocation,
    *,
    after_instances: tuple[ExpertInstance, ...] | None = None,
    select_after_instance: Callable[[RoutingAssignment], str] | None = None,
) -> dict[str, object]:
    before_catalog = baseline_instances(invocation)
    after_catalog = after_instances or before_catalog
    if select_after_instance is None:
        select_after_instance = lambda assignment: f"logical:{assignment.expert_id}"

    ids: set[str] = set()
    for instance in after_catalog:
        if not instance.instance_id or instance.instance_id in ids:
            raise ValidationError(
                f"duplicate or empty expert instance ID: {instance.instance_id!r}"
            )
        ids.add(instance.instance_id)
        invocation.placement.validate_rank(instance.rank)
        if (
            instance.logical_expert < 0
            or instance.logical_expert >= invocation.placement.num_experts
        ):
            raise ValidationError(
                f"invalid logical expert in instance {instance.instance_id}"
            )

    before_loads: dict[str, int] = {}
    after_loads: dict[str, int] = {}
    for assignment in invocation.sorted_assignments():
        before_id = f"logical:{assignment.expert_id}"
        before_loads[before_id] = before_loads.get(before_id, 0) + 1
        after_id = select_after_instance(assignment)
        if after_id not in ids:
            raise ValidationError(f"assignment selected unknown expert instance: {after_id}")
        after_loads[after_id] = after_loads.get(after_id, 0) + 1

    before = _snapshot(invocation, before_catalog, before_loads)
    after = _snapshot(invocation, after_catalog, after_loads)
    if before["total_routes"] != after["total_routes"]:
        raise ValidationError("expert load profile lost routes between before and after")
    if before["logical_expert_loads"] != after["logical_expert_loads"]:
        raise ValidationError("expert load profile changed logical Gate assignments")
    return {
        "schema": "expert_load_profile_v1",
        "before": before,
        "after": after,
    }
