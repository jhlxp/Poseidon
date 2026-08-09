from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, TypeVar

from ..graph import TaskGraph
from ..schema import MoEInvocation, Placement, RoutingAssignment, ValidationError


T = TypeVar("T")


PayloadScope = Literal["none", "destination_rank"]


@dataclass(frozen=True)
class TokenPayloadPolicy:
    deduplicate: bool = True
    scope: PayloadScope = "destination_rank"

    def __post_init__(self) -> None:
        if self.scope not in {"none", "destination_rank"}:
            raise ValidationError(f"unsupported token payload scope: {self.scope}")
        if self.deduplicate and self.scope != "destination_rank":
            raise ValidationError(
                "deduplicated token payloads require destination_rank scope"
            )
        if not self.deduplicate and self.scope != "none":
            raise ValidationError("non-deduplicated token payloads require none scope")


@dataclass(frozen=True)
class TokenPayload:
    src_rank: int
    token_id: int
    dst_rank: int
    routes: tuple[RoutingAssignment, ...]

    @property
    def token(self) -> tuple[int, int]:
        return (self.src_rank, self.token_id)


@dataclass(frozen=True)
class ServerTokenPayload:
    src_rank: int
    token_id: int
    dst_server: int
    rank_payloads: tuple[TokenPayload, ...]

    @property
    def routes(self) -> tuple[RoutingAssignment, ...]:
        return tuple(
            route
            for payload in self.rank_payloads
            for route in payload.routes
        )

    @property
    def destination_ranks(self) -> tuple[int, ...]:
        return tuple(payload.dst_rank for payload in self.rank_payloads)


@dataclass(frozen=True)
class HierarchicalPayloadPlan:
    rank_payloads_by_pair: dict[tuple[int, int], tuple[TokenPayload, ...]]
    server_payloads_by_pair: dict[
        tuple[int, int], tuple[ServerTokenPayload, ...]
    ]
    route_count: int

    @property
    def rank_payload_count(self) -> int:
        return sum(len(items) for items in self.rank_payloads_by_pair.values())

    @property
    def server_payload_count(self) -> int:
        return sum(len(items) for items in self.server_payloads_by_pair.values())


@dataclass
class HierarchicalTransferSummary:
    task_count_by_leg: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    bytes_by_leg: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    payload_count_by_leg: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record(self, leg: str, transfer_bytes: int, payload_count: int) -> None:
        self.task_count_by_leg[leg] += 1
        self.bytes_by_leg[leg] += transfer_bytes
        self.payload_count_by_leg[leg] += payload_count

    def manifest(self) -> dict[str, object]:
        return {
            "task_count_by_leg": dict(sorted(self.task_count_by_leg.items())),
            "bytes_by_leg": dict(sorted(self.bytes_by_leg.items())),
            "payload_count_by_leg": dict(
                sorted(self.payload_count_by_leg.items())
            ),
        }


def hierarchical_token_server_pair_profile(
    invocation: MoEInvocation,
    plan: HierarchicalPayloadPlan,
) -> list[dict[str, object]]:
    """Summarize physical fabric token traffic by directed server pair and rail."""
    placement = invocation.placement
    rows: dict[tuple[int, int], dict[str, object]] = {}

    def row_for(source_server: int, destination_server: int) -> dict[str, object]:
        key = (source_server, destination_server)
        if key not in rows:
            rows[key] = {
                "source_server": source_server,
                "destination_server": destination_server,
                "dispatch_token_payloads": 0,
                "dispatch_expert_routes": 0,
                "dispatch_bytes": 0,
                "combine_token_payloads": 0,
                "combine_expert_routes": 0,
                "combine_bytes": 0,
                "nics": {
                    rail: {
                        "rail": rail,
                        "source_rank": placement.server_rank(source_server, rail),
                        "destination_rank": placement.server_rank(
                            destination_server, rail
                        ),
                        "dispatch_token_payloads": 0,
                        "dispatch_expert_routes": 0,
                        "dispatch_bytes": 0,
                        "combine_token_payloads": 0,
                        "combine_expert_routes": 0,
                        "combine_bytes": 0,
                    }
                    for rail in range(placement.gpus_per_server)
                },
            }
        return rows[key]

    for (source_rank, destination_server), payloads in (
        plan.server_payloads_by_pair.items()
    ):
        source_server = placement.rank_server(source_rank)
        if source_server == destination_server:
            continue
        token_count = len(payloads)
        route_count = sum(len(payload.routes) for payload in payloads)
        rail = placement.rank_local(source_rank)

        dispatch_row = row_for(source_server, destination_server)
        dispatch_nic = dispatch_row["nics"][rail]
        dispatch_bytes = token_count * invocation.dispatch_token_bytes
        for target in (dispatch_row, dispatch_nic):
            target["dispatch_token_payloads"] += token_count
            target["dispatch_expert_routes"] += route_count
            target["dispatch_bytes"] += dispatch_bytes

        combine_row = row_for(destination_server, source_server)
        combine_nic = combine_row["nics"][rail]
        combine_bytes = token_count * invocation.combine_token_bytes
        for target in (combine_row, combine_nic):
            target["combine_token_payloads"] += token_count
            target["combine_expert_routes"] += route_count
            target["combine_bytes"] += combine_bytes

    result: list[dict[str, object]] = []
    for row in rows.values():
        nic_rows = list(row.pop("nics").values())
        for nic in nic_rows:
            nic["token_bytes_total"] = (
                nic["dispatch_bytes"] + nic["combine_bytes"]
            )
        row["token_bytes_total"] = row["dispatch_bytes"] + row["combine_bytes"]
        row["nics"] = nic_rows
        result.append(row)
    return sorted(
        result,
        key=lambda item: (item["source_server"], item["destination_server"]),
    )


def plan_token_payloads(
    assignments: Iterable[RoutingAssignment],
    destination_rank: Callable[[RoutingAssignment], int],
    policy: TokenPayloadPolicy = TokenPayloadPolicy(),
) -> dict[tuple[int, int], tuple[TokenPayload, ...]]:
    grouped: dict[tuple[int, ...], list[RoutingAssignment]] = {}
    destinations: dict[tuple[int, ...], int] = {}
    for assignment in sorted(assignments):
        dst_rank = destination_rank(assignment)
        if policy.deduplicate:
            payload_key = (assignment.src_rank, assignment.token_id, dst_rank)
        else:
            payload_key = (
                assignment.src_rank,
                assignment.token_id,
                assignment.topk_slot,
                assignment.expert_id,
                dst_rank,
            )
        grouped.setdefault(payload_key, []).append(assignment)
        destinations[payload_key] = dst_rank

    by_pair: dict[tuple[int, int], list[TokenPayload]] = {}
    for payload_key, routes in grouped.items():
        first = routes[0]
        dst_rank = destinations[payload_key]
        by_pair.setdefault((first.src_rank, dst_rank), []).append(
            TokenPayload(
                src_rank=first.src_rank,
                token_id=first.token_id,
                dst_rank=dst_rank,
                routes=tuple(routes),
            )
        )
    return {
        pair: tuple(
            sorted(
                payloads,
                key=lambda item: (
                    item.src_rank,
                    item.token_id,
                    item.routes[0].topk_slot,
                    item.routes[0].expert_id,
                ),
            )
        )
        for pair, payloads in sorted(by_pair.items())
    }


def plan_hierarchical_token_payloads(
    assignments: Iterable[RoutingAssignment],
    destination_rank: Callable[[RoutingAssignment], int],
    placement: Placement,
) -> HierarchicalPayloadPlan:
    values = tuple(sorted(assignments))
    rank_payloads = plan_token_payloads(
        values,
        destination_rank,
        TokenPayloadPolicy(),
    )
    grouped: dict[tuple[int, int, int], list[TokenPayload]] = defaultdict(list)
    for (_, dst_rank), payloads in rank_payloads.items():
        dst_server = placement.rank_server(dst_rank)
        for payload in payloads:
            grouped[(payload.src_rank, payload.token_id, dst_server)].append(
                payload
            )

    by_pair: dict[tuple[int, int], list[ServerTokenPayload]] = defaultdict(list)
    for (src_rank, token_id, dst_server), payloads in sorted(grouped.items()):
        by_pair[(src_rank, dst_server)].append(
            ServerTokenPayload(
                src_rank=src_rank,
                token_id=token_id,
                dst_server=dst_server,
                rank_payloads=tuple(sorted(payloads, key=lambda item: item.dst_rank)),
            )
        )
    return HierarchicalPayloadPlan(
        rank_payloads_by_pair=rank_payloads,
        server_payloads_by_pair={
            pair: tuple(items) for pair, items in sorted(by_pair.items())
        },
        route_count=len(values),
    )


def build_hierarchical_dispatch(
    graph: TaskGraph,
    invocation: MoEInvocation,
    plan: HierarchicalPayloadPlan,
    *,
    algorithm: str,
    chunk_tokens: int,
    predecessors_for_server: Callable[[int], set[str]],
    metadata_sample_limit: int,
    arrivals: dict[int, set[str]],
    summary: HierarchicalTransferSummary,
) -> None:
    placement = invocation.placement
    phase_id = f"{invocation.invocation_id}:dispatch:hierarchical"

    # Same-server traffic needs only the rank-level NVLink leg.
    for (src, dst), payloads in plan.rank_payloads_by_pair.items():
        server = placement.rank_server(dst)
        if placement.rank_server(src) != server or src == dst:
            continue
        for chunk_id, payload_chunk in enumerate(chunked(payloads, chunk_tokens)):
            key = (
                f"{invocation.invocation_id}.dispatch.local."
                f"src{src}.dst{dst}.chunk{chunk_id}"
            )
            transfer_bytes = len(payload_chunk) * invocation.dispatch_token_bytes
            graph.add_transfer(
                key,
                src,
                dst,
                transfer_bytes,
                "dispatch_hidden",
                phase_id,
                predecessors=predecessors_for_server(server),
                chunk_id=chunk_id,
                metadata=_rank_leg_metadata(
                    payload_chunk,
                    algorithm=algorithm,
                    leg="dispatch_local",
                    sample_limit=metadata_sample_limit,
                ),
            )
            arrivals[dst].add(key)
            summary.record("dispatch_local", transfer_bytes, len(payload_chunk))

    # Remote traffic sends one hidden per token/server over RDMA, then fans out
    # to each unique execution rank through the destination NVLink domain.
    for (src, dst_server), server_payloads in plan.server_payloads_by_pair.items():
        if placement.rank_server(src) == dst_server:
            continue
        relay = placement.server_rank(dst_server, placement.rank_local(src))
        for chunk_id, server_chunk in enumerate(
            chunked(server_payloads, chunk_tokens)
        ):
            fabric_key = (
                f"{invocation.invocation_id}.dispatch.fabric."
                f"src{src}.server{dst_server}.relay{relay}.chunk{chunk_id}"
            )
            fabric_bytes = len(server_chunk) * invocation.dispatch_token_bytes
            graph.add_transfer(
                fabric_key,
                src,
                relay,
                fabric_bytes,
                "dispatch_hidden",
                phase_id,
                predecessors=predecessors_for_server(dst_server),
                chunk_id=chunk_id,
                metadata=_server_leg_metadata(
                    server_chunk,
                    algorithm=algorithm,
                    leg="dispatch_fabric",
                    relay=relay,
                    sample_limit=metadata_sample_limit,
                ),
            )
            summary.record("dispatch_fabric", fabric_bytes, len(server_chunk))

            by_destination: dict[int, list[TokenPayload]] = defaultdict(list)
            for server_payload in server_chunk:
                for rank_payload in server_payload.rank_payloads:
                    by_destination[rank_payload.dst_rank].append(rank_payload)
            for dst, rank_payloads in sorted(by_destination.items()):
                if dst == relay:
                    arrivals[dst].add(fabric_key)
                    continue
                local_key = (
                    f"{invocation.invocation_id}.dispatch.local."
                    f"src{src}.relay{relay}.dst{dst}.serverchunk{chunk_id}"
                )
                local_bytes = len(rank_payloads) * invocation.dispatch_token_bytes
                graph.add_transfer(
                    local_key,
                    relay,
                    dst,
                    local_bytes,
                    "dispatch_hidden",
                    phase_id,
                    predecessors={fabric_key},
                    chunk_id=chunk_id,
                    metadata=_rank_leg_metadata(
                        tuple(rank_payloads),
                        algorithm=algorithm,
                        leg="dispatch_local",
                        relay=relay,
                        sample_limit=metadata_sample_limit,
                    ),
                )
                arrivals[dst].add(local_key)
                summary.record("dispatch_local", local_bytes, len(rank_payloads))


def build_hierarchical_combine(
    graph: TaskGraph,
    invocation: MoEInvocation,
    plan: HierarchicalPayloadPlan,
    expert_keys: dict[int, str],
    *,
    algorithm: str,
    chunk_tokens: int,
    metadata_sample_limit: int,
    arrivals: dict[int, set[str]],
    local_expert_origins: set[int],
    summary: HierarchicalTransferSummary,
) -> None:
    placement = invocation.placement
    phase_id = f"{invocation.invocation_id}:combine:hierarchical"

    # Results produced inside the origin server reduce through local links.
    for (origin, execution_rank), payloads in plan.rank_payloads_by_pair.items():
        if placement.rank_server(origin) != placement.rank_server(execution_rank):
            continue
        if origin == execution_rank:
            local_expert_origins.add(origin)
            continue
        for chunk_id, payload_chunk in enumerate(chunked(payloads, chunk_tokens)):
            key = (
                f"{invocation.invocation_id}.combine.local."
                f"src{execution_rank}.origin{origin}.chunk{chunk_id}"
            )
            transfer_bytes = len(payload_chunk) * invocation.combine_token_bytes
            graph.add_transfer(
                key,
                execution_rank,
                origin,
                transfer_bytes,
                "combine_partial",
                phase_id,
                predecessors={expert_keys[execution_rank]},
                chunk_id=chunk_id,
                metadata=_rank_leg_metadata(
                    payload_chunk,
                    algorithm=algorithm,
                    leg="combine_local_reduce",
                    relay=origin,
                    sample_limit=metadata_sample_limit,
                ),
            )
            arrivals[origin].add(key)
            summary.record(
                "combine_local_reduce", transfer_bytes, len(payload_chunk)
            )

    # A remote expert server first reduces rank partials at the relay whose
    # local index matches the original source rank, then returns one partial
    # per token/server over the same rail.
    for (origin, execution_server), server_payloads in plan.server_payloads_by_pair.items():
        if placement.rank_server(origin) == execution_server:
            continue
        relay = placement.server_rank(
            execution_server, placement.rank_local(origin)
        )
        for chunk_id, server_chunk in enumerate(
            chunked(server_payloads, chunk_tokens)
        ):
            by_execution: dict[int, list[TokenPayload]] = defaultdict(list)
            for server_payload in server_chunk:
                for rank_payload in server_payload.rank_payloads:
                    by_execution[rank_payload.dst_rank].append(rank_payload)

            fabric_predecessors: set[str] = set()
            for execution_rank, rank_payloads in sorted(by_execution.items()):
                if execution_rank == relay:
                    fabric_predecessors.add(expert_keys[execution_rank])
                    continue
                local_key = (
                    f"{invocation.invocation_id}.combine.local."
                    f"src{execution_rank}.relay{relay}."
                    f"origin{origin}.serverchunk{chunk_id}"
                )
                local_bytes = len(rank_payloads) * invocation.combine_token_bytes
                graph.add_transfer(
                    local_key,
                    execution_rank,
                    relay,
                    local_bytes,
                    "combine_partial",
                    phase_id,
                    predecessors={expert_keys[execution_rank]},
                    chunk_id=chunk_id,
                    metadata=_rank_leg_metadata(
                        tuple(rank_payloads),
                        algorithm=algorithm,
                        leg="combine_local_reduce",
                        relay=relay,
                        sample_limit=metadata_sample_limit,
                    ),
                )
                fabric_predecessors.add(local_key)
                summary.record(
                    "combine_local_reduce", local_bytes, len(rank_payloads)
                )

            fabric_key = (
                f"{invocation.invocation_id}.combine.fabric."
                f"relay{relay}.origin{origin}.server{execution_server}."
                f"chunk{chunk_id}"
            )
            fabric_bytes = len(server_chunk) * invocation.combine_token_bytes
            graph.add_transfer(
                fabric_key,
                relay,
                origin,
                fabric_bytes,
                "combine_partial",
                phase_id,
                predecessors=fabric_predecessors,
                chunk_id=chunk_id,
                metadata=_server_leg_metadata(
                    server_chunk,
                    algorithm=algorithm,
                    leg="combine_fabric",
                    relay=relay,
                    sample_limit=metadata_sample_limit,
                ),
            )
            arrivals[origin].add(fabric_key)
            summary.record("combine_fabric", fabric_bytes, len(server_chunk))


def _rank_leg_metadata(
    payloads: tuple[TokenPayload, ...],
    *,
    algorithm: str,
    leg: str,
    sample_limit: int,
    relay: int | None = None,
) -> dict[str, object]:
    sample = payloads[:sample_limit]
    return {
        "algorithm": algorithm,
        "hierarchical_leg": leg,
        "relay_rank": relay,
        "payload_count": len(payloads),
        "payload_sample_count": len(sample),
        "payload_metadata_truncated": len(sample) < len(payloads),
        "payloads": [
            {
                "src_rank": payload.src_rank,
                "token_id": payload.token_id,
                "destination_rank": payload.dst_rank,
                "route_count": len(payload.routes),
                "topk_slots": [route.topk_slot for route in payload.routes],
                "expert_ids": [route.expert_id for route in payload.routes],
            }
            for payload in sample
        ],
    }


def _server_leg_metadata(
    payloads: tuple[ServerTokenPayload, ...],
    *,
    algorithm: str,
    leg: str,
    relay: int,
    sample_limit: int,
) -> dict[str, object]:
    sample = payloads[:sample_limit]
    return {
        "algorithm": algorithm,
        "hierarchical_leg": leg,
        "relay_rank": relay,
        "payload_count": len(payloads),
        "payload_sample_count": len(sample),
        "payload_metadata_truncated": len(sample) < len(payloads),
        "payloads": [
            {
                "src_rank": payload.src_rank,
                "token_id": payload.token_id,
                "destination_server": payload.dst_server,
                "destination_ranks": list(payload.destination_ranks),
                "route_count": len(payload.routes),
                "topk_slots": [route.topk_slot for route in payload.routes],
                "expert_ids": [route.expert_id for route in payload.routes],
            }
            for payload in sample
        ],
    }


@dataclass(frozen=True)
class AlgorithmBuildResult:
    algorithm: str
    terminal_keys: frozenset[str]
    rank_terminal_keys: dict[int, frozenset[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


def chunked(items: Iterable[T], chunk_size: int) -> tuple[tuple[T, ...], ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    values = tuple(items)
    return tuple(
        values[index : index + chunk_size]
        for index in range(0, len(values), chunk_size)
    )
