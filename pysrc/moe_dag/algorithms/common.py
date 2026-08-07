from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, TypeVar

from ..schema import Placement, RoutingAssignment, ValidationError


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


def destination_forward_route(
    placement: Placement, src_rank: int, dst_rank: int
) -> tuple[str | None, int | None]:
    if placement.rank_server(src_rank) == placement.rank_server(dst_rank):
        return None, None
    relay = placement.server_rank(
        placement.rank_server(dst_rank), placement.rank_local(src_rank)
    )
    return f"server_forward src_relay:{src_rank} dst_relay:{relay}", relay


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
