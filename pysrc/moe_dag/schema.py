from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


class ValidationError(ValueError):
    pass


_DTYPE_BYTES = {
    "fp8": 1,
    "int8": 1,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


def dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype.lower()]
    except KeyError as exc:
        raise ValidationError(f"unsupported dtype: {dtype}") from exc


@dataclass(frozen=True, order=True)
class RoutingAssignment:
    src_rank: int
    token_id: int
    topk_slot: int
    expert_id: int
    route_weight: float = 1.0


@dataclass(frozen=True)
class Placement:
    num_ranks: int
    gpus_per_server: int
    expert_to_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.num_ranks <= 0:
            raise ValidationError("num_ranks must be positive")
        if self.gpus_per_server <= 0:
            raise ValidationError("gpus_per_server must be positive")
        if self.num_ranks % self.gpus_per_server != 0:
            raise ValidationError("num_ranks must be divisible by gpus_per_server")
        if not self.expert_to_rank:
            raise ValidationError("expert_to_rank must not be empty")
        for rank in self.expert_to_rank:
            self.validate_rank(rank)

    @property
    def num_experts(self) -> int:
        return len(self.expert_to_rank)

    @property
    def num_servers(self) -> int:
        return self.num_ranks // self.gpus_per_server

    def validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.num_ranks:
            raise ValidationError(
                f"rank {rank} outside configured range [0, {self.num_ranks})"
            )

    def rank_server(self, rank: int) -> int:
        self.validate_rank(rank)
        return rank // self.gpus_per_server

    def rank_local(self, rank: int) -> int:
        self.validate_rank(rank)
        return rank % self.gpus_per_server

    def server_rank(self, server: int, local_rank: int) -> int:
        if server < 0 or server >= self.num_servers:
            raise ValidationError(f"invalid server: {server}")
        if local_rank < 0 or local_rank >= self.gpus_per_server:
            raise ValidationError(f"invalid local rank: {local_rank}")
        return server * self.gpus_per_server + local_rank

    def expert_rank(self, expert_id: int) -> int:
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValidationError(f"invalid expert ID: {expert_id}")
        return self.expert_to_rank[expert_id]


@dataclass(frozen=True)
class MoEInvocation:
    invocation_id: str
    placement: Placement
    tokens_per_source_rank: tuple[int, ...]
    hidden: int
    ffn_hidden: int
    topk: int
    dispatch_dtype: str
    combine_dtype: str
    weight_dtype: str
    assignments: tuple[RoutingAssignment, ...]

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValidationError("invocation_id must not be empty")
        if len(self.tokens_per_source_rank) != self.placement.num_ranks:
            raise ValidationError("tokens_per_source_rank length must equal num_ranks")
        if any(count < 0 for count in self.tokens_per_source_rank):
            raise ValidationError("token counts must be non-negative")
        if self.hidden <= 0 or self.ffn_hidden <= 0:
            raise ValidationError("hidden dimensions must be positive")
        if self.topk <= 0 or self.topk > self.placement.num_experts:
            raise ValidationError("topk must be in [1, num_experts]")
        dtype_bytes(self.dispatch_dtype)
        dtype_bytes(self.combine_dtype)
        dtype_bytes(self.weight_dtype)
        self._validate_assignments()

    def _validate_assignments(self) -> None:
        seen: set[tuple[int, int, int]] = set()
        slots_by_token: dict[tuple[int, int], set[int]] = {}
        for assignment in self.assignments:
            self.placement.validate_rank(assignment.src_rank)
            token_count = self.tokens_per_source_rank[assignment.src_rank]
            if assignment.token_id < 0 or assignment.token_id >= token_count:
                raise ValidationError(
                    f"token {assignment.token_id} outside source rank "
                    f"{assignment.src_rank} range [0, {token_count})"
                )
            if assignment.topk_slot < 0 or assignment.topk_slot >= self.topk:
                raise ValidationError(f"invalid top-k slot: {assignment.topk_slot}")
            self.placement.expert_rank(assignment.expert_id)
            key = (
                assignment.src_rank,
                assignment.token_id,
                assignment.topk_slot,
            )
            if key in seen:
                raise ValidationError(f"duplicate routing assignment: {key}")
            seen.add(key)
            slots_by_token.setdefault(key[:2], set()).add(assignment.topk_slot)

        expected_tokens = {
            (rank, token_id)
            for rank, count in enumerate(self.tokens_per_source_rank)
            for token_id in range(count)
        }
        if set(slots_by_token) != expected_tokens:
            missing = sorted(expected_tokens - set(slots_by_token))
            raise ValidationError(f"routing assignments missing tokens: {missing[:5]}")
        expected_slots = set(range(self.topk))
        for token, slots in slots_by_token.items():
            if slots != expected_slots:
                raise ValidationError(
                    f"token {token} has slots {sorted(slots)}, expected "
                    f"{sorted(expected_slots)}"
                )

    @property
    def dispatch_token_bytes(self) -> int:
        return self.hidden * dtype_bytes(self.dispatch_dtype)

    @property
    def combine_token_bytes(self) -> int:
        return self.hidden * dtype_bytes(self.combine_dtype)

    @property
    def expert_weight_bytes(self) -> int:
        return 3 * self.hidden * self.ffn_hidden * dtype_bytes(self.weight_dtype)

    def sorted_assignments(self) -> tuple[RoutingAssignment, ...]:
        return tuple(sorted(self.assignments))

    def assignments_by_token(
        self,
    ) -> dict[tuple[int, int], tuple[RoutingAssignment, ...]]:
        grouped: dict[tuple[int, int], list[RoutingAssignment]] = {}
        for assignment in self.sorted_assignments():
            grouped.setdefault(
                (assignment.src_rank, assignment.token_id), []
            ).append(assignment)
        return {key: tuple(value) for key, value in grouped.items()}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    ffn_hidden: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    topk: int
    sequence_length: int
    num_layers: int = 1
    micro_batches: int = 1
    batch_size: int = 1
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden,
            self.ffn_hidden,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_experts,
            self.topk,
            self.sequence_length,
            self.num_layers,
            self.micro_batches,
            self.batch_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValidationError("all model dimensions and counts must be positive")
        if self.topk > self.num_experts:
            raise ValidationError("model topk must not exceed num_experts")
        dtype_bytes(self.dtype)


def make_uniform_assignments(
    tokens_per_source_rank: Iterable[int],
    topk: int,
    num_experts: int,
) -> tuple[RoutingAssignment, ...]:
    if topk <= 0 or topk > num_experts:
        raise ValidationError("topk must be in [1, num_experts]")

    expert_order = list(range(num_experts))
    random.Random(0).shuffle(expert_order)
    assignments: list[RoutingAssignment] = []
    global_token_id = 0
    for src_rank, token_count in enumerate(tokens_per_source_rank):
        for token_id in range(token_count):
            first_slot = (global_token_id * topk) % num_experts
            for slot in range(topk):
                assignments.append(
                    RoutingAssignment(
                        src_rank=src_rank,
                        token_id=token_id,
                        topk_slot=slot,
                        expert_id=expert_order[(first_slot + slot) % num_experts],
                        route_weight=1.0 / topk,
                    )
                )
            global_token_id += 1
    return tuple(assignments)


def make_contiguous_expert_placement(
    num_experts: int,
    num_ranks: int,
) -> tuple[int, ...]:
    if num_experts <= 0:
        raise ValidationError("num_experts must be positive")
    if num_ranks <= 0:
        raise ValidationError("num_ranks must be positive")
    return tuple(
        min(num_ranks - 1, expert_id * num_ranks // num_experts)
        for expert_id in range(num_experts)
    )
