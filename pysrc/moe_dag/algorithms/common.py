from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, TypeVar


T = TypeVar("T")


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
