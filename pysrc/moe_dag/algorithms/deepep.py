from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from ..cost import H100CostModel
from ..graph import TaskGraph
from ..schema import MoEInvocation, ValidationError
from .common import AlgorithmBuildResult, chunked


Token = tuple[int, int]


@dataclass(frozen=True)
class DeepEPConfig:
    mode: Literal["hybrid", "direct"] = "hybrid"
    chunk_tokens: int = 128
    overlap_expert_compute: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"hybrid", "direct"}:
            raise ValidationError(f"unsupported DeepEP mode: {self.mode}")
        if self.chunk_tokens <= 0:
            raise ValidationError("chunk_tokens must be positive")


class DeepEPBuilder:
    def __init__(self, cost_model: H100CostModel, config: DeepEPConfig) -> None:
        self.cost_model = cost_model
        self.config = config

    def build(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        *,
        entry_keys: set[str] | None = None,
    ) -> AlgorithmBuildResult:
        if graph.num_ranks != invocation.placement.num_ranks:
            raise ValidationError("graph and placement rank counts differ")
        roots = set(entry_keys or ())
        before = len(graph.tasks)

        tokens_by_src_dst_rank: dict[tuple[int, int], set[Token]] = defaultdict(set)
        route_count_by_rank: dict[int, int] = defaultdict(int)
        for assignment in invocation.sorted_assignments():
            dst_rank = invocation.placement.expert_rank(assignment.expert_id)
            token = (assignment.src_rank, assignment.token_id)
            tokens_by_src_dst_rank[(assignment.src_rank, dst_rank)].add(token)
            route_count_by_rank[dst_rank] += 1

        dispatch_arrivals: dict[int, set[str]] = defaultdict(set)
        if self.config.mode == "direct":
            self._build_direct_dispatch(
                graph,
                invocation,
                roots,
                tokens_by_src_dst_rank,
                dispatch_arrivals,
            )
        else:
            self._build_hybrid_dispatch(
                graph,
                invocation,
                roots,
                tokens_by_src_dst_rank,
                dispatch_arrivals,
            )

        expert_keys: dict[int, str] = {}
        for rank, route_count in sorted(route_count_by_rank.items()):
            key = f"{invocation.invocation_id}.expert.rank{rank}"
            flops = route_count * 6 * invocation.hidden * invocation.ffn_hidden
            graph.add_compute(
                key,
                rank,
                self.cost_model.estimate(
                    flops,
                    overlaps_communication=self.config.overlap_expert_compute,
                ),
                predecessors=roots | dispatch_arrivals[rank],
                metadata={
                    "algorithm": "deepep",
                    "operation": "expert_ffn",
                    "real_token_routes": route_count,
                },
            )
            expert_keys[rank] = key

        combine_arrivals: dict[int, set[str]] = defaultdict(set)
        if self.config.mode == "direct":
            self._build_direct_combine(
                graph,
                invocation,
                tokens_by_src_dst_rank,
                expert_keys,
                combine_arrivals,
            )
        else:
            self._build_hybrid_combine(
                graph,
                invocation,
                tokens_by_src_dst_rank,
                expert_keys,
                combine_arrivals,
            )

        rank_terminals: dict[int, frozenset[str]] = {}
        terminal_keys: set[str] = set()
        assignments_by_token = invocation.assignments_by_token()
        for src_rank, token_count in enumerate(invocation.tokens_per_source_rank):
            if token_count == 0:
                continue
            relevant_experts = {
                invocation.placement.expert_rank(assignment.expert_id)
                for token, assignments in assignments_by_token.items()
                if token[0] == src_rank
                for assignment in assignments
            }
            predecessors = combine_arrivals[src_rank] | {
                expert_keys[rank]
                for rank in relevant_experts
                if rank in expert_keys
                and invocation.placement.rank_server(rank)
                == invocation.placement.rank_server(src_rank)
            }
            key = f"{invocation.invocation_id}.combine_reduce.rank{src_rank}"
            partials = sum(
                len(
                    {
                        invocation.placement.rank_server(
                            invocation.placement.expert_rank(item.expert_id)
                        )
                        for item in assignments
                    }
                )
                for token, assignments in assignments_by_token.items()
                if token[0] == src_rank
            )
            reduction_flops = max(
                1, partials * invocation.hidden * 2
            )
            graph.add_compute(
                key,
                src_rank,
                self.cost_model.estimate(reduction_flops),
                predecessors=predecessors,
                metadata={
                    "algorithm": "deepep",
                    "operation": "combine_final_reduce",
                    "token_count": token_count,
                    "node_partials": partials,
                },
            )
            rank_terminals[src_rank] = frozenset({key})
            terminal_keys.add(key)

        created = graph.tasks[before:]
        transfer_bytes: dict[str, int] = defaultdict(int)
        for task in created:
            if task.kind == "transfer":
                transfer_bytes[task.payload_kind or "unspecified"] += task.transfer_bytes
        return AlgorithmBuildResult(
            algorithm=f"deepep_{self.config.mode}",
            terminal_keys=frozenset(terminal_keys),
            rank_terminal_keys=rank_terminals,
            metadata={
                "mode": self.config.mode,
                "chunk_tokens": self.config.chunk_tokens,
                "transfer_bytes_by_payload": dict(sorted(transfer_bytes.items())),
                "created_tasks": len(created),
            },
        )

    def _build_direct_dispatch(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        roots: set[str],
        tokens_by_src_dst_rank: dict[tuple[int, int], set[Token]],
        arrivals: dict[int, set[str]],
    ) -> None:
        for (src, dst), tokens in sorted(tokens_by_src_dst_rank.items()):
            if src == dst:
                continue
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.dispatch.direct."
                    f"src{src}.dst{dst}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    src,
                    dst,
                    len(token_chunk) * invocation.dispatch_token_bytes,
                    "dispatch_hidden",
                    f"{invocation.invocation_id}:dispatch:rank:{src}",
                    predecessors=roots,
                    chunk_id=chunk_id,
                    metadata={"tokens": list(token_chunk), "hop": "direct"},
                )
                arrivals[dst].add(key)

    def _build_hybrid_dispatch(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        roots: set[str],
        tokens_by_src_dst_rank: dict[tuple[int, int], set[Token]],
        arrivals: dict[int, set[str]],
    ) -> None:
        placement = invocation.placement
        by_src_server: dict[tuple[int, int], set[Token]] = defaultdict(set)
        for (src, dst), tokens in tokens_by_src_dst_rank.items():
            by_src_server[(src, placement.rank_server(dst))].update(tokens)

        for (src, dst_server), tokens in sorted(by_src_server.items()):
            src_server = placement.rank_server(src)
            target_ranks = sorted(
                dst
                for (item_src, dst), item_tokens in tokens_by_src_dst_rank.items()
                if item_src == src
                and placement.rank_server(dst) == dst_server
                and item_tokens
            )
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                chunk_set = set(token_chunk)
                if dst_server == src_server:
                    for dst in target_ranks:
                        rank_tokens = chunk_set & tokens_by_src_dst_rank[(src, dst)]
                        if not rank_tokens or dst == src:
                            continue
                        key = (
                            f"{invocation.invocation_id}.dispatch.local."
                            f"src{src}.dst{dst}.chunk{chunk_id}"
                        )
                        graph.add_transfer(
                            key,
                            src,
                            dst,
                            len(rank_tokens) * invocation.dispatch_token_bytes,
                            "dispatch_local",
                            f"{invocation.invocation_id}:dispatch:rank:{src}",
                            predecessors=roots,
                            chunk_id=chunk_id,
                            metadata={"tokens": sorted(rank_tokens), "hop": "local"},
                        )
                        arrivals[dst].add(key)
                    continue

                relay = placement.server_rank(dst_server, placement.rank_local(src))
                fabric_key = (
                    f"{invocation.invocation_id}.dispatch.fabric."
                    f"src{src}.server{dst_server}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    fabric_key,
                    src,
                    relay,
                    len(token_chunk) * invocation.dispatch_token_bytes,
                    "dispatch_fabric",
                    f"{invocation.invocation_id}:dispatch:rank:{src}",
                    predecessors=roots,
                    chunk_id=chunk_id,
                    metadata={
                        "tokens": list(token_chunk),
                        "hop": "scale_out",
                        "destination_server": dst_server,
                    },
                )
                for dst in target_ranks:
                    rank_tokens = chunk_set & tokens_by_src_dst_rank[(src, dst)]
                    if not rank_tokens:
                        continue
                    if dst == relay:
                        arrivals[dst].add(fabric_key)
                        continue
                    local_key = (
                        f"{invocation.invocation_id}.dispatch.fanout."
                        f"relay{relay}.dst{dst}.src{src}.chunk{chunk_id}"
                    )
                    graph.add_transfer(
                        local_key,
                        relay,
                        dst,
                        len(rank_tokens) * invocation.dispatch_token_bytes,
                        "dispatch_local_fanout",
                        f"{invocation.invocation_id}:dispatch:rank:{src}",
                        predecessors={fabric_key},
                        chunk_id=chunk_id,
                        metadata={
                            "tokens": sorted(rank_tokens),
                            "hop": "scale_up_fanout",
                            "origin_rank": src,
                        },
                    )
                    arrivals[dst].add(local_key)

    def _build_direct_combine(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        tokens_by_src_dst_rank: dict[tuple[int, int], set[Token]],
        expert_keys: dict[int, str],
        arrivals: dict[int, set[str]],
    ) -> None:
        for (origin, execution_rank), tokens in sorted(tokens_by_src_dst_rank.items()):
            if execution_rank == origin:
                continue
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                key = (
                    f"{invocation.invocation_id}.combine.direct."
                    f"src{execution_rank}.dst{origin}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    key,
                    execution_rank,
                    origin,
                    len(token_chunk) * invocation.combine_token_bytes,
                    "combine_partial",
                    f"{invocation.invocation_id}:combine:rank:{origin}",
                    predecessors={expert_keys[execution_rank]},
                    chunk_id=chunk_id,
                    metadata={"tokens": list(token_chunk), "hop": "direct"},
                )
                arrivals[origin].add(key)

    def _build_hybrid_combine(
        self,
        graph: TaskGraph,
        invocation: MoEInvocation,
        tokens_by_src_dst_rank: dict[tuple[int, int], set[Token]],
        expert_keys: dict[int, str],
        arrivals: dict[int, set[str]],
    ) -> None:
        placement = invocation.placement
        by_origin_server: dict[tuple[int, int], set[Token]] = defaultdict(set)
        for (origin, execution_rank), tokens in tokens_by_src_dst_rank.items():
            by_origin_server[(origin, placement.rank_server(execution_rank))].update(tokens)

        for (origin, execution_server), tokens in sorted(by_origin_server.items()):
            origin_server = placement.rank_server(origin)
            execution_ranks = sorted(
                rank
                for (item_origin, rank), item_tokens in tokens_by_src_dst_rank.items()
                if item_origin == origin
                and placement.rank_server(rank) == execution_server
                and item_tokens
            )
            for chunk_id, token_chunk in enumerate(
                chunked(sorted(tokens), self.config.chunk_tokens)
            ):
                chunk_set = set(token_chunk)
                if execution_server == origin_server:
                    for rank in execution_ranks:
                        rank_tokens = chunk_set & tokens_by_src_dst_rank[(origin, rank)]
                        if not rank_tokens or rank == origin:
                            continue
                        key = (
                            f"{invocation.invocation_id}.combine.local."
                            f"src{rank}.dst{origin}.chunk{chunk_id}"
                        )
                        graph.add_transfer(
                            key,
                            rank,
                            origin,
                            len(rank_tokens) * invocation.combine_token_bytes,
                            "combine_local",
                            f"{invocation.invocation_id}:combine:rank:{origin}",
                            predecessors={expert_keys[rank]},
                            chunk_id=chunk_id,
                            metadata={"tokens": sorted(rank_tokens), "hop": "local"},
                        )
                        arrivals[origin].add(key)
                    continue

                relay = placement.server_rank(
                    execution_server, placement.rank_local(origin)
                )
                gather_dependencies: set[str] = set()
                for rank in execution_ranks:
                    rank_tokens = chunk_set & tokens_by_src_dst_rank[(origin, rank)]
                    if not rank_tokens:
                        continue
                    if rank == relay:
                        gather_dependencies.add(expert_keys[rank])
                        continue
                    gather_key = (
                        f"{invocation.invocation_id}.combine.gather."
                        f"src{rank}.relay{relay}.origin{origin}.chunk{chunk_id}"
                    )
                    graph.add_transfer(
                        gather_key,
                        rank,
                        relay,
                        len(rank_tokens) * invocation.combine_token_bytes,
                        "combine_local_gather",
                        f"{invocation.invocation_id}:combine:rank:{origin}",
                        predecessors={expert_keys[rank]},
                        chunk_id=chunk_id,
                        metadata={
                            "tokens": sorted(rank_tokens),
                            "hop": "scale_up_gather",
                            "origin_rank": origin,
                        },
                    )
                    gather_dependencies.add(gather_key)

                fabric_key = (
                    f"{invocation.invocation_id}.combine.fabric."
                    f"server{execution_server}.dst{origin}.chunk{chunk_id}"
                )
                graph.add_transfer(
                    fabric_key,
                    relay,
                    origin,
                    len(token_chunk) * invocation.combine_token_bytes,
                    "combine_fabric_partial",
                    f"{invocation.invocation_id}:combine:rank:{origin}",
                    predecessors=gather_dependencies,
                    chunk_id=chunk_id,
                    metadata={
                        "tokens": list(token_chunk),
                        "hop": "scale_out_return",
                        "execution_server": execution_server,
                    },
                )
                arrivals[origin].add(fabric_key)
