from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..cost import ComputeCostModel
from ..gate import GateProvider
from ..graph import Task, TaskGraph
from ..schema import ValidationError
from .transformer import TransformerWorkloadConfig, build_transformer_workload


@dataclass(frozen=True)
class IncrementalLayerResult:
    layer: int
    new_task_keys: tuple[str, ...]
    deferred_final_keys: tuple[str, ...]
    algorithm_metadata: tuple[dict[str, object], ...]


class _LayerGateProvider:
    def __init__(self, provider: GateProvider, layer: int) -> None:
        self._provider = provider
        self._layer = layer

    def sample(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        kwargs["layer_id"] = self._layer
        return self._provider.sample(**kwargs)


def _rename_prefixed(value: str | None, layer: int) -> str | None:
    if value is None:
        return None
    for micro_batch in (0, 1):
        for separator in (".", ":"):
            old = f"mb{micro_batch}{separator}"
            if value.startswith(old):
                return (
                    f"mb{micro_batch}.layer{layer}{separator}"
                    + value[len(old):]
                )
    return value


def _participant_ranks(task: Task) -> set[int]:
    if task.kind != "transfer" or task.src_rank is None or task.dst_rank is None:
        return set()
    return {task.src_rank, task.dst_rank}


class IncrementalTransformerWorkloadBuilder:
    """Build one layer at a time while preserving the two-stream wavefront."""

    def __init__(
        self,
        config: TransformerWorkloadConfig,
        *,
        cost_model: ComputeCostModel,
    ) -> None:
        if config.model.micro_batches != 2:
            raise ValidationError(
                "incremental builder currently requires exactly two microbatches"
            )
        if config.algorithm != "probeep":
            raise ValidationError("incremental feedback builder currently targets ProbeEP")
        self.config = config
        self.cost_model = cost_model
        self.graph = TaskGraph(config.model.name, config.placement.num_ranks)
        self._next_layer = 0
        self._layer_results: list[IncrementalLayerResult] = []
        self._layer_metadata: list[dict[str, object]] = []

    @property
    def next_layer(self) -> int:
        return self._next_layer

    @property
    def layer_results(self) -> tuple[IncrementalLayerResult, ...]:
        return tuple(self._layer_results)

    def build_next_layer(
        self,
        budgets_by_microbatch: dict[int, tuple[int, ...]],
    ) -> IncrementalLayerResult:
        layer = self._next_layer
        if layer >= self.config.model.num_layers:
            raise ValidationError("all configured layers have already been built")
        if set(budgets_by_microbatch) != {0, 1}:
            raise ValidationError("incremental ProbeEP layer needs MB0 and MB1 budgets")

        one_layer_config = replace(
            self.config,
            model=replace(
                self.config.model,
                num_layers=1,
                name=f"{self.config.model.name}.layer{layer}",
            ),
            gate_provider=_LayerGateProvider(self.config.gate_provider, layer),
            probeep_nic_budget_by_dispatch={
                (micro_batch, 0): budgets
                for micro_batch, budgets in budgets_by_microbatch.items()
            },
        )
        isolated = build_transformer_workload(
            one_layer_config, cost_model=self.cost_model
        )
        key_mapping = {
            task.key: task.key.replace(
                f"mb{task.metadata.get('micro_batch')}.",
                f"mb{task.metadata.get('micro_batch')}.layer{layer}.",
                1,
            )
            for task in isolated.graph.tasks
        }
        new_keys: list[str] = []
        for source in isolated.graph.tasks:
            task = deepcopy(source)
            task.key = key_mapping[source.key]
            task.predecessors = {key_mapping[key] for key in source.predecessors}
            task.barrier_group = _rename_prefixed(task.barrier_group, layer)
            task.communication_phase_id = _rename_prefixed(
                task.communication_phase_id, layer
            )
            for field in ("stream_phase_id", "communication_stage_id"):
                current = task.metadata.get(field)
                if isinstance(current, str):
                    task.metadata[field] = _rename_prefixed(current, layer)
            task.metadata["layer"] = layer
            self.graph.add_task_copy(task)
            new_keys.append(task.key)

        self._connect_layer_wavefront(layer)
        is_final_layer = layer + 1 == self.config.model.num_layers
        deferred = () if is_final_layer else self._detach_mb1_final(layer)
        self.graph.validate()

        algorithm_metadata: list[dict[str, object]] = []
        for item in isolated.metadata["micro_batch_algorithms"]:
            adjusted = deepcopy(item)
            adjusted["layer"] = layer
            adjusted["invocation_index"] = layer * 2 + int(adjusted["micro_batch"])
            adjusted["feedback_mode"] = "in_process_dynamic_dispatch_observation"
            algorithm_metadata.append(adjusted)
        result = IncrementalLayerResult(
            layer=layer,
            new_task_keys=tuple(new_keys),
            deferred_final_keys=deferred,
            algorithm_metadata=tuple(algorithm_metadata),
        )
        self._layer_results.append(result)
        self._layer_metadata.extend(algorithm_metadata)
        self._next_layer += 1
        return result

    def metadata(self) -> dict[str, object]:
        if not self._layer_results:
            raise ValidationError("incremental builder has no layers")
        model = self.config.model
        first_gate = self._layer_metadata[0]["gate"]
        return {
            "generator": "moe_dag_dynamic_v1",
            "algorithm": self.config.algorithm,
            "model": {
                "name": model.name,
                "hidden": model.hidden,
                "ffn_hidden": model.ffn_hidden,
                "num_attention_heads": model.num_attention_heads,
                "num_kv_heads": model.num_kv_heads,
                "head_dim": model.head_dim,
                "num_experts": model.num_experts,
                "topk": model.topk,
                "sequence_length": model.sequence_length,
                "num_layers": model.num_layers,
                "micro_batches": model.micro_batches,
                "batch_size": model.batch_size,
                "dtype": model.dtype,
            },
            "placement": {
                "num_ranks": self.config.placement.num_ranks,
                "gpus_per_server": self.config.placement.gpus_per_server,
                "expert_to_rank": list(self.config.placement.expert_to_rank),
            },
            "tokens_per_rank": self.config.tokens_per_rank,
            "chunk_tokens": self.config.chunk_tokens,
            "token_padding": self.config.token_padding,
            "routing_provider": {
                "name": first_gate["name"],
                "random_seed": first_gate["seed"],
                "routing_fidelity": first_gate["routing_fidelity"],
                "parameters": first_gate["parameters"],
            },
            "compute_cost": self.cost_model.manifest(),
            "stream_schedule": {
                "model": "per_rank_two_stream_dynamic_wavefront_v1",
                "compute_streams_per_rank": 1,
                "communication_streams_per_rank": 1,
                "layer_count": len(self._layer_results),
                "microbatch_group_size": 2,
                "stream_order_lowering": "immutable_predecessor_edges",
                "deferred_mb1_final": True,
                "cpu_task_count": 0,
                "cpu_streams_global": 0,
            },
            "dynamic_dag": {
                "protocol": "stdin_stdout_append_v1",
                "htsim_process_count": 1,
                "append_unit": "previous_mb1_final_plus_current_layer_body",
                "feedback_source": "in_process_dynamic_dispatch_observation",
            },
            "micro_batch_algorithms": sorted(
                self._layer_metadata,
                key=lambda item: (int(item["layer"]), int(item["micro_batch"])),
            ),
        }

    def _tasks(
        self,
        layer: int,
        micro_batch: int,
        *,
        operation: str | None = None,
        phase: str | None = None,
    ) -> list[Task]:
        result = []
        for task in self.graph.tasks:
            if task.metadata.get("layer") != layer:
                continue
            if task.metadata.get("micro_batch") != micro_batch:
                continue
            if operation is not None and task.metadata.get("operation") != operation:
                continue
            if phase is not None and task.metadata.get("stream_phase") != phase:
                continue
            result.append(task)
        return result

    def _by_rank(self, tasks: list[Task]) -> dict[int, Task]:
        result: dict[int, Task] = {}
        for task in tasks:
            if task.rank is None:
                raise ValidationError(f"compute task {task.key} has no rank")
            result[task.rank] = task
        if len(result) != self.graph.num_ranks:
            raise ValidationError("incremental compute section misses ranks")
        return result

    def _connect_layer_wavefront(self, layer: int) -> None:
        if layer == 0:
            return
        previous_final0 = self._by_rank(
            self._tasks(layer - 1, 0, operation="combine_reduce")
        )
        previous_final1 = self._by_rank(
            self._tasks(layer - 1, 1, operation="combine_reduce")
        )
        current_attention0 = self._by_rank(
            self._tasks(layer, 0, operation="attention")
        )
        current_router0 = self._by_rank(
            self._tasks(layer, 0, operation="router_projection")
        )
        current_attention1 = self._by_rank(
            self._tasks(layer, 1, operation="attention")
        )
        for rank in range(self.graph.num_ranks):
            self._mark_compute_overlap(current_attention0[rank])
            self._mark_compute_overlap(current_router0[rank])
            self.graph.add_dependency(
                current_attention0[rank].key, previous_final0[rank].key
            )
            self.graph.add_dependency(
                previous_final1[rank].key, current_router0[rank].key
            )
            self.graph.add_dependency(
                current_attention1[rank].key, previous_final1[rank].key
            )

        previous_tail_by_rank: dict[int, list[Task]] = {
            rank: [] for rank in range(self.graph.num_ranks)
        }
        for task in self._tasks(layer - 1, 1, phase="combine"):
            for rank in _participant_ranks(task):
                previous_tail_by_rank[rank].append(task)
        current_stage = [
            task
            for task in self.graph.tasks
            if task.metadata.get("layer") == layer
            and task.metadata.get("micro_batch") == 0
            and task.metadata.get("communication_stage_id")
            == f"mb0.layer{layer}.weight_dispatch"
        ]
        for task in current_stage:
            for rank in _participant_ranks(task):
                for predecessor in previous_tail_by_rank[rank]:
                    self.graph.add_dependency(task.key, predecessor.key)

    def _mark_compute_overlap(self, task: Task) -> None:
        operation = task.metadata.get("operation")
        token_count = task.metadata.get("compute_token_count")
        if not isinstance(operation, str):
            raise ValidationError(f"compute task {task.key} has no operation")
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise ValidationError(f"compute task {task.key} has no token count")
        estimate = self.cost_model.estimate(
            task.operation_flops,
            operation=operation,
            overlaps_communication=True,
            token_count=token_count,
        )
        task.duration_us = estimate.duration_us
        task.overlaps_communication = True
        task.available_sms = estimate.available_sms
        task.peak_flops_per_second = estimate.peak_flops_per_second
        task.metadata["cost_source"] = estimate.source
        task.metadata["compute_us_per_token"] = estimate.us_per_token
        task.metadata["compute_token_kind"] = estimate.token_kind

    def _detach_mb1_final(self, layer: int) -> tuple[str, ...]:
        final0 = self._by_rank(
            self._tasks(layer, 0, operation="combine_reduce")
        )
        final1 = self._by_rank(
            self._tasks(layer, 1, operation="combine_reduce")
        )
        for rank in range(self.graph.num_ranks):
            final1[rank].predecessors.discard(final0[rank].key)
        return tuple(final1[rank].key for rank in range(self.graph.num_ranks))
