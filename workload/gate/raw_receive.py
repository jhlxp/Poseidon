from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from moe_dag.gate import GateSample
from moe_dag.schema import Placement, ValidationError

from .base import sample_exact_global_quotas


@dataclass(frozen=True)
class RawReceiveDataset:
    placement_path: Path
    csv_pattern: str
    num_ranks: int
    num_layers: int
    slots_per_rank: int
    num_logical_experts: int
    logical_loads: tuple[tuple[int, ...], ...]
    total_receive_by_layer: tuple[int, ...]

    @classmethod
    def load(
        cls,
        placement_path: Path | str,
        *,
        csv_pattern: str = "decode_{rank}.csv",
    ) -> "RawReceiveDataset":
        placement_path = Path(placement_path).resolve()
        if not placement_path.is_file():
            raise ValidationError(f"missing raw placement JSON: {placement_path}")
        try:
            root = json.loads(placement_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid raw placement JSON: {placement_path}") from exc
        layers = root.get("layer_list") if isinstance(root, dict) else None
        if not isinstance(layers, list) or not layers:
            raise ValidationError("raw placement JSON needs a non-empty layer_list")
        if root.get("moe_layer_count") != len(layers):
            raise ValidationError("raw placement moe_layer_count disagrees with layer_list")

        first_devices = layers[0].get("device_list")
        if not isinstance(first_devices, list) or not first_devices:
            raise ValidationError("raw placement layer needs a device_list")
        num_ranks = len(first_devices)
        device_zero = first_devices[0]
        first_slots = device_zero.get("device_expert") if isinstance(device_zero, dict) else None
        if not isinstance(first_slots, list) or not first_slots:
            raise ValidationError("raw placement device needs device_expert slots")
        slots_per_rank = len(first_slots)

        mappings: list[list[list[int]]] = []
        expert_ids: set[int] = set()
        for expected_layer, layer in enumerate(layers):
            if layer.get("layer_id") != expected_layer:
                raise ValidationError("raw placement layer IDs must be contiguous")
            devices = layer.get("device_list")
            if not isinstance(devices, list) or len(devices) != num_ranks:
                raise ValidationError("raw placement device count changed across layers")
            by_rank: list[list[int] | None] = [None] * num_ranks
            for device in devices:
                rank = device.get("device_id") if isinstance(device, dict) else None
                slots = device.get("device_expert") if isinstance(device, dict) else None
                if not isinstance(rank, int) or rank < 0 or rank >= num_ranks:
                    raise ValidationError("raw placement has invalid device_id")
                if by_rank[rank] is not None:
                    raise ValidationError("raw placement has duplicate device_id")
                if not isinstance(slots, list) or len(slots) != slots_per_rank:
                    raise ValidationError("raw placement slot count changed across devices")
                if any(not isinstance(expert, int) or expert < 0 for expert in slots):
                    raise ValidationError("raw placement expert IDs must be non-negative integers")
                by_rank[rank] = list(slots)
                expert_ids.update(slots)
            if any(slots is None for slots in by_rank):
                raise ValidationError("raw placement is missing a device")
            mappings.append([slots for slots in by_rank if slots is not None])

        if expert_ids != set(range(max(expert_ids) + 1)):
            raise ValidationError("raw placement logical expert IDs must be contiguous")
        num_logical_experts = max(expert_ids) + 1

        csv_rows: list[list[list[int]]] = []
        for rank in range(num_ranks):
            path = placement_path.parent / csv_pattern.format(rank=rank)
            if not path.is_file():
                raise ValidationError(f"missing raw receive CSV: {path}")
            rows: list[list[int]] = []
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row_index, row in enumerate(csv.reader(handle)):
                        if len(row) != slots_per_rank:
                            raise ValidationError(
                                f"{path}:{row_index + 1}: expected {slots_per_rank} columns"
                            )
                        try:
                            values = [int(value) for value in row]
                        except ValueError as exc:
                            raise ValidationError(
                                f"{path}:{row_index + 1}: counts must be integers"
                            ) from exc
                        if any(value < 0 for value in values):
                            raise ValidationError(
                                f"{path}:{row_index + 1}: counts must be non-negative"
                            )
                        rows.append(values)
            except OSError as exc:
                raise ValidationError(f"failed to read raw receive CSV: {path}") from exc
            if len(rows) != len(layers):
                raise ValidationError(
                    f"{path}: expected {len(layers)} rows, found {len(rows)}"
                )
            csv_rows.append(rows)

        logical_loads: list[tuple[int, ...]] = []
        totals: list[int] = []
        for layer in range(len(layers)):
            loads = [0] * num_logical_experts
            physical_total = 0
            for rank in range(num_ranks):
                for slot in range(slots_per_rank):
                    count = csv_rows[rank][layer][slot]
                    expert = mappings[layer][rank][slot]
                    loads[expert] += count
                    physical_total += count
            if sum(loads) != physical_total:
                raise ValidationError("raw physical-to-logical folding lost receive counts")
            logical_loads.append(tuple(loads))
            totals.append(physical_total)

        return cls(
            placement_path=placement_path,
            csv_pattern=csv_pattern,
            num_ranks=num_ranks,
            num_layers=len(layers),
            slots_per_rank=slots_per_rank,
            num_logical_experts=num_logical_experts,
            logical_loads=tuple(logical_loads),
            total_receive_by_layer=tuple(totals),
        )


@dataclass(frozen=True)
class RawReceiveCDFGateProvider:
    dataset: RawReceiveDataset
    seed: int = 0
    layer_map: tuple[int, ...] | None = None
    name: str = "raw_receive_cdf"

    def _raw_layer(self, layer_id: int) -> int:
        if layer_id < 0:
            raise ValidationError("model layer ID must be non-negative")
        if self.layer_map is None:
            raw_layer = layer_id
        else:
            if layer_id >= len(self.layer_map):
                raise ValidationError("gate layer_map does not cover the model layer")
            raw_layer = self.layer_map[layer_id]
        if raw_layer < 0 or raw_layer >= self.dataset.num_layers:
            raise ValidationError(f"raw layer {raw_layer} is outside the dataset")
        return raw_layer

    def sample(
        self,
        *,
        layer_id: int,
        microbatch_id: int,
        tokens_per_source_rank: tuple[int, ...],
        placement: Placement,
        topk: int,
    ) -> GateSample:
        if placement.num_ranks != self.dataset.num_ranks:
            raise ValidationError("raw receive rank count differs from model placement")
        if placement.num_experts != self.dataset.num_logical_experts:
            raise ValidationError("raw receive expert count differs from model placement")
        raw_layer = self._raw_layer(layer_id)
        return sample_exact_global_quotas(
            expert_weights=np.asarray(
                self.dataset.logical_loads[raw_layer], dtype=np.float64
            ),
            tokens_per_source_rank=tokens_per_source_rank,
            placement=placement,
            topk=topk,
            base_seed=self.seed,
            layer_id=layer_id,
            microbatch_id=microbatch_id,
            provider_name=self.name,
            provider_parameters={
                "placement_json": str(self.dataset.placement_path),
                "csv_pattern": self.dataset.csv_pattern,
                "source_label": "decode",
                "raw_layer": raw_layer,
                "raw_layer_total_receive": self.dataset.total_receive_by_layer[
                    raw_layer
                ],
                "physical_slots_per_rank": self.dataset.slots_per_rank,
                "sampling": "exact_global_logical_receive_quota",
            },
            routing_fidelity="quota_matched_global_receive_histogram",
        )
