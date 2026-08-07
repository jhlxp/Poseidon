from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Protocol

from .schema import ValidationError


@dataclass(frozen=True)
class ComputeEstimate:
    operation_flops: int
    duration_us: float
    overlaps_communication: bool
    available_sms: int
    peak_flops_per_second: float
    source: str


class ComputeCostModel(Protocol):
    communication_sms: int

    def estimate(
        self,
        operation_flops: int,
        *,
        operation: str,
        overlaps_communication: bool = False,
    ) -> ComputeEstimate: ...

    def manifest(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class H100CostModel:
    dense_bf16_tflops: float = 989.0
    total_sms: int = 132
    communication_sms: int = 20

    def __post_init__(self) -> None:
        if self.dense_bf16_tflops <= 0:
            raise ValidationError("dense_bf16_tflops must be positive")
        if self.total_sms <= 0:
            raise ValidationError("total_sms must be positive")
        if self.communication_sms < 0 or self.communication_sms >= self.total_sms:
            raise ValidationError("communication_sms must be in [0, total_sms)")

    @property
    def peak_flops_per_second(self) -> float:
        return self.dense_bf16_tflops * 1e12

    @property
    def overlap_sms(self) -> int:
        return self.total_sms - self.communication_sms

    @property
    def overlap_peak_flops_per_second(self) -> float:
        return self.peak_flops_per_second * self.overlap_sms / self.total_sms

    def estimate(
        self,
        operation_flops: int,
        *,
        operation: str = "unspecified",
        overlaps_communication: bool = False,
    ) -> ComputeEstimate:
        if operation_flops <= 0:
            raise ValidationError("operation_flops must be positive")
        available_sms = self.overlap_sms if overlaps_communication else self.total_sms
        peak = (
            self.overlap_peak_flops_per_second
            if overlaps_communication
            else self.peak_flops_per_second
        )
        return ComputeEstimate(
            operation_flops=operation_flops,
            duration_us=operation_flops / peak * 1e6,
            overlaps_communication=overlaps_communication,
            available_sms=available_sms,
            peak_flops_per_second=peak,
            source=(
                "h100_sxm_fixed_comm_sm_partition"
                if overlaps_communication
                else "h100_sxm_bf16_dense_peak"
            ),
        )

    def manifest(self) -> dict[str, float | int | str]:
        data = asdict(self)
        data.update(
            {
                "model": "h100_sxm_static_theoretical",
                "peak_flops_per_second": self.peak_flops_per_second,
                "overlap_sms": self.overlap_sms,
                "overlap_peak_flops_per_second": (
                    self.overlap_peak_flops_per_second
                ),
            }
        )
        return data


@dataclass(frozen=True)
class ModuleComputeTime:
    theoretical_us: float | None
    profiled_us: float | None


@dataclass(frozen=True)
class JsonComputeCostModel:
    config_path: Path
    hardware: str
    profile_scope: str
    selected_source: str
    modules: dict[str, ModuleComputeTime]
    total_sms: int
    communication_sms: int

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        selected_source: str | None = None,
    ) -> JsonComputeCostModel:
        config_path = path.resolve()
        if not config_path.is_file():
            raise ValidationError(f"compute config does not exist: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid compute config JSON {config_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValidationError("compute config root must be an object")
        if payload.get("schema_version") != 1:
            raise ValidationError("compute config schema_version must be 1")
        if payload.get("time_unit") != "us":
            raise ValidationError("compute config time_unit must be us")

        hardware = payload.get("hardware")
        profile_scope = payload.get("profile_scope")
        if not isinstance(hardware, str) or not hardware:
            raise ValidationError("compute config hardware must be a non-empty string")
        if not isinstance(profile_scope, str) or not profile_scope:
            raise ValidationError(
                "compute config profile_scope must be a non-empty string"
            )

        source = selected_source or payload.get("selected_source")
        if source not in {"theoretical", "profiled"}:
            raise ValidationError(
                "compute config selected_source must be theoretical or profiled"
            )

        total_sms = payload.get("total_sms")
        communication_sms = payload.get("communication_sms")
        if isinstance(total_sms, bool) or not isinstance(total_sms, int):
            raise ValidationError("compute config total_sms must be an integer")
        if (
            isinstance(communication_sms, bool)
            or not isinstance(communication_sms, int)
        ):
            raise ValidationError(
                "compute config communication_sms must be an integer"
            )
        if total_sms <= 0 or communication_sms < 0 or communication_sms >= total_sms:
            raise ValidationError(
                "compute config requires 0 <= communication_sms < total_sms"
            )

        raw_modules = payload.get("modules")
        if not isinstance(raw_modules, dict) or not raw_modules:
            raise ValidationError("compute config modules must be a non-empty object")
        modules: dict[str, ModuleComputeTime] = {}
        for operation, values in raw_modules.items():
            if not isinstance(operation, str) or not operation:
                raise ValidationError("compute module names must be non-empty strings")
            if not isinstance(values, dict):
                raise ValidationError(
                    f"compute module {operation} must be an object"
                )
            theoretical = cls._optional_positive_time(
                values.get("theoretical_us"), operation, "theoretical_us"
            )
            profiled = cls._optional_positive_time(
                values.get("profiled_us"), operation, "profiled_us"
            )
            modules[operation] = ModuleComputeTime(theoretical, profiled)
        return cls(
            config_path=config_path,
            hardware=hardware,
            profile_scope=profile_scope,
            selected_source=source,
            modules=modules,
            total_sms=total_sms,
            communication_sms=communication_sms,
        )

    @staticmethod
    def _optional_positive_time(
        value: object, operation: str, field: str
    ) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                f"compute module {operation}.{field} must be a number or null"
            )
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValidationError(
                f"compute module {operation}.{field} must be positive and finite"
            )
        return result

    @property
    def overlap_sms(self) -> int:
        return self.total_sms - self.communication_sms

    def estimate(
        self,
        operation_flops: int,
        *,
        operation: str,
        overlaps_communication: bool = False,
    ) -> ComputeEstimate:
        if operation_flops <= 0:
            raise ValidationError("operation_flops must be positive")
        try:
            module = self.modules[operation]
        except KeyError as exc:
            raise ValidationError(
                f"compute config has no module named {operation!r}"
            ) from exc
        duration_us = (
            module.theoretical_us
            if self.selected_source == "theoretical"
            else module.profiled_us
        )
        if duration_us is None:
            raise ValidationError(
                f"compute module {operation}.{self.selected_source}_us is null"
            )
        available_sms = self.overlap_sms if overlaps_communication else self.total_sms
        effective_peak = operation_flops / duration_us * 1e6
        return ComputeEstimate(
            operation_flops=operation_flops,
            duration_us=duration_us,
            overlaps_communication=overlaps_communication,
            available_sms=available_sms,
            peak_flops_per_second=effective_peak,
            source=(
                f"json_{self.selected_source}:{self.hardware}:{operation}"
            ),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "model": "json_fixed_module_duration_v1",
            "config_path": str(self.config_path),
            "hardware": self.hardware,
            "profile_scope": self.profile_scope,
            "selected_source": self.selected_source,
            "time_unit": "us",
            "total_sms": self.total_sms,
            "communication_sms": self.communication_sms,
            "overlap_sms": self.overlap_sms,
            "modules": {
                operation: {
                    "theoretical_us": values.theoretical_us,
                    "profiled_us": values.profiled_us,
                }
                for operation, values in sorted(self.modules.items())
            },
        }
