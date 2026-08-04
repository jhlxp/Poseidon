from __future__ import annotations

from dataclasses import asdict, dataclass

from .schema import ValidationError


@dataclass(frozen=True)
class ComputeEstimate:
    operation_flops: int
    duration_us: float
    overlaps_communication: bool
    available_sms: int
    peak_flops_per_second: float
    source: str


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
        self, operation_flops: int, *, overlaps_communication: bool = False
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
