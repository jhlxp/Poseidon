#!/usr/bin/env python3
"""Plot MpRail link throughput from HTSim link-load sampler CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class PanelSpec:
    key: str
    layer: str
    direction: str
    title: str
    color: str


PANELS = (
    PanelSpec(
        "server_local",
        "server_local",
        "local",
        "Server-local FullMesh",
        "#23847A",
    ),
    PanelSpec("l0_l1_up", "l0_l1", "up", "L0 -> L1", "#3979A8"),
    PanelSpec("l1_l0_down", "l0_l1", "down", "L1 -> L0", "#B56A32"),
    PanelSpec("host_l0_up", "host_l0", "up", "Host -> L0", "#3979A8"),
    PanelSpec("l0_host_down", "host_l0", "down", "L0 -> Host", "#B56A32"),
)
PANEL_BY_KEY = {panel.key: panel for panel in PANELS}
DISPLAY_PANELS = (
    PANEL_BY_KEY["l0_l1_up"],
    PANEL_BY_KEY["l1_l0_down"],
    PANEL_BY_KEY["host_l0_up"],
    PANEL_BY_KEY["l0_host_down"],
    PANEL_BY_KEY["server_local"],
)


@dataclass(frozen=True)
class ParsedLink:
    link_id: int
    link_name: str
    panel: str
    layer: str
    direction: str
    rate_gbps: float
    src_rank: int | None = None
    dst_rank: int | None = None
    rail: int | None = None
    plane: int | None = None
    spine: int | None = None
    bundle: int | None = None


@dataclass(frozen=True)
class LinkSample:
    bucket: int
    time_us: float
    bytes_sent: int
    throughput_gbps: float
    max_queue_bytes: int


_PATTERNS = (
    (
        re.compile(
            r"^MPRAIL_LOCAL_MPRAIL_HOST_SRC_(?P<src_rank>\d+)"
            r"->MPRAIL_HOST_DST_(?P<dst_rank>\d+)\(b(?P<bundle>\d+)\)$"
        ),
        "server_local",
    ),
    (
        re.compile(
            r"^MPRAIL_HOST_SRC_(?P<src_rank>\d+)"
            r"->MPRAIL_L0_r(?P<rail>\d+)_p(?P<plane>\d+)"
            r"\(b(?P<bundle>\d+)\)$"
        ),
        "host_l0_up",
    ),
    (
        re.compile(
            r"^MPRAIL_L0_r(?P<rail>\d+)_p(?P<plane>\d+)"
            r"->MPRAIL_HOST_DST_(?P<dst_rank>\d+)\(b(?P<bundle>\d+)\)$"
        ),
        "l0_host_down",
    ),
    (
        re.compile(
            r"^MPRAIL_L0_r(?P<rail>\d+)_p(?P<src_plane>\d+)"
            r"->MPRAIL_L1_p(?P<dst_plane>\d+)_s(?P<spine>\d+)"
            r"\(b(?P<bundle>\d+)\)$"
        ),
        "l0_l1_up",
    ),
    (
        re.compile(
            r"^MPRAIL_L1_p(?P<src_plane>\d+)_s(?P<spine>\d+)"
            r"->MPRAIL_L0_r(?P<rail>\d+)_p(?P<dst_plane>\d+)"
            r"\(b(?P<bundle>\d+)\)$"
        ),
        "l1_l0_down",
    ),
)


def classify_mprail_link(
    link_id: int, link_name: str, rate_gbps: float
) -> ParsedLink:
    if link_id < 0:
        raise ValueError("link_id must be non-negative")
    if rate_gbps <= 0:
        raise ValueError(f"link {link_id} has non-positive rate_gbps")

    for pattern, panel_key in _PATTERNS:
        match = pattern.fullmatch(link_name)
        if match is None:
            continue
        values = {key: int(value) for key, value in match.groupdict().items()}
        if "src_plane" in values:
            if values["src_plane"] != values["dst_plane"]:
                raise ValueError(
                    f"MpRail link crosses planes: {link_name}"
                )
            values["plane"] = values.pop("src_plane")
            values.pop("dst_plane")
        panel = PANEL_BY_KEY[panel_key]
        return ParsedLink(
            link_id=link_id,
            link_name=link_name,
            panel=panel.key,
            layer=panel.layer,
            direction=panel.direction,
            rate_gbps=rate_gbps,
            **values,
        )
    raise ValueError(f"unrecognized MpRail link name: {link_name}")


def read_link_info(path: Path) -> dict[int, ParsedLink]:
    if not path.is_file():
        raise FileNotFoundError(f"missing link inventory: {path}")
    links: dict[int, ParsedLink] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"link_id", "link_name", "rate_gbps"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("link_name") or "").strip()
            if not name.startswith("MPRAIL_"):
                continue
            try:
                link_id = int(row["link_id"])
                rate_gbps = float(row["rate_gbps"])
                link = classify_mprail_link(link_id, name, rate_gbps)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if link_id in links:
                raise ValueError(f"{path}:{line_number}: duplicate link_id {link_id}")
            links[link_id] = link
    if not links:
        raise ValueError(f"{path}: no MpRail links found")
    return links


def read_link_samples(
    path: Path, links: dict[int, ParsedLink]
) -> tuple[dict[int, dict[int, LinkSample]], dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing link samples: {path}")
    samples: dict[int, dict[int, LinkSample]] = {}
    bucket_times: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "time_ms",
            "bucket",
            "link_id",
            "bytes",
            "throughput_gbps",
            "max_queue_bytes",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                link_id = int(row["link_id"])
                bucket = int(row["bucket"])
                time_us = float(row["time_ms"]) * 1000.0
                bytes_sent = int(row["bytes"])
                throughput = float(row["throughput_gbps"])
                max_queue = int(row["max_queue_bytes"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
            if link_id not in links:
                continue
            if bucket < 0 or time_us < 0 or bytes_sent < 0 or throughput < 0:
                raise ValueError(f"{path}:{line_number}: negative sample value")
            previous_time = bucket_times.setdefault(bucket, time_us)
            if not math.isclose(previous_time, time_us, abs_tol=1e-9):
                raise ValueError(f"{path}:{line_number}: inconsistent bucket time")
            per_link = samples.setdefault(link_id, {})
            if bucket in per_link:
                raise ValueError(
                    f"{path}:{line_number}: duplicate link/bucket sample "
                    f"{link_id}/{bucket}"
                )
            per_link[bucket] = LinkSample(
                bucket, time_us, bytes_sent, throughput, max_queue
            )
    if not bucket_times:
        raise ValueError(f"{path}: no MpRail samples found")
    return samples, bucket_times


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _selected_buckets(
    bucket_times: dict[int, float], x_min_us: float, x_max_us: float | None
) -> list[int]:
    return [
        bucket
        for bucket, time_us in sorted(bucket_times.items())
        if time_us >= x_min_us and (x_max_us is None or time_us <= x_max_us)
    ]


def write_inventory(path: Path, links: Iterable[ParsedLink]) -> None:
    fields = list(ParsedLink.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for link in sorted(links, key=lambda item: item.link_id):
            writer.writerow(asdict(link))


def write_summary(
    path: Path,
    links: dict[int, ParsedLink],
    samples: dict[int, dict[int, LinkSample]],
    selected_buckets: list[int],
) -> None:
    fields = [
        "panel",
        "layer",
        "direction",
        "discovered_links",
        "active_links",
        "active_samples",
        "line_rate_gbps",
        "total_bytes",
        "per_link_total_bytes_mean",
        "per_link_total_bytes_cv",
        "throughput_active_p50_gbps",
        "throughput_active_p99_gbps",
        "throughput_active_max_gbps",
        "max_queue_bytes",
    ]
    selected = set(selected_buckets)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for panel in PANELS:
            panel_links = [item for item in links.values() if item.panel == panel.key]
            active = [
                item
                for item in panel_links
                if any(bucket in selected for bucket in samples.get(item.link_id, {}))
            ]
            totals: list[float] = []
            throughputs: list[float] = []
            max_queue = 0
            for link in active:
                selected_samples = [
                    sample
                    for bucket, sample in samples.get(link.link_id, {}).items()
                    if bucket in selected
                ]
                totals.append(float(sum(item.bytes_sent for item in selected_samples)))
                throughputs.extend(item.throughput_gbps for item in selected_samples)
                max_queue = max(
                    max_queue,
                    max((item.max_queue_bytes for item in selected_samples), default=0),
                )
            mean_total = fmean(totals) if totals else 0.0
            cv_total = pstdev(totals) / mean_total if mean_total and len(totals) > 1 else 0.0
            rates = {item.rate_gbps for item in panel_links}
            writer.writerow(
                {
                    "panel": panel.key,
                    "layer": panel.layer,
                    "direction": panel.direction,
                    "discovered_links": len(panel_links),
                    "active_links": len(active),
                    "active_samples": len(throughputs),
                    "line_rate_gbps": (
                        ""
                        if not rates
                        else next(iter(rates))
                        if len(rates) == 1
                        else "multiple"
                    ),
                    "total_bytes": round(sum(totals)),
                    "per_link_total_bytes_mean": mean_total,
                    "per_link_total_bytes_cv": cv_total,
                    "throughput_active_p50_gbps": percentile(throughputs, 50),
                    "throughput_active_p99_gbps": percentile(throughputs, 99),
                    "throughput_active_max_gbps": max(throughputs, default=0.0),
                    "max_queue_bytes": max_queue,
                }
            )


def _endpoint_link_groups(
    links: dict[int, ParsedLink], direction: str
) -> dict[int, list[ParsedLink]]:
    if direction == "output":
        panel_key = "host_l0_up"
        rank_field = "src_rank"
    elif direction == "input":
        panel_key = "l0_host_down"
        rank_field = "dst_rank"
    else:
        raise ValueError(f"unknown endpoint direction: {direction}")

    groups: dict[int, list[ParsedLink]] = {}
    for link in links.values():
        if link.panel != panel_key:
            continue
        rank = getattr(link, rank_field)
        if rank is None:
            raise ValueError(f"link {link.link_id} is missing its endpoint rank")
        groups.setdefault(rank, []).append(link)
    return groups


def _endpoint_series(
    groups: dict[int, list[ParsedLink]],
    samples: dict[int, dict[int, LinkSample]],
    selected_buckets: list[int],
) -> dict[int, list[float]]:
    selected = set(selected_buckets)
    result: dict[int, list[float]] = {}
    for rank, rank_links in groups.items():
        if not any(
            bucket in selected
            for link in rank_links
            for bucket in samples.get(link.link_id, {})
        ):
            continue
        result[rank] = [
            sum(
                samples[link.link_id][bucket].throughput_gbps
                for link in rank_links
                if bucket in samples.get(link.link_id, {})
            )
            for bucket in selected_buckets
        ]
    return result


def write_endpoint_summary(
    path: Path,
    links: dict[int, ParsedLink],
    samples: dict[int, dict[int, LinkSample]],
    selected_buckets: list[int],
    configured_planes: int,
) -> None:
    fields = [
        "scope",
        "direction",
        "configured_planes",
        "discovered_endpoints",
        "active_endpoints",
        "per_plane_line_rate_gbps",
        "aggregate_line_rate_gbps",
        "total_bytes",
        "throughput_p50_gbps",
        "throughput_p99_gbps",
        "throughput_max_gbps",
        "mean_utilization",
        "peak_utilization",
        "peak_headroom",
    ]
    selected = set(selected_buckets)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for direction in ("output", "input"):
            groups = _endpoint_link_groups(links, direction)
            series = _endpoint_series(groups, samples, selected_buckets)
            values = [value for rank_values in series.values() for value in rank_values]
            rates = {
                link.rate_gbps
                for rank_links in groups.values()
                for link in rank_links
            }
            per_plane_rate = next(iter(rates)) if len(rates) == 1 else None
            aggregate_rate = (
                configured_planes * per_plane_rate
                if per_plane_rate is not None
                else None
            )
            total_bytes = sum(
                sample.bytes_sent
                for rank_links in groups.values()
                for link in rank_links
                for bucket, sample in samples.get(link.link_id, {}).items()
                if bucket in selected
            )
            mean_throughput = fmean(values) if values else 0.0
            peak_throughput = max(values, default=0.0)
            mean_utilization = (
                mean_throughput / aggregate_rate if aggregate_rate else math.nan
            )
            peak_utilization = (
                peak_throughput / aggregate_rate if aggregate_rate else math.nan
            )
            writer.writerow(
                {
                    "scope": "endpoint_rank_sum_over_planes",
                    "direction": direction,
                    "configured_planes": configured_planes,
                    "discovered_endpoints": len(groups),
                    "active_endpoints": len(series),
                    "per_plane_line_rate_gbps": (
                        per_plane_rate if per_plane_rate is not None else "multiple"
                    ),
                    "aggregate_line_rate_gbps": (
                        aggregate_rate if aggregate_rate is not None else "multiple"
                    ),
                    "total_bytes": total_bytes,
                    "throughput_p50_gbps": percentile(values, 50),
                    "throughput_p99_gbps": percentile(values, 99),
                    "throughput_max_gbps": peak_throughput,
                    "mean_utilization": mean_utilization,
                    "peak_utilization": peak_utilization,
                    "peak_headroom": 1.0 - peak_utilization,
                }
            )


def _plot_endpoint_aggregate(
    axis: plt.Axes,
    direction: str,
    color: str,
    links: dict[int, ParsedLink],
    samples: dict[int, dict[int, LinkSample]],
    selected_buckets: list[int],
    times: list[float],
    configured_planes: int,
) -> None:
    groups = _endpoint_link_groups(links, direction)
    series = _endpoint_series(groups, samples, selected_buckets)
    alpha = max(0.05, min(0.35, 2.5 / math.sqrt(max(len(series), 1))))
    observed_max = 0.0
    for values in series.values():
        observed_max = max(observed_max, max(values, default=0.0))
        axis.plot(times, values, color=color, alpha=alpha, linewidth=0.75)

    aggregate_rates = sorted(
        {
            configured_planes * link.rate_gbps
            for rank_links in groups.values()
            for link in rank_links
        }
    )
    for index, rate in enumerate(aggregate_rates):
        axis.axhline(
            rate,
            color="#D43D3D",
            linestyle="--",
            linewidth=1.1,
            label=(
                f"{configured_planes} planes x {rate / configured_planes:g} "
                f"= {rate:g} Gbps"
                if index == 0
                else f"aggregate line rate {rate:g} Gbps"
            ),
        )
    if aggregate_rates:
        axis.legend(loc="upper right", fontsize=8, frameon=True)
    if not series:
        axis.text(
            0.5,
            0.5,
            "No sampled endpoints",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )
    axis.set_title(
        f"Server endpoint {direction}: sum over {configured_planes} planes "
        f"({len(series)}/{len(groups)} active endpoints)",
        fontsize=12,
    )
    axis.grid(True, alpha=0.20)
    axis.set_ylim(0, max([observed_max, *aggregate_rates, 1.0]) * 1.08)
    axis.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1.0)
    axis.tick_params(axis="both", labelsize=10)


def plot_link_load(
    output_path: Path,
    links: dict[int, ParsedLink],
    samples: dict[int, dict[int, LinkSample]],
    bucket_times: dict[int, float],
    *,
    title: str,
    x_min_us: float,
    x_max_us: float | None,
    dpi: int,
    configured_planes: int,
) -> list[int]:
    selected_buckets = _selected_buckets(bucket_times, x_min_us, x_max_us)
    if not selected_buckets:
        raise ValueError("selected time window contains no samples")
    times = [bucket_times[bucket] for bucket in selected_buckets]

    fig = plt.figure(figsize=(14.0, 13.2))
    grid = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.22)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, :]),
    )

    physical_axes = (axes[0], axes[1], axes[2], axes[3], axes[6])
    for panel, axis in zip(DISPLAY_PANELS, physical_axes):
        panel_links = sorted(
            (item for item in links.values() if item.panel == panel.key),
            key=lambda item: item.link_id,
        )
        selected = set(selected_buckets)
        active_links = [
            item
            for item in panel_links
            if any(bucket in selected for bucket in samples.get(item.link_id, {}))
        ]
        alpha = max(0.035, min(0.28, 3.5 / math.sqrt(max(len(active_links), 1))))
        observed_max = 0.0
        for link in active_links:
            per_bucket = samples[link.link_id]
            values = [
                per_bucket[bucket].throughput_gbps if bucket in per_bucket else 0.0
                for bucket in selected_buckets
            ]
            observed_max = max(observed_max, max(values, default=0.0))
            axis.plot(times, values, color=panel.color, alpha=alpha, linewidth=0.65)

        rates = sorted({item.rate_gbps for item in panel_links})
        for index, rate in enumerate(rates):
            axis.axhline(
                rate,
                color="#D43D3D",
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
                label=(f"line rate {rate:g} Gbps" if index == 0 else f"{rate:g} Gbps"),
            )
        if rates:
            axis.legend(loc="upper right", fontsize=8, frameon=True)
        if not active_links:
            axis.text(
                0.5,
                0.5,
                "No sampled links",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
        axis.set_title(
            f"{panel.title} ({len(active_links)}/{len(panel_links)} active links)",
            fontsize=12,
        )
        axis.grid(True, alpha=0.20)
        axis.set_ylim(0, max([observed_max, *rates, 1.0]) * 1.08)
        axis.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1.0)
        axis.tick_params(axis="both", labelsize=10)

    _plot_endpoint_aggregate(
        axes[4],
        "output",
        PANEL_BY_KEY["host_l0_up"].color,
        links,
        samples,
        selected_buckets,
        times,
        configured_planes,
    )
    _plot_endpoint_aggregate(
        axes[5],
        "input",
        PANEL_BY_KEY["l0_host_down"].color,
        links,
        samples,
        selected_buckets,
        times,
        configured_planes,
    )

    axes[0].set_ylabel("Throughput (Gbps)")
    axes[2].set_ylabel("Throughput (Gbps)")
    axes[4].set_ylabel("Throughput (Gbps)")
    axes[6].set_ylabel("Throughput (Gbps)")
    axes[6].set_xlabel("Time (us)")
    fig.suptitle(title, fontsize=15, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return selected_buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing link_info.csv and link_load_1ms.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --metrics-dir.",
    )
    parser.add_argument("--title", default="MpRail Link Throughput")
    parser.add_argument("--x-min-us", type=float, default=0.0)
    parser.add_argument("--x-max-us", type=float, default=None)
    parser.add_argument(
        "--planes",
        type=int,
        default=8,
        help="Configured plane count used for endpoint aggregate line rate.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.x_min_us < 0:
        raise SystemExit("--x-min-us must be non-negative")
    if args.x_max_us is not None and args.x_max_us < args.x_min_us:
        raise SystemExit("--x-max-us must be >= --x-min-us")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.planes <= 0:
        raise SystemExit("--planes must be positive")

    metrics_dir = args.metrics_dir.resolve()
    output_dir = (args.output_dir or metrics_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    links = read_link_info(metrics_dir / "link_info.csv")
    samples, bucket_times = read_link_samples(
        metrics_dir / "link_load_1ms.csv", links
    )
    image_path = output_dir / "mprail_link_load_by_layer.png"
    selected_buckets = plot_link_load(
        image_path,
        links,
        samples,
        bucket_times,
        title=args.title,
        x_min_us=args.x_min_us,
        x_max_us=args.x_max_us,
        dpi=args.dpi,
        configured_planes=args.planes,
    )
    write_inventory(output_dir / "mprail_link_inventory.csv", links.values())
    write_summary(
        output_dir / "mprail_link_load_summary.csv",
        links,
        samples,
        selected_buckets,
    )
    write_endpoint_summary(
        output_dir / "mprail_endpoint_load_summary.csv",
        links,
        samples,
        selected_buckets,
        args.planes,
    )
    print(f"wrote {image_path}")
    print(f"parsed {len(links)} MpRail links across {len(selected_buckets)} time buckets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
