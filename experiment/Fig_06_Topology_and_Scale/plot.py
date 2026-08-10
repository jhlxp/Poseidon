#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, MARKERS, figure, load_csv, number, save, style_axis  # noqa: E402


def grouped(rows):
    output = defaultdict(dict)
    for row in rows:
        output[row["case_id"]][row["algorithm"]] = row
    return output


def speedup(pair):
    return number(pair["moonep"], "makespan_us") / number(pair["probeep"], "makespan_us")


def main() -> None:
    results = grouped(load_csv(TYPE_DIR / "data/results.csv"))
    network = grouped(load_csv(TYPE_DIR / "data/network_metrics.csv"))

    boundary = sorted(
        (case for case, pair in results.items() if pair["probeep"]["sweep"] == "boundary"),
        key=lambda case: number(results[case]["probeep"], "servers"),
    )
    fig, ax = figure()
    ax.plot(
        [number(results[case]["probeep"], "servers") for case in boundary],
        [speedup(results[case]) for case in boundary], marker="o", color=COLORS[0],
    )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Servers in EP32")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_06a_server_boundary")

    fig, ax = figure()
    for index, algorithm in enumerate(("moonep", "probeep")):
        ax.plot(
            [number(results[case][algorithm], "servers") for case in boundary],
            [number(network[case][algorithm], "endpoint_peak_utilization") for case in boundary],
            marker=MARKERS[index], color=COLORS[index], label=algorithm,
        )
    ax.set_xlabel("Servers in EP32")
    ax.set_ylabel("Peak endpoint utilization")
    ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_06b_endpoint_load")

    path_cases = [case for case, pair in results.items() if pair["probeep"]["sweep"] == "paths"]
    path_cases.sort(key=lambda case: number(results[case]["probeep"], "spines") * number(results[case]["probeep"], "links"))
    capacity = [number(results[case]["probeep"], "spines") * number(results[case]["probeep"], "links") for case in path_cases]
    fig, ax = figure()
    ax.plot(capacity, [speedup(results[case]) for case in path_cases], marker="s", color=COLORS[2])
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Spine-path multiplicity")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_06c_path_diversity")

    fig, ax = figure()
    for index, algorithm in enumerate(("moonep", "probeep")):
        ax.plot(
            capacity,
            [number(network[case][algorithm], "max_queue_bytes") / 1024 for case in path_cases],
            marker=MARKERS[index], color=COLORS[index], label=algorithm,
        )
    ax.set_xlabel("Spine-path multiplicity")
    ax.set_ylabel("Max queue (KiB)")
    ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_06d_queue_pressure")

    plane_cases = sorted(
        (
            case
            for case, pair in results.items()
            if pair["probeep"]["sweep"] == "planes"
        ),
        key=lambda case: number(results[case]["probeep"], "planes"),
    )
    fig, ax = figure()
    ax.plot(
        [number(results[case]["probeep"], "planes") for case in plane_cases],
        [speedup(results[case]) for case in plane_cases],
        marker="D",
        color=COLORS[4],
    )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Configured planes")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_06e_plane_capacity")


if __name__ == "__main__":
    main()
