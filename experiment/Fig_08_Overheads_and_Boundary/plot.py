#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, figure, load_csv, number, save, style_axis  # noqa: E402


def scaling_panel(rows, sweep, xfield, xlabel, stem):
    selected = sorted(
        (row for row in rows if row["sweep"] == sweep),
        key=lambda row: number(row, xfield),
    )
    fig, ax = figure()
    ax.plot(
        [number(row, xfield) for row in selected],
        [number(row, "runtime_median_ms") for row in selected],
        marker="o", color=COLORS[0], label="Median",
    )
    ax.plot(
        [number(row, xfield) for row in selected],
        [number(row, "runtime_p95_ms") for row in selected],
        marker="s", color=COLORS[1], label="P95",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reference plan time (ms)")
    ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, stem)


def main() -> None:
    planner = load_csv(TYPE_DIR / "data/planner_scaling.csv")
    boundary = load_csv(TYPE_DIR / "data/break_even.csv")
    scaling_panel(planner, "ep", "ep", "EP ranks", "fig_08a_ep_scaling")
    scaling_panel(planner, "experts", "experts", "Logical experts", "fig_08b_expert_scaling")
    scaling_panel(planner, "routes", "logical_routes", "Logical routes", "fig_08c_route_scaling")

    selected = [row for row in boundary if number(row, "effective_rate_gbps") == 400]
    weights = sorted({number(row, "weight_scale") for row in selected})
    routes = sorted({number(row, "moved_routes") for row in selected})
    matrix = np.array(
        [
            [
                next(
                    number(row, "net_saving_us")
                    for row in selected
                    if number(row, "weight_scale") == weight
                    and number(row, "moved_routes") == route
                )
                for route in routes
            ]
            for weight in weights
        ]
    )
    fig, ax = figure()
    scale = max(abs(float(matrix.min())), abs(float(matrix.max())))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale, origin="lower")
    ax.contour(matrix, levels=[0], colors="black", linewidths=1.5)
    ax.set_xticks(np.arange(len(routes)), [f"{int(value / 1024)}K" if value >= 1024 else str(int(value)) for value in routes], rotation=30)
    ax.set_yticks(np.arange(len(weights)), [f"{value:g}x" for value in weights])
    ax.set_xlabel("Moved routes / replica")
    ax.set_ylabel("Expert state scale")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Net saving (us)", fontsize=22)
    colorbar.ax.tick_params(labelsize=19)
    save(fig, TYPE_DIR, "fig_08d_break_even")


if __name__ == "__main__":
    main()
