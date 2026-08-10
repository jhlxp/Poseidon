#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, HATCHES, figure, load_csv, number, save, style_axis  # noqa: E402


def mean_by(rows, keys, value):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(number(row, value))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def main() -> None:
    loads = load_csv(TYPE_DIR / "data/load_balance.csv")
    planning = load_csv(TYPE_DIR / "data/planning.csv")
    network = load_csv(TYPE_DIR / "data/network_cost.csv")

    base = [row for row in loads if row["hardware"] == "H20" and row["algorithm"] == "nccl" and row["phase"] == "before"]
    fig, ax = figure()
    x = np.arange(len(base))
    ax.plot(x, [number(row, "rank_max_mean") for row in base], marker="o", label="Rank")
    ax.plot(x, [number(row, "server_max_mean") for row in base], marker="s", label="Server")
    ax.set_xlabel("Layer / microbatch")
    ax.set_ylabel("Load imbalance")
    ax.set_xticks(x, [f"L{row['layer']}M{row['micro_batch']}" for row in base], rotation=25)
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_02a_gate_skew")

    after = mean_by(
        [row for row in loads if row["hardware"] == "H20" and row["phase"] == "after"],
        ("algorithm",), "server_max_mean",
    )
    rank_after = mean_by(
        [row for row in loads if row["hardware"] == "H20" and row["phase"] == "after"],
        ("algorithm",), "rank_max_mean",
    )
    algorithms = [name for name in ("nccl", "deepep", "eplb", "moonep", "probeep") if (name,) in after]
    fig, ax = figure()
    x = np.arange(len(algorithms))
    width = 0.36
    ax.bar(x - width / 2, [rank_after[(name,)] for name in algorithms], width, label="Rank", color=COLORS[0], edgecolor="black")
    ax.bar(x + width / 2, [after[(name,)] for name in algorithms], width, label="Server", color=COLORS[1], hatch=HATCHES[1], edgecolor="black")
    ax.set_xticks(x, [name.upper() if name != "probeep" else "ProbeEP" for name in algorithms], rotation=25)
    ax.set_ylabel("Post-plan imbalance")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_02b_local_ceiling")

    probe = [row for row in planning if row["hardware"] == "H20" and row["algorithm"] == "probeep"]
    stages = ("baseline_padded_routes", "planned_padded_routes", "admitted_padded_routes")
    labels = ("Baseline", "Planned", "Admitted")
    values = [statistics.fmean(number(row, field) for row in probe) / 1e6 for field in stages]
    fig, ax = figure()
    ax.bar(np.arange(3), values, color=COLORS[:3], edgecolor="black", hatch=HATCHES[:3])
    ax.set_xticks(np.arange(3), labels, rotation=20)
    ax.set_ylabel("Padded routes (million)")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_02c_padding")

    probe_network = next(row for row in network if row["hardware"] == "H20" and row["algorithm"] == "probeep")
    values = [number(probe_network, "weight_rdma_bytes") / 2**30, number(probe_network, "weight_local_bytes") / 2**30]
    fig, ax = figure()
    ax.bar(np.arange(2), values, color=(COLORS[3], COLORS[2]), edgecolor="black", hatch=(HATCHES[3], HATCHES[2]))
    ax.set_xticks(np.arange(2), ("Scale-out", "Server-local"))
    ax.set_ylabel("Weight movement (GiB)")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_02d_network_cost")

    moved = [number(row, "moved_routes") for row in probe]
    replicas = [number(row, "remote_replicas") for row in probe]
    efficiency = [
        routes / replica if replica else 0.0
        for routes, replica in zip(moved, replicas)
    ]
    fig, ax = figure()
    x = np.arange(len(probe))
    ax.bar(
        x, efficiency, color=COLORS[4], edgecolor="black", hatch=HATCHES[4]
    )
    ax.set_xticks(
        x,
        [f"L{row['layer']}M{row['micro_batch']}" for row in probe],
        rotation=25,
    )
    ax.set_xlabel("Invocation")
    ax.set_ylabel("Moved routes / replica")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_02e_migration_efficiency")


if __name__ == "__main__":
    main()
