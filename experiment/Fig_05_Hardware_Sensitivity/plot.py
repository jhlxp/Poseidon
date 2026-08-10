#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, MARKERS, figure, load_csv, number, save, style_axis  # noqa: E402


def pairs(rows):
    output = defaultdict(dict)
    for row in rows:
        output[row["case_id"]][row["algorithm"]] = row
    return output


def speedup(pair):
    return number(pair["moonep"], "makespan_us") / number(pair["probeep"], "makespan_us")


def line_panel(grouped, cases, xfield, xlabel, stem):
    fig, ax = figure()
    for index, (label, selected) in enumerate(cases.items()):
        selected = sorted(selected, key=lambda case: number(grouped[case]["probeep"], xfield))
        ax.plot(
            [number(grouped[case]["probeep"], xfield) for case in selected],
            [speedup(grouped[case]) for case in selected],
            marker=MARKERS[index], color=COLORS[index], label=label,
        )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup over MoonEP")
    if len(cases) > 1:
        ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, stem)


def main() -> None:
    rows = load_csv(TYPE_DIR / "data/results.csv")
    grouped = pairs(rows)
    nic_cases = {
        hardware: [
            case for case, pair in grouped.items()
            if pair["probeep"]["sweep"] == "nic" and pair["probeep"]["hardware"] == hardware
        ]
        for hardware in ("H20", "H100")
    }
    line_panel(grouped, nic_cases, "nic_gbps", "NIC rate (Gbps)", "fig_05a_compute_nic")
    line_panel(
        grouped,
        {"Expert state": [case for case, pair in grouped.items() if pair["probeep"]["sweep"] == "weight"]},
        "weight_scale", "Expert state scale", "fig_05b_expert_state",
    )
    line_panel(
        grouped,
        {"Local fabric": [case for case, pair in grouped.items() if pair["probeep"]["sweep"] == "local"]},
        "local_gbps", "Local fabric (Gbps)", "fig_05c_local_fabric",
    )

    observations = load_csv(TYPE_DIR / "data/observations.csv")
    migration = defaultdict(float)
    for row in observations:
        if row["sweep"] == "nic":
            migration[row["case_id"]] += number(row, "migration_tx_total_bytes") / 2**30
    fig, ax = figure()
    for index, hardware in enumerate(("H20", "H100")):
        selected = sorted(nic_cases[hardware], key=lambda case: number(grouped[case]["probeep"], "nic_gbps"))
        ax.plot(
            [number(grouped[case]["probeep"], "nic_gbps") for case in selected],
            [migration[case] for case in selected],
            marker=MARKERS[index], color=COLORS[index], label=hardware,
        )
    ax.set_xlabel("NIC rate (Gbps)")
    ax.set_ylabel("Migration RDMA (GiB)")
    ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_05d_admission")


if __name__ == "__main__":
    main()
