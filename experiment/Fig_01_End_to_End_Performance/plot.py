#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, HATCHES, figure, load_csv, number, ordered, save, save_legend, style_axis  # noqa: E402


ALGORITHM_ORDER = ("nccl", "deepep", "eplb", "moonep", "probeep")
LABELS = {name: name.upper() if name != "probeep" else "ProbeEP" for name in ALGORITHM_ORDER}


def rows_by_hardware(rows):
    output = defaultdict(dict)
    for row in rows:
        if row.get("status", "passed") == "passed":
            output[row["hardware"]][row["algorithm"]] = row
    return output


def grouped_bars(grouped, value_fn, ylabel, stem):
    hardwares = ordered(grouped)
    algorithms = [name for name in ALGORITHM_ORDER if all(name in grouped[hw] for hw in hardwares)]
    fig, ax = figure()
    x = np.arange(len(hardwares), dtype=float)
    width = 0.82 / len(algorithms)
    handles = []
    for index, algorithm in enumerate(algorithms):
        values = [value_fn(grouped[hw][algorithm], hw, grouped) for hw in hardwares]
        bars = ax.bar(
            x + (index - (len(algorithms) - 1) / 2) * width,
            values,
            width,
            color=COLORS[index],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[index],
            label=LABELS[algorithm],
        )
        handles.append(bars[0])
    ax.set_xticks(x, hardwares)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    save(fig, TYPE_DIR, stem)
    return handles, [LABELS[name] for name in algorithms]


def main() -> None:
    rows = load_csv(TYPE_DIR / "data/results.csv")
    grouped = rows_by_hardware(rows)
    handles, labels = grouped_bars(
        grouped,
        lambda row, _hw, _all: number(row, "makespan_us") / 1000.0,
        "Makespan (ms)",
        "fig_01a_makespan",
    )
    grouped_bars(
        grouped,
        lambda row, hw, all_rows: number(row, "makespan_us")
        / min(
            number(candidate, "makespan_us")
            for name, candidate in all_rows[hw].items()
            if name != "probeep"
        ),
        "Normalized makespan",
        "fig_01b_normalized",
    )

    fig, ax = figure()
    hardwares = ordered(grouped)
    speedups = []
    for hardware in hardwares:
        baseline = min(
            number(row, "makespan_us")
            for name, row in grouped[hardware].items()
            if name != "probeep"
        )
        speedups.append(baseline / number(grouped[hardware]["probeep"], "makespan_us"))
    ax.bar(
        np.arange(len(hardwares)), speedups, width=0.62,
        color=COLORS[4], edgecolor="black", linewidth=0.8, hatch=HATCHES[4],
    )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.3)
    ax.set_xticks(np.arange(len(hardwares)), hardwares)
    ax.set_ylabel("Speedup over best baseline")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_01c_speedup")
    grouped_bars(
        grouped,
        lambda row, _hw, _all: (
            number(row, "tokens_per_rank_per_microbatch")
            * 32
            * 2
            / number(row, "makespan_us")
        ),
        "Throughput (Mtoken/s)",
        "fig_01d_throughput",
    )
    save_legend(handles, labels, TYPE_DIR, "fig_01_legend", ncol=3)


if __name__ == "__main__":
    main()
