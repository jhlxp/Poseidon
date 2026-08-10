#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, HATCHES, figure, load_csv, number, save, style_axis  # noqa: E402


ORDER = ("no_remote", "fixed_8", "fixed_64", "fine_1", "full", "coarse_128", "target_05")
LABEL = {
    "no_remote": "No remote",
    "fixed_8": "Fixed 8",
    "fixed_64": "Fixed 64",
    "fine_1": "1 MiB",
    "full": "Full",
    "coarse_128": "128 MiB",
    "target_05": "Target .5",
}


def main() -> None:
    results = {row["case_id"]: row for row in load_csv(TYPE_DIR / "data/results.csv")}
    observations = load_csv(TYPE_DIR / "data/observations.csv")
    cases = [case for case in ORDER if case in results]
    reference = number(results["full"], "makespan_us")

    fig, ax = figure()
    ax.bar(
        np.arange(len(cases)),
        [number(results[case], "makespan_us") / reference for case in cases],
        color=[COLORS[index % len(COLORS)] for index in range(len(cases))],
        hatch=[HATCHES[index % len(HATCHES)] for index in range(len(cases))],
        edgecolor="black",
    )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xticks(np.arange(len(cases)), [LABEL[case] for case in cases], rotation=35, ha="right")
    ax.set_ylabel("Normalized makespan")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_07a_makespan")

    migration = defaultdict(float)
    admitted = defaultdict(float)
    deferred = defaultdict(float)
    for row in observations:
        case = row["case_id"]
        migration[case] += number(row, "migration_tx_total_bytes") / 2**30
        admitted[case] += number(row, "admitted_migration_intent_count")
        deferred[case] += number(row, "deferred_migration_intent_count")

    fig, ax = figure()
    ax.bar(np.arange(len(cases)), [migration[case] for case in cases], color=COLORS[3], edgecolor="black")
    ax.set_xticks(np.arange(len(cases)), [LABEL[case] for case in cases], rotation=35, ha="right")
    ax.set_ylabel("Migration RDMA (GiB)")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_07b_network_cost")

    fig, ax = figure()
    x = np.arange(len(cases))
    ax.bar(x, [admitted[case] for case in cases], color=COLORS[2], edgecolor="black", label="Admitted")
    ax.bar(x, [deferred[case] for case in cases], bottom=[admitted[case] for case in cases], color=COLORS[1], hatch="//", edgecolor="black", label="Deferred")
    ax.set_xticks(x, [LABEL[case] for case in cases], rotation=35, ha="right")
    ax.set_ylabel("Migration intents")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_07c_admission")

    chunk_cases = [case for case in ("fine_1", "full", "coarse_128") if case in results]
    fig, ax = figure()
    ax.plot(
        [number(results[case], "weight_chunk_mib") for case in chunk_cases],
        [number(results[case], "makespan_us") / 1000.0 for case in chunk_cases],
        marker="o", color=COLORS[0],
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Weight chunk (MiB)")
    ax.set_ylabel("Makespan (ms)")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_07d_chunk_granularity")


if __name__ == "__main__":
    main()
