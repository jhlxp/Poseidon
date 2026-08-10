#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, MARKERS, figure, load_csv, number, save, style_axis  # noqa: E402


def feedback_rows(rows):
    return sorted(
        (row for row in rows if row["case_id"] == "feedback"),
        key=lambda row: int(row["observation_id"]),
    )


def main() -> None:
    results = load_csv(TYPE_DIR / "data/results.csv")
    observations = load_csv(TYPE_DIR / "data/observations.csv")
    feedback = feedback_rows(observations)

    fig, ax = figure()
    for index, kind in enumerate(("attention", "moe")):
        rows = [row for row in feedback if row["compute_kind"] == kind]
        ax.plot(
            [int(row["observation_id"]) for row in rows],
            [number(row, "global_network_to_compute_ratio") for row in rows],
            marker=MARKERS[index], color=COLORS[index], label=kind.capitalize(),
        )
    ax.axhline(0.9, color="#444444", linestyle="--", linewidth=1.3, label="Target")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Network / compute")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_03a_ratio")

    fig, ax = figure()
    x = [int(row["observation_id"]) for row in feedback]
    ax.step(x, [number(row, "budget_before_mean_mib") for row in feedback], where="mid", label="Before", color=COLORS[0])
    ax.step(x, [number(row, "budget_after_mean_mib") for row in feedback], where="mid", label="After", color=COLORS[1])
    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean budget (MiB)")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_03b_budget")

    fig, ax = figure()
    planned = [number(row, "planned_migration_intent_count") for row in feedback]
    admitted = [number(row, "admitted_migration_intent_count") for row in feedback]
    deferred = [number(row, "deferred_migration_intent_count") for row in feedback]
    xarr = np.arange(len(feedback))
    ax.bar(xarr, admitted, color=COLORS[2], edgecolor="black", label="Admitted")
    ax.bar(xarr, deferred, bottom=admitted, color=COLORS[3], hatch="//", edgecolor="black", label="Deferred")
    ax.plot(xarr, planned, color="#222222", marker="o", label="Planned")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Migration intents")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_03c_admission")

    migration = defaultdict(float)
    for row in observations:
        migration[row["case_id"]] += number(row, "migration_tx_total_bytes") / 2**30
    ordered_cases = ("fixed_0", "fixed_16", "fixed_64", "feedback")
    by_case = {row["case_id"]: row for row in results}
    fig, ax = figure()
    for index, case_id in enumerate(ordered_cases):
        ax.scatter(
            migration[case_id], number(by_case[case_id], "makespan_us") / 1000.0,
            s=95, marker=MARKERS[index], color=COLORS[index], edgecolor="black", label=case_id,
        )
    ax.set_xlabel("Migration RDMA (GiB)")
    ax.set_ylabel("Makespan (ms)")
    ax.legend(frameon=False)
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_03d_fixed_vs_feedback")


if __name__ == "__main__":
    main()
