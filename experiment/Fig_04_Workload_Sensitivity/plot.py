#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


TYPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TYPE_DIR.parent / "common"))
from plotting import COLORS, MARKERS, figure, load_csv, number, save, save_legend, style_axis  # noqa: E402


def paired_results(rows):
    output = defaultdict(dict)
    for row in rows:
        output[row["case_id"]][row["algorithm"]] = row
    return output


def speedup(pair):
    return number(pair["moonep"], "makespan_us") / number(pair["probeep"], "makespan_us")


def main() -> None:
    results = paired_results(load_csv(TYPE_DIR / "data/results.csv"))
    gates = load_csv(TYPE_DIR / "data/gate_metrics.csv")
    observations = load_csv(TYPE_DIR / "data/observations.csv")
    gate_by_case = {
        row["case_id"]: row for row in gates if row["algorithm"] == "moonep"
    }
    providers = list(dict.fromkeys(row["gate_provider"] for row in gates))
    provider_style = {name: index for index, name in enumerate(providers)}

    fig, ax = figure()
    for case_id, pair in results.items():
        if set(pair) != {"moonep", "probeep"} or case_id.startswith("raw_t"):
            continue
        provider = pair["probeep"]["gate_provider"]
        index = provider_style[provider] % len(COLORS)
        ax.scatter(
            number(gate_by_case[case_id], "server_max_mean"), speedup(pair),
            s=90, color=COLORS[index], marker=MARKERS[index], edgecolor="black", label=provider,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Server imbalance")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_04a_skew_speedup")
    save_legend(
        list(unique.values()),
        list(unique.keys()),
        TYPE_DIR,
        "fig_04_legend",
        ncol=2,
    )

    migration = defaultdict(float)
    for row in observations:
        migration[row["case_id"]] += number(row, "migration_tx_total_bytes") / 2**30
    cases = [case for case in results if case in gate_by_case and not case.startswith("raw_t")]
    fig, ax = figure()
    ax.scatter(
        [number(gate_by_case[case], "server_max_mean") for case in cases],
        [migration[case] for case in cases],
        s=82, color=COLORS[3], edgecolor="black",
    )
    ax.set_xlabel("Server imbalance")
    ax.set_ylabel("Migration RDMA (GiB)")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_04b_migration")

    token_cases = [case for case in ("raw_t1024", "raw", "raw_t8192") if case in results]
    fig, ax = figure()
    ax.plot(
        [int(results[case]["probeep"]["tokens"]) for case in token_cases],
        [speedup(results[case]) for case in token_cases],
        marker="o", color=COLORS[0],
    )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Tokens per rank")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax, grid="both")
    save(fig, TYPE_DIR, "fig_04c_token_volume")

    seed_cases = [case for case in ("ultra_2", "ultra_seed23", "ultra_seed41") if case in results]
    fig, ax = figure()
    values = [speedup(results[case]) for case in seed_cases]
    ax.bar(np.arange(len(values)), values, color=COLORS[2], edgecolor="black")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xticks(np.arange(len(values)), [results[case]["probeep"]["seed"] for case in seed_cases])
    ax.set_xlabel("Gate seed")
    ax.set_ylabel("Speedup over MoonEP")
    style_axis(ax)
    save(fig, TYPE_DIR, "fig_04d_seed_variation")


if __name__ == "__main__":
    main()
