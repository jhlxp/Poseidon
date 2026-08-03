import os
import argparse
import csv
import re
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

"""
Single bar plot of collection completion time (CCT) per routing.

The preferred input is the metrics CSV emitted by the simulator. Older Spritz
outputs only carried completion times in run logs, so log parsing is retained as
a fallback.
"""

routings = [
    "MINIMAL",
    "VALIANT",
    "UGAL_L",
    "SOURCE_ECMP",
    "SOURCE_FLICR",
    "SOURCE_OPS_U",
    "SOURCE_OPS_W",
    "SOURCE_FLOW_V1",
    "SOURCE_FLOW_V2_U",
    "SOURCE_FLOW_V2_W",
]

tick_labels = [
    "Minimal",
    "Valiant",
    "UGAL-L",
    "ECMP",
    "Flicr (w)",
    "OPS (u)",
    "OPS (w)",
    "Spritz-Scout (w)",
    "Spritz-Spray (u)",
    "Spritz-Spray (w)",
]

colors = [
    "#262636",
    "#7e2c40",
    "#a42f44",
    "#d85034",
    "#ee8143",
    "#228a8d",
    "#145a5c",
    "#45448b",
    "#7c7ae6",
    "#3885cf",
]


def _find_routing_log_files(base_path: str, routing: str) -> List[str]:
    """Find log files for a given routing under base_path/run_logs (or base_path fallback)."""
    search_roots: List[str] = []
    run_logs = os.path.join(base_path, "run_logs")
    if os.path.isdir(run_logs):
        search_roots.append(run_logs)
    search_roots.append(base_path)

    matches: List[str] = []
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                lower = fname.lower()
                if not (lower.endswith(".tmp") or lower.endswith(".log")):
                    continue
                if (
                    fname.endswith(f"_{routing}.tmp")
                    or fname.endswith(f"_{routing}.log")
                    or f"_{routing}_" in fname
                    or f"_{routing}." in fname
                ):
                    matches.append(os.path.join(dirpath, fname))
    # Deduplicate, preserve order
    seen, result = set(), []
    for p in matches:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _max_finished_at_in_file(path: str) -> float:
    """Return the maximum 'finished at <val>' seen in a single log file (us)."""
    max_us = 0.0
    pattern = re.compile(r"\bfinished at\s+([0-9]*\.?[0-9]+)\b")
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                        if val > max_us:
                            max_us = val
                    except ValueError:
                        continue
    except OSError:
        return 0.0
    return max_us


def _read_ccts_from_summary(base_path: str) -> Dict[str, float]:
    summaries = [
        os.path.join(base_path, fname)
        for fname in os.listdir(base_path)
        if fname.endswith("_summary.csv")
    ]
    ccts: dict[str, float] = {}
    for path in summaries:
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    routing = row.get("routing")
                    traffic_end = row.get("traffic_end")
                    if not routing or not traffic_end:
                        continue
                    ccts[routing] = max(ccts.get(routing, 0.0), float(traffic_end) / 1000.0)
        except (OSError, ValueError):
            continue
    return ccts


def _max_end_time_in_flows_file(path: str) -> float:
    max_us = 0.0
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                end_time = row.get("endTimeNs")
                if not end_time:
                    continue
                max_us = max(max_us, float(end_time) / 1000.0)
    except (OSError, ValueError):
        return 0.0
    return max_us


def _find_routing_flow_files(base_path: str, routing: str) -> List[str]:
    matches: List[str] = []
    suffix = f"_{routing}_flows.csv"
    for fname in os.listdir(base_path):
        if fname.endswith(suffix):
            matches.append(os.path.join(base_path, fname))
    return matches


def _compute_cct_by_routing(base_path: str) -> List[float]:
    """Compute Collection Completion Time (CCT, us) per routing."""
    summary_ccts = _read_ccts_from_summary(base_path)
    ccts_us: List[float] = []
    for routing in routings:
        routing_max = summary_ccts.get(routing, 0.0)
        if routing_max == 0.0:
            flow_paths = _find_routing_flow_files(base_path, routing)
            per_file_max = [_max_end_time_in_flows_file(p) for p in flow_paths]
            routing_max = max(per_file_max) if per_file_max else 0.0
        if routing_max == 0.0:
            log_paths = _find_routing_log_files(base_path, routing)
            per_file_max = [_max_finished_at_in_file(p) for p in log_paths]
            routing_max = max(per_file_max) if per_file_max else 0.0
        ccts_us.append(routing_max)
    return ccts_us


def create_collection_barplot(base_path: str, unit: str = "us", outfile: Optional[str] = None, show: bool = True, fig_width: float = 3.6):
    """Create a single bar plot of Collection Completion Time per routing.

    unit: 'us' (microseconds) or 'ms' (milliseconds) for y-axis and labels.
    """
    ccts_us = _compute_cct_by_routing(base_path)

    # Convert if needed
    if unit == "ms":
        values = [v / 1000.0 for v in ccts_us]
        ylabel = "Collection completion time (ms)"
    else:
        values = ccts_us
        ylabel = "Collection completion time (us)"

    positions = np.arange(len(routings))
    fig_height = 2.2  # compact
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    bars = ax.bar(positions, values, width=0.6, color=colors, zorder=3)

    # Pretty x-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    # Y-axis
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#bbbbbb", zorder=0)

    # Bar labels
    def fmt(v: float) -> str:
        # keep one decimal for ms/us when large
        if unit == "us":
            return f"{v/1000:.1f}k" if v >= 10000 else f"{v:.1f}"
        return f"{v/1000:.1f}k" if v >= 10000 else f"{v:.2f}"

    labels = [fmt(h) for h in values]
    ax.bar_label(bars, labels=labels, fontsize=7, padding=3)

    ax.set_xlim(-0.5, len(routings) - 0.5)
    ymax = max(values) if values else 0
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.15)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig("collection_times.pdf", bbox_inches="tight")
    if show:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Bar plot of collection completion time per routing.")
    ap.add_argument("--experiment", required=True, help="Experiment folder name.")
    ap.add_argument("--df-topo", default="p4a8h4", help="Dragonfly topology folder (default p4a8h4).")
    ap.add_argument("--sf-topo", help="Slim Fly topology folder (e.g., p7q9). If omitted, Dragonfly is used.")
    ap.add_argument("--output-root", default="experiments_output", help="Root output directory (default experiments_output).")
    ap.add_argument("--scenario", choices=["no_fail", "fail_2p"], default="no_fail", help="Which scenario subfolder to use (default no_fail).")
    ap.add_argument("--unit", choices=["us", "ms"], default="us", help="Time unit for the y-axis (default us).")
    ap.add_argument("--outfile", default=None, help="Output PDF filename.")
    ap.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    ap.add_argument("--fig-width", type=float, default=3.6, help="Figure width in inches (default 3.6).")
    args = ap.parse_args()

    # Determine base path
    if args.sf_topo:
        base_path = os.path.join(args.output_root, "sf", args.sf_topo, args.scenario, args.experiment)
    else:
        base_path = os.path.join(args.output_root, "df", args.df_topo, args.scenario, args.experiment)

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    create_collection_barplot(
        base_path=base_path,
        unit=args.unit,
        outfile=args.outfile,
        show=not args.no_show,
        fig_width=args.fig_width,
    )

if __name__ == "__main__":
    main()
