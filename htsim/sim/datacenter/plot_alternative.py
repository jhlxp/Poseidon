import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import re

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
    "SOURCE_FLOW_V2_W"
]
tick_labels = ["Minimal", "Valiant", "UGAL-L", "ECMP", "Flicr (w)", "OPS (u)", "OPS (w)", "Spritz-Scout (w)", "Spritz-Spray (u)", "Spritz-Spray (w)"]
colors = ["#262636", "#7e2c40", "#a42f44", "#d85034", "#ee8143", "#228a8d", "#145a5c", "#45448b", "#7c7ae6", "#3885cf"]

# Scale factors
BAR_ANNOTATION_SCALE = 1.08
AXIS_TITLE_SCALE = 1.2

def create_thesis_plot(base_path, experiment, unit, srcs_filter, detail_type, log_scale, show_x_labels, outfile, layout="stacked", show=True, hide_y_titles=False, fig_width=3.6, flow_kind=None, incast_dst=None):
    data = []
    data_avg = []
    data_traffic_end = []
    percentiles_99 = []
    ooo_from_csv = []

    srcs = [298, 941, 931, 357, 804, 716, 887, 1038, 225, 248, 164, 933, 539, 98, 418, 686, 470, 633, 417, 365, 288, 386, 710, 757, 839, 431, 825, 944, 564, 768, 328, 254, 369, 11, 810, 302, 332, 394, 342, 53, 485, 917, 793, 259, 107, 506, 593, 693, 703, 743, 990, 822, 126, 207, 614, 889, 517, 491, 895, 1051, 1010, 520, 301, 119, 669, 337, 111, 563, 247, 103, 448, 634, 443, 731, 683, 25, 233, 973, 243, 428, 89, 798, 213, 820, 310, 976, 79, 911, 329, 101, 970, 96, 405, 509, 322, 193, 286, 433, 184, 251, 515, 136, 823, 75, 873, 971, 922, 930, 397, 514, 855, 297, 427, 845, 158, 550, 533, 194, 390, 466, 799, 555, 5, 623, 77, 603, 475, 824]

    # Load summary once (used later, but do not gate FCT plotting with it)
    summary_csv = f"{base_path}/{experiment}_summary.csv"
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")
    df_summary_all = pd.read_csv(summary_csv)

    for routing in routings:
        flows_csv = f"{base_path}/{experiment}_{routing}_flows.csv"
        # summary_csv removed from here (loaded once above)
        if not os.path.isfile(flows_csv):
            raise FileNotFoundError(f"Missing flows file: {flows_csv}")
        df = pd.read_csv(flows_csv)
        # df_summary = pd.read_csv(summary_csv)  # removed
        if srcs_filter in (1, 2) or flow_kind:
            if "srcNode_dstNode_flowId" in df.columns:
                df[["srcNode", "dstNode", "flowId"]] = df["srcNode_dstNode_flowId"].str.split("_", expand=True)
                df["srcNode"] = df["srcNode"].astype(int)
                df["dstNode"] = df["dstNode"].astype(int)
                if srcs_filter == 1:
                    df = df[df["srcNode"].isin(srcs) & df["dstNode"].isin(srcs)]
                elif srcs_filter == 2:
                    df = df[~(df["srcNode"].isin(srcs) & df["dstNode"].isin(srcs))]
                if flow_kind and incast_dst is not None:
                    if flow_kind == "incast":
                        df = df[df["dstNode"] == incast_dst]
                    elif flow_kind == "bystanders":
                        df = df[df["dstNode"] != incast_dst]
        if unit == "us":
            fct_vals = (df["fctNs"] / 1000.0).dropna()
            end_vals = (df["endTimeNs"] / 1000.0).dropna()
        else:
            fct_vals = (df["fctNs"] / 1_000_000.0).dropna()
            end_vals = (df["endTimeNs"] / 1_000_000.0).dropna()

        # Always plot the data; do not gate on ack/new
        data_avg.append(fct_vals.mean() if len(fct_vals) else 0)
        data_traffic_end.append(end_vals.max() if len(end_vals) else 0)
        data.append(fct_vals.values if len(fct_vals) else [0])
        percentiles_99.append(np.percentile(fct_vals.values, 99) if len(fct_vals) else 0)

        if {"oooCount", "oooAvgDistance", "oooMaxDistance"}.issubset(df.columns):
            if "totalPackets" in df.columns:
                total_packets = pd.to_numeric(df["totalPackets"], errors="coerce")
            else:
                total_packets = pd.to_numeric(df["flowSizeBytes"], errors="coerce") / 4096.0
            ooo_counts = pd.to_numeric(df["oooCount"], errors="coerce").fillna(0)
            valid_totals = total_packets.fillna(0) > 0
            pct_values = 100.0 * ooo_counts[valid_totals] / total_packets[valid_totals]
            ooo_from_csv.append((
                float(pct_values.mean()) if len(pct_values) else 0.0,
                float(pd.to_numeric(df["oooAvgDistance"], errors="coerce").fillna(0).mean()) if len(df) else 0.0,
                float(pd.to_numeric(df["oooMaxDistance"], errors="coerce").fillna(0).max()) if len(df) else 0.0,
            ))
        else:
            ooo_from_csv.append(None)

    positions = np.arange(0, len(data))

    if layout == "stacked":
        row_heights = [1.95, 1.25, 1.25]
        total_height = sum(row_heights)
        fig = plt.figure(figsize=(fig_width, total_height))
        gs = gridspec.GridSpec(3, 1, height_ratios=row_heights, hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
    else:
        row_heights = [1.8, 1.3]
        total_height = sum(row_heights)
        ratios = [h / total_height for h in row_heights]
        fig = plt.figure(figsize=(fig_width, total_height))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.08)
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=ratios, hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs_right[0], sharex=ax1)
        ax3 = fig.add_subplot(gs_right[1], sharex=ax1)

    # Apply light background to middle and bottom plots
    ax2.set_facecolor("#f1f1f1")
    ax3.set_facecolor("#f1f1f1")

    # ================================================

    # Violin plot (FCT)
    ax = ax1
    parts = ax.violinplot(
        data, 
        showmeans=True, 
        showmedians=False, 
        showextrema=True,
        positions=positions,
    )

    for i, color in enumerate(colors):
        parts["bodies"][i].set_facecolor(color)
        parts["bodies"][i].set_alpha(0.5)
    parts["cbars"].set_color(colors)
    parts["cmeans"].set_color(colors)
    parts["cmaxes"].set_color(colors)
    parts["cmins"].set_color(colors)

    ax.scatter(
        positions, 
        percentiles_99, 
        color="black", 
        marker="D",
        s=22,
        zorder=3,
        label="99th Percentile",
    )

    ax.grid(color="gray", linestyle=":", linewidth=0.5, axis="y")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    ax.tick_params(axis="y", labelsize=8)
    if not hide_y_titles:
        ax.set_ylabel(f"FCT ({unit})", fontsize=8 * AXIS_TITLE_SCALE)
    
    if log_scale:
        ax.set_yscale("log")

    # X labels: stacked -> only bottom; classic -> show on left panel
    if layout == "stacked":
        ax.tick_params(bottom=False, labelbottom=False)
    else:
        ax.tick_params(bottom=True, labelbottom=True)

    # ================================================

    # Helpers for bar label formatting
    def format_label_time(value):
        if unit == "us":
            return f"{value/1000:.1f}k" if value >= 10000 else f"{round(value, 1)}"
        elif unit == "ms":
            return f"{value/1000:.1f}k" if value >= 10000 else f"{round(value, 2)}"

    def format_label_int(value):
        return f"{value/1000:.1f}k" if value >= 10000 else f"{int(round(value))}"

    def format_label_pct(value):
        return f"{value:.1f}%"

    # Helpers to find and parse OOO counts per routing
    def find_ooo_log_files(base_path, routing):
        # Search run_logs recursively, fallback to base_path
        search_roots = []
        run_logs = os.path.join(base_path, "run_logs")
        if os.path.isdir(run_logs):
            search_roots.append(run_logs)
        search_roots.append(base_path)

        matches = []
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
        # Deduplicate while preserving order
        seen, result = set(), []
        for p in matches:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result

    def parse_ooo_stats(paths):
        # Extract per-flow: count, avgDist, maxDist, totalPackets
        pct_values, avg_dists, max_dists = [], [], []
        for path in paths:
            try:
                with open(path, "r", errors="ignore") as f:
                    for line in f:
                        m_count = re.search(r"\bOOO count=(\d+)\b", line)
                        m_avg   = re.search(r"\bavgDist=([0-9]*\.?[0-9]+)\b", line)
                        m_max   = re.search(r"\bmaxDist=(\d+)\b", line)
                        m_total = re.search(r"\btotalPackets=(\d+)\b", line)
                        if m_count and m_total:
                            total = int(m_total.group(1))
                            if total > 0:
                                pct_values.append(100.0 * int(m_count.group(1)) / total)
                        if m_avg:
                            avg_dists.append(float(m_avg.group(1)))
                        if m_max:
                            max_dists.append(int(m_max.group(1)))
            except OSError:
                continue
        pct_avg = float(np.mean(pct_values)) if pct_values else 0.0
        avgdist_mean = float(np.mean(avg_dists)) if avg_dists else 0.0
        maxdist_max = float(np.max(max_dists)) if max_dists else 0.0
        return pct_avg, avgdist_mean, maxdist_max

    # Compute OOO percentage and distances for each routing (used when detail_type == "ooo_avg")
    ooo_pct_avgs, ooo_avgdists, ooo_maxdists = [], [], []
    for routing, csv_stats in zip(routings, ooo_from_csv):
        if csv_stats is not None:
            pct_avg, avgdist_mean, maxdist_max = csv_stats
        else:
            paths = find_ooo_log_files(base_path, routing)
            pct_avg, avgdist_mean, maxdist_max = parse_ooo_stats(paths)
        ooo_pct_avgs.append(pct_avg)
        ooo_avgdists.append(avgdist_mean)
        ooo_maxdists.append(maxdist_max)

    # Pre-compute dropped packets once
    data_rtx = df_summary_all.set_index("routing").reindex(routings)["rtx"].fillna(0).astype(float).values

    bar_width = 0.55
    ax = ax2
    if layout == "stacked":
        # Middle subplot: dropped packets
        bars = ax.bar(
            positions,
            data_rtx,
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_int(bar.get_height()) for bar in bars]
    elif detail_type == "duration":
        bars = ax.bar(
            positions,                  # use numeric positions
            data_traffic_end, 
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_time(bar.get_height()) for bar in bars]
    elif detail_type == "ooo_avg":
        bars = ax.bar(
            positions,                  # use numeric positions
            ooo_pct_avgs,
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_pct(bar.get_height()) for bar in bars]
    else:
        bars = ax.bar(
            positions,                  # use numeric positions
            percentiles_99, 
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_time(bar.get_height()) for bar in bars]

    ax.grid(False)
    ax.bar_label(
        bars, 
        labels=[label if bar.get_height() > 0 else "" for bar, label in zip(bars, labels)],
        fontsize=5.5 * BAR_ANNOTATION_SCALE, 
        padding=3
    )

    # Add Avg Dist annotations for OOO when shown on ax2 (classic layout)
    if layout != "stacked" and detail_type == "ooo_avg":
        for bar, avgd, maxd in zip(bars, ooo_avgdists, ooo_maxdists):
            if bar.get_height() <= 0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            y = bar.get_height()
            """ ax.annotate(
                f"Avg. Dist: {avgd:.2f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.5,
            ) """

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.tick_params(bottom=False, labelbottom=False)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    # Y-axis config for ax2: remove y ticks and grid (no horizontal lines)
    if layout == "stacked":
        ax.set_yticks([])
        if not hide_y_titles:
            ax.set_ylabel("Dropped\nPackets", fontsize=8 * AXIS_TITLE_SCALE)
        ax.tick_params(labelleft=False)
    else:
        if not hide_y_titles:
            if detail_type == "duration":
                ax.set_ylabel(f"Duration ({unit})", fontsize=8 * AXIS_TITLE_SCALE)
            elif detail_type == "ooo_avg":
                ax.set_ylabel("Avg OOO Packets (%)", fontsize=8 * AXIS_TITLE_SCALE)
            else:
                ax.set_ylabel(f"99th Percentile ({unit})", fontsize=8 * AXIS_TITLE_SCALE)
        ax.set_yticks([])
        ax.tick_params(labelleft=False)
        ax.grid(False)

    # =============
    ax = ax3
    if layout == "stacked":
        # Bottom subplot: OOO percentage
        bars = ax.bar(
            positions,
            ooo_pct_avgs,
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_pct(bar.get_height()) for bar in bars]
        ax.bar_label(bars, labels=[label if bar.get_height() > 0 else "" for bar, label in zip(bars, labels)], fontsize=5.5 * BAR_ANNOTATION_SCALE, padding=3)
        # ...annotation removed as requested...
    else:
        # Classic layout: bottom-right shows dropped packets
        bars = ax.bar(
            positions,
            data_rtx, 
            width=bar_width,
            color=colors,
            zorder=3,
        )
        labels = [format_label_int(bar.get_height()) for bar in bars]
        ax.bar_label(bars, labels=[label if bar.get_height() > 0 else "" for bar, label in zip(bars, labels)], fontsize=5.5 * BAR_ANNOTATION_SCALE, padding=3)

    ax.grid(False)
    ymax = max([bar.get_height() for bar in bars])
    ax.set_ylim(top=ymax * 1.28) 
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    if layout == "stacked":
        ax.set_yticks([])
        if not hide_y_titles:
            ax.set_ylabel("% OOO\nPackets", fontsize=8 * AXIS_TITLE_SCALE)
        ax.tick_params(labelleft=False)
        ax.tick_params(bottom=True, labelbottom=True)
    else:
        ax.set_yticks([])
        if not hide_y_titles:
            ax.set_ylabel("Dropped\nPackets", fontsize=8 * AXIS_TITLE_SCALE)
        ax.tick_params(labelleft=False)
        ax.tick_params(bottom=False, labelbottom=False)

    # Remove any grid/horizontal lines on bottom plot
    ax.grid(False)

    # Ensure full x-range is visible on shared axes
    ax1.set_xlim(-0.5, len(routings) - 0.5)

    # ================================================

    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import blended_transform_factory

    # Define Spritz variant indices (based on your routing list)
    spritz_indices = [7, 8, 9]

    def add_improvement_labels(ax, values, metric_name, shift_down=0.0, draw_spritz=True):
        competitor_values = [v for i, v in enumerate(values) if i not in spritz_indices and v > 0]
        if not competitor_values: return
        best_competitor = min(competitor_values)
        x_competitor = [i for i, v in enumerate(values) if v == best_competitor][0]

        if metric_name == "fct":
            y_base, y_text = 0.93, 0.895
        elif metric_name in ("duration", "ooo"):
            y_base, y_text = 0.875, 0.815
        else:
            y_base, y_text = 0.83, 0.75

        # Move annotations down when requested
        y_base = max(0.0, y_base - shift_down)
        y_text = max(0.0, y_text - shift_down)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        
        ylim = ax.get_ylim()
        if draw_spritz:
            for idx in spritz_indices:
                if values[idx] <= 0: 
                    continue
                improvement = best_competitor / values[idx]
                label = f"{improvement:.1f}x"
                color = "mediumseagreen" if improvement >= 1 else "lightgray"
                ax.text(
                    idx, y_base, label,
                    ha="center", va="bottom",
                    fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"),
                    transform=trans, clip_on=False, zorder=5
                )

        span = (ylim[1] - ylim[0])
        if metric_name == "fct": 
            extra = 0.03
        elif metric_name in ("duration", "ooo"): 
            extra = 0.035
        else: 
            extra = 0.06
        ax.set_ylim(ylim[0], ylim[1] + 2 * extra * span)

        # Star for best competitor (second-best indicator relative to Spritz)
        ax.text(
            x_competitor, y_text, "*",
            ha="center", va="bottom",
            fontsize=14.5,
            transform=trans, clip_on=False, zorder=5
        )
     
    # Apply to each subplot
    add_improvement_labels(ax1, percentiles_99, metric_name="fct", shift_down=0.08, draw_spritz=True)
    if layout == "stacked":
        add_improvement_labels(ax2, data_rtx, metric_name="dropped", shift_down=0.06, draw_spritz=True)
        # Remove the three Spritz annotations from the bottom plot (keep the star)
        add_improvement_labels(ax3, ooo_pct_avgs, metric_name="ooo", shift_down=0.08, draw_spritz=False)
    else:
        add_improvement_labels(
            ax2,
            data_traffic_end if detail_type == "duration" else (ooo_pct_avgs if detail_type == "ooo_avg" else percentiles_99),
            metric_name="duration" if detail_type == "duration" else ("ooo" if detail_type == "ooo_avg" else "duration"),
            shift_down=0.06,
            draw_spritz=True,
        )
        # Remove the three Spritz annotations from the bottom plot (keep the star)
        add_improvement_labels(ax3, data_rtx, metric_name="dropped", shift_down=0.06, draw_spritz=False)

    # Ensure axes borders are always above other artists
    for _ax in (ax1, ax2, ax3):
        for spine in _ax.spines.values():
            spine.set_zorder(100)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(f"{experiment}_thesis_plot.pdf", bbox_inches="tight")
    if show:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Alternative thesis-style plot using simulation outputs.")
    ap.add_argument("--experiment", required=True, help="Experiment folder name.")
    ap.add_argument("--df-topo", default="p4a8h4", help="Dragonfly topology folder (default p4a8h4).")
    ap.add_argument("--sf-topo", help="Slim Fly topology folder (e.g., p7q9).")
    ap.add_argument("--output-root", default="experiments_output", help="Root output directory (default experiments_output).")
    ap.add_argument("--scenario", choices=["no_fail", "fail_2p"], default="no_fail", help="Which scenario subfolder to use (default no_fail).")
    ap.add_argument("--unit", choices=["us", "ms"], default="us", help="Time unit for FCT (default us).")
    ap.add_argument("--srcs-filter", type=int, choices=[0,1,2], default=0, help="0=all,1=only chosen subset,2=exclude subset.")
    ap.add_argument("--detail-type", choices=["duration","percentiles_99","ooo_avg"], default="duration", help="Second panel metric (classic layout only).")
    ap.add_argument("--log-scale", action="store_true", help="Log scale for FCT violins.")
    ap.add_argument("--show-x-labels", action="store_true", help="Show x labels under all subplots.")
    ap.add_argument("--outfile", default=None, help="Output PDF filename.")
    ap.add_argument("--layout", choices=["stacked","classic"], default="stacked", help="Figure layout: stacked (default) or classic.")
    ap.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    ap.add_argument("--no-y-titles", action="store_true", help="Disable y-axis titles on all subplots.")
    ap.add_argument("--fig-width", type=float, default=3.5, help="Figure width in inches (default 3.6).")
    ap.add_argument("--flow-kind", choices=["incast", "bystanders"], default=None, help="Filter flows: incast (to target dst) or bystanders (all others). Requires --incast-dst.")
    ap.add_argument("--incast-dst", type=int, default=None, help="Destination node ID for incast flows (e.g., 160 for DF, 161 for SF).")
    args = ap.parse_args()

    if args.flow_kind and args.incast_dst is None:
        ap.error("--flow-kind requires --incast-dst")

    # Use Slim Fly when --sf-topo is provided; otherwise default to Dragonfly.
    if args.sf_topo:
        base_path = os.path.join(args.output_root, "sf", args.sf_topo, args.scenario, args.experiment)
    else:
        base_path = os.path.join(args.output_root, "df", args.df_topo, args.scenario, args.experiment)

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    create_thesis_plot(
        base_path=base_path,
        experiment=args.experiment,
        unit=args.unit,
        srcs_filter=args.srcs_filter,
        detail_type=args.detail_type,
        log_scale=args.log_scale,
        show_x_labels=args.show_x_labels,
        outfile=args.outfile,
        layout=args.layout,
        show=not args.no_show,
        hide_y_titles=args.no_y_titles,
        fig_width=args.fig_width,
        flow_kind=args.flow_kind,
        incast_dst=args.incast_dst,
    )

if __name__ == "__main__":
    main()
