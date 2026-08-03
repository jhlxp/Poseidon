import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Plot FCT violins from simulate_df_no_fail.py outputs.")
    p.add_argument("--experiment", required=True, help="Experiment name (folder created by simulator).")
    p.add_argument("--output-root", default="experiments_output", help="Root output directory (default: experiments_output).")
    p.add_argument("--df-topo", default="p4a8h4", help="Dragonfly topology folder (default: p4a8h4).")
    p.add_argument("--sf-topo", default="p7q9", help="Slim Fly topology folder (if present).")
    p.add_argument("--scenario", choices=["no_fail", "fail_2p"], default="no_fail", help="Which scenario subfolder to plot (default: no_fail).")
    p.add_argument("--no-sf", action="store_true", help="Disable Slim Fly (bottom) plot.")
    p.add_argument("--no-df", action="store_true", help="Disable Dragonfly (top) plot.")
    p.add_argument("--outfile", default=None, help="PDF output filename (default: poster_child_<experiment>.pdf).")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window (useful for batch/headless runs).")
    return p.parse_args()

args = parse_args()

if args.no_show:
    # Ensure headless-safe rendering
    plt.switch_backend("Agg")

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
tick_labels = ["Minimal", "Valiant", "UGAL-L", "ECMP", "FLICR", "OPS (u)", "OPS (w)", "Spritz-Scout (w)", "Spritz-Spray (u)", "Spritz-Spray (w)"]
colors = ["#262636", "#7e2c40", "#a42f44", "#3885cf"]
colors = ["#262636", "#743141", "#973846", "#c8593e", "#e08751", "#44888c", "#2b595b", "#454486", "#7c7adf", "#4d83c9"]

experiment = args.experiment

# Determine whether DF / SF data exists (unless forced off)
df_base_path = os.path.join(args.output_root, "df", args.df_topo, args.scenario, experiment)
sf_base_path = os.path.join(args.output_root, "sf", args.sf_topo, args.scenario, experiment)
have_df = (not args.no_df) and os.path.isdir(df_base_path)
have_sf = (not args.no_sf) and os.path.isdir(sf_base_path)

# Plotting setup
rows = int(have_df) + int(have_sf)
if rows == 0:
    print(f"Error: no data found for experiment '{experiment}' in DF or SF paths.")
    import sys; sys.exit(1)
row_heights = [1.2] * rows
total_height = sum(row_heights)
#fig = plt.figure(figsize=(1.7, total_height))
fig = plt.figure(figsize=(4.7, total_height*2))
ratios = [h / total_height for h in row_heights]
gs = gridspec.GridSpec(rows, 1, height_ratios=ratios, hspace=0.05 if rows == 2 else 0.15)

def load_dataset(base_path, routings):
    data = []
    p99 = []
    for routing in routings:
        csv_path = os.path.join(base_path, f"{experiment}_{routing}_flows.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing flows file: {csv_path}")
        df = pd.read_csv(csv_path)
        fct_us = (df["fctNs"] / 1000.0).dropna()
        data.append(fct_us.values)
        p99.append(np.percentile(fct_us.values, 99))
    return data, p99

current_row = 0

# Dragonfly (top) — only if data exists
if have_df:
    ax1 = fig.add_subplot(gs[current_row])
    df_data, df_p99 = load_dataset(df_base_path, routings)
    positions = np.arange(len(df_data))
    parts = ax1.violinplot(df_data, showmeans=True, showmedians=False, showextrema=True, positions=positions)
    for i, color in enumerate(colors):
        parts["bodies"][i].set_facecolor(color)
        parts["bodies"][i].set_alpha(0.5)
    for k in ["cbars", "cmeans", "cmaxes", "cmins"]:
        parts[k].set_color(colors)
    ax1.scatter(positions, df_p99, color="black", marker="D", s=22, zorder=3, label="99th Percentile")
    ax1.grid(color="gray", linestyle=":", linewidth=0.5, axis="y")
    ax1.set_xticks(positions)
    ax1.tick_params(axis="y", labelsize=9)
    if have_sf:
        ax1.tick_params(bottom=False, labelbottom=False)
    else:
        ax1.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        for label in ax1.get_xticklabels():
            label.set_horizontalalignment("right")
    current_row += 1

# Slim Fly (bottom) if available
if have_sf:
    if not have_df:
        positions = np.arange(len(routings))
    share_ax = ax1 if have_df else None
    ax2 = fig.add_subplot(gs[current_row], sharex=share_ax)
    try:
        sf_data, sf_p99 = load_dataset(sf_base_path, routings)
        parts = ax2.violinplot(sf_data, showmeans=True, showmedians=False, showextrema=True, positions=positions)
        for i, color in enumerate(colors):
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.5)
        for k in ["cbars", "cmeans", "cmaxes", "cmins"]:
            parts[k].set_color(colors)
        ax2.scatter(positions, sf_p99, color="black", marker="D", s=22, zorder=3, label="99th Percentile")
        ax2.grid(color="gray", linestyle=":", linewidth=0.5, axis="y")
        ax2.set_xticks(positions)
        ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        for label in ax2.get_xticklabels():
            label.set_horizontalalignment("right")
        ax2.tick_params(axis="y", labelsize=9)
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping Slim Fly plot.")
        fig.delaxes(ax2)

plt.tight_layout()
outfile = args.outfile or f"poster_child_{experiment}.pdf"
plt.savefig(outfile, bbox_inches="tight")
if not args.no_show:
    plt.show()