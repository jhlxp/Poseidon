#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path


PAPER_FLAGS = [
    "-flow-explore-threshold", "44",
    "-flow-ecn-threshold", "8",
    "-flow-sort-insert", "1",
    "-flow-small-flows-bias", "1",
    "-flow-small-flows-threshold", str(2**19),
    "-flow-small-flows-weight", "100.0",
]


SUMMARY_COLUMNS = [
    "algorithm",
    "routing",
    "lb",
    "weight_scaling",
    "runtime_s",
    "flows",
    "avg_FCT_ns",
    "max_FCT_ns",
    "traffic_end_ns",
    "newPkts",
    "rtxPkts",
    "ackPkts",
    "nackPkts",
]


ALGORITHMS = [
    {"name": "MINIMAL", "routing": "MINIMAL"},
    {"name": "VALIANT", "routing": "VALIANT"},
    {"name": "UGAL_L", "routing": "UGAL_L"},
    {"name": "ECMP", "routing": "SOURCE", "lb": "ECMP", "weight_scaling": "0"},
    {"name": "OPS_U", "routing": "SOURCE", "lb": "OPS", "weight_scaling": "0"},
    {"name": "OPS_W", "routing": "SOURCE", "lb": "OPS", "weight_scaling": "3"},
    {"name": "FLICR", "routing": "SOURCE", "lb": "FLICR", "weight_scaling": "3"},
    {"name": "SPRITZ_SCOUT", "routing": "SOURCE", "lb": "FLOW_V1", "weight_scaling": "3"},
    {"name": "SPRITZ_SPRAY_U", "routing": "SOURCE", "lb": "FLOW_V2", "weight_scaling": "0"},
    {"name": "SPRITZ_SPRAY_W", "routing": "SOURCE", "lb": "FLOW_V2", "weight_scaling": "3"},
]


def clear_metrics(metrics_dir):
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)


def read_metrics(metrics_dir):
    flows_path = metrics_dir / "flowsInfo.csv"
    packets_path = metrics_dir / "packetInfo.csv"
    with flows_path.open() as f:
        flows = list(csv.DictReader(f))
    fcts = [float(row["fctNs"]) for row in flows]
    with packets_path.open() as f:
        packets = next(csv.DictReader(f))
    return {
        "flows": len(flows),
        "avg_FCT_ns": sum(fcts) / len(fcts),
        "max_FCT_ns": max(fcts),
        "traffic_end_ns": max(float(row["endTimeNs"]) for row in flows),
        "newPkts": int(packets["newPkts"]),
        "rtxPkts": int(packets["rtxPkts"]),
        "ackPkts": int(packets["ackPkts"]),
        "nackPkts": int(packets["nackPkts"]),
    }


def move_metrics(metrics_dir, output_dir, algorithm):
    dst = output_dir / f"{algorithm}_metrics"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(metrics_dir), dst)


def load_summary(summary_path):
    if not summary_path.exists():
        return []
    with summary_path.open() as f:
        return list(csv.DictReader(f))


def write_summary(summary_path, rows):
    order = {algo["name"]: index for index, algo in enumerate(ALGORITHMS)}
    rows = sorted(rows, key=lambda row: order.get(row["algorithm"], len(order)))
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run a compact Spritz reproduction subset on the paper Dragonfly workload."
    )
    parser.add_argument("--topology", default="p4a8h4")
    parser.add_argument("--workload", default="permutation_global_4MiB.cm")
    parser.add_argument("--cc", default="SMARTT_ECN")
    parser.add_argument("--output-dir", default="experiments_output/spritz_subset")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[algo["name"] for algo in ALGORITHMS],
        help="Optional subset of algorithms to run.",
    )
    args = parser.parse_args()

    workdir = Path.cwd()
    metrics_dir = workdir / "output_metrics"
    output_dir = workdir / args.output_dir / args.topology / Path(args.workload).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    input_dir = Path("experiments") / "df" / args.topology
    topo_dir = Path("topologies") / "dragonfly" / args.topology

    algorithms = [algo for algo in ALGORITHMS if args.only is None or algo["name"] in args.only]
    rows = load_summary(summary_path)
    for algo in algorithms:
        clear_metrics(metrics_dir)
        log_path = output_dir / f"{algo['name']}.log"
        cmd = [
            "./htsim_uec_df",
            "-basepath", f"./{topo_dir}",
            "-tm", f"./{input_dir / args.workload}",
            "-p", args.topology.split("a", 1)[0][1:],
            "-routing", algo["routing"],
            "-CC", args.cc,
        ]
        if algo["routing"] == "SOURCE":
            cmd.extend([
                "-LB", algo["lb"],
                "-flow-weight-scaling", algo["weight_scaling"],
                *PAPER_FLAGS,
            ])
        start = time.time()
        with log_path.open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        elapsed_s = time.time() - start
        metrics = read_metrics(metrics_dir)
        move_metrics(metrics_dir, output_dir, algo["name"])
        row = {
            "algorithm": algo["name"],
            "routing": algo["routing"],
            "lb": algo.get("lb", ""),
            "weight_scaling": algo.get("weight_scaling", ""),
            "runtime_s": f"{elapsed_s:.3f}",
            **metrics,
        }
        rows = [existing for existing in rows if existing["algorithm"] != algo["name"]]
        rows.append(row)
        write_summary(summary_path, rows)

    print(summary_path)


if __name__ == "__main__":
    main()
