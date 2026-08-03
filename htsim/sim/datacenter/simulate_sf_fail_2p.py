import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run slimfly simulations with 2% failed links.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of variant runs to execute in parallel.")
    parser.add_argument("--only-experiment", type=str, default=None, help="Run only the experiment file with this base name (without extension).")
    parser.add_argument("--cc", type=str, default="SMARTT_ECN", help="Congestion control (default SMARTT_ECN).")
    parser.add_argument("--output-root", type=str, default="experiments_output", help="Output root directory (default: experiments_output).")
    parser.add_argument("--verbose", action="store_true", help="Print each command before executing.")
    parser.add_argument("--only-lb-name", type=str, default=None, help="Only run SOURCE+LB variants with this lb_name (e.g., ECMP, OPS_U, OPS_W, FLICR, FLOW_V1, FLOW_V2_U, FLOW_V2_W).")
    return parser.parse_args()


def extract_metrics(output_metrics_dir: str):
    flows_path = os.path.join(output_metrics_dir, "flowsInfo.csv")
    packets_path = os.path.join(output_metrics_dir, "packetInfo.csv")

    flows_df = pd.read_csv(flows_path)
    max_time = flows_df["fctNs"].max()
    avg_time = flows_df["fctNs"].mean()
    traffic_end = flows_df["endTimeNs"].max()

    pkt_df = pd.read_csv(packets_path)
    return {
        "max_FCT": max_time,
        "avg_FCT": avg_time,
        "traffic_end": traffic_end,
        "new": int(pkt_df["newPkts"].iloc[0]),
        "rtx": int(pkt_df["rtxPkts"].iloc[0]),
        "ack": int(pkt_df["ackPkts"].iloc[0]),
    }


def run_variant(binary_path: str, basepath_abs: str, tm_abs: str, p: int, cc: str, output_experiment_dir: str, experiment_name: str, variant: dict, verbose: bool, logs_dir: str):
    tag = variant["tag"]
    routing = variant["routing_arg"]
    routing_label = variant["routing_label"]

    work_parent = os.path.join(output_experiment_dir, "_tmp")
    os.makedirs(work_parent, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{tag}_", dir=work_parent) as tmpdir:
        start = time.time() * 1000.0
        cmd = [
            binary_path,
            "-basepath", basepath_abs,
            "-tm", tm_abs,
            "-fail-percentage", "0.02",
            "-p", str(p),
            "-routing", routing,
            "-CC", cc,
        ]

        if routing == "SOURCE":
            cmd += [
                "-LB", variant["lb"],
                "-flow-explore-threshold", "44",
                "-flow-ecn-threshold", "8",
                "-flow-weight-scaling", str(variant["weight_scaling"]),
                "-flow-sort-insert", "1",
                "-flow-small-flows-bias", "1",
                "-flow-small-flows-threshold", str(2**19),
                "-flow-small-flows-weight", "100.0",
            ]

        log_path = os.path.join(logs_dir, f"{experiment_name}__{tag}.log")
        if verbose:
            print(" ".join(cmd))

        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n")
            lf.flush()
            subprocess.run(cmd, cwd=tmpdir, stdout=lf, stderr=lf, check=True)

        end = time.time() * 1000.0

        metrics_dir = os.path.join(tmpdir, "output_metrics")
        metrics = extract_metrics(metrics_dir)

        flows_src = os.path.join(metrics_dir, "flowsInfo.csv")
        flows_dst = os.path.join(output_experiment_dir, f"{experiment_name}_{routing_label}_flows.csv")
        shutil.copy2(flows_src, flows_dst)

        row = {
            "routing": routing_label,
            "runtime": end - start,
            **metrics,
        }
        return row


if __name__ == '__main__':
    args = parse_args()

    p, q = (7, 9)
    topo = f"p{p}q{q}"
    input_dir = f"experiments/sf/{topo}"
    output_dir = os.path.join(args.output_root, "sf", topo, "fail_2p")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    binary_path = os.path.join(repo_root, "htsim/sim/datacenter/htsim_uec_sf")
    basepath_abs = os.path.abspath(os.path.join(repo_root, "htsim/sim/datacenter/topologies", "slimfly", topo))

    experiment_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".cm") and os.path.isfile(os.path.join(input_dir, f))
    ]
    if args.only_experiment:
        experiment_files = [f for f in experiment_files if f.split(".")[0] == args.only_experiment]
    if not experiment_files:
        print("No experiment files to run (check --only-experiment filter).")
        raise SystemExit(0)

    for experiment_file in sorted(experiment_files):
        experiment_name = experiment_file.split(".")[0]
        output_experiment_dir = os.path.join(output_dir, experiment_name)
        logs_dir = os.path.join(output_experiment_dir, "run_logs")
        os.makedirs(logs_dir, exist_ok=True)

        tm_abs = os.path.abspath(os.path.join(repo_root, "htsim/sim/datacenter", input_dir, experiment_file))

        variants = []
        for routing in ["MINIMAL", "VALIANT", "UGAL_L"]:
            variants.append({
                "routing_label": routing,
                "routing_arg": routing,
                "tag": routing,
            })

        weight_scaling_list = [0, 0, 3, 3, 3, 0, 3]
        lb_list = ["ECMP", "OPS", "OPS", "FLICR", "FLOW_V1", "FLOW_V2", "FLOW_V2"]
        lb_names_list = ["ECMP", "OPS_U", "OPS_W", "FLICR", "FLOW_V1", "FLOW_V2_U", "FLOW_V2_W"]
        for ws, lb, lb_name in zip(weight_scaling_list, lb_list, lb_names_list):
            if args.only_lb_name and lb_name != args.only_lb_name:
                continue
            variants.append({
                "routing_label": f"SOURCE_{lb_name}",
                "routing_arg": "SOURCE",
                "lb": lb,
                "weight_scaling": ws,
                "tag": f"SOURCE_{lb_name}_ws{ws}",
            })

        rows = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as ex:
            futs = [
                ex.submit(
                    run_variant,
                    binary_path,
                    basepath_abs,
                    tm_abs,
                    p,
                    args.cc,
                    output_experiment_dir,
                    experiment_name,
                    v,
                    args.verbose,
                    logs_dir,
                )
                for v in variants
            ]
            for fut in as_completed(futs):
                rows.append(fut.result())

        for r in rows:
            r.update({"p": p, "q": q})

        df = pd.DataFrame(rows)
        df.sort_values(by=["routing"], inplace=True)
        df.to_csv(os.path.join(output_experiment_dir, f"{experiment_name}_summary.csv"), index=False)

