import os
import time
import subprocess
import shutil
import pandas as pd
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def extract_metrics(metrics_dir):
    flows_csv = os.path.join(metrics_dir, "flowsInfo.csv")
    packets_csv = os.path.join(metrics_dir, "packetInfo.csv")
    df_flows = pd.read_csv(flows_csv)
    max_time = df_flows["fctNs"].max()
    avg_time = df_flows["fctNs"].mean()
    traffic_end = df_flows["endTimeNs"].max()
    df_packets = pd.read_csv(packets_csv)
    return [max_time, avg_time, traffic_end, df_packets["newPkts"][0], df_packets["rtxPkts"][0], df_packets["ackPkts"][0]]


def clear_metrics(metrics_dir):
    if os.path.exists(metrics_dir):
        for f in os.listdir(metrics_dir):
            fp = os.path.join(metrics_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)


def move_metrics(metrics_dir, dst):
    src = os.path.join(metrics_dir, "flowsInfo.csv")
    shutil.move(src, dst)


def build_output_filename(experiment_name, tag, args, p, q):
    return f"{experiment_name}_p{p}q{q}_{args.cc}_{tag}.tmp"


def parse_args():
    parser = argparse.ArgumentParser(description="Run slimfly simulations (optional parallelism and single experiment selection).")
    parser.add_argument("--parallel", type=int, default=1, help="Number of variant runs to execute in parallel.")
    parser.add_argument("--only-experiment", type=str, default=None, help="Run only the experiment file with this base name (without extension).")
    parser.add_argument("--cc", type=str, default="SMARTT_ECN", help="Congestion control (default SMARTT_ECN).")
    parser.add_argument("--output-root", type=str, default="experiments_output", help="Output root directory (default: experiments_output).")
    parser.add_argument("--verbose", action="store_true", help="Print each command before executing.")
    parser.add_argument("--only-lb-name", type=str, default=None, help="Only run SOURCE+LB variants with this lb_name (e.g., ECMP, OPS_U, OPS_W, FLICR, FLOW_V1, FLOW_V2_U, FLOW_V2_W).")
    return parser.parse_args()


# Run a single variant (routing or routing+LB) in its own temp workspace
def run_variant(task):
    (experiment_file, variant, ctx) = task
    experiment_name = experiment_file.split(".")[0]
    routing_label = variant["routing_label"]  # value stored in summary 'routing' column
    routing_arg = variant["routing_arg"]      # value for -routing
    lb = variant.get("lb")
    weight_scaling = variant.get("weight_scaling")
    lb_name = variant.get("lb_name")
    tag = variant["tag"]  # for log filename uniqueness
    args = ctx["args"]
    p = ctx["p"]; q = ctx["q"]
    binary_path = ctx["binary_path"]
    basepath_abs = ctx["basepath_abs"]
    tm_abs = os.path.abspath(os.path.join(ctx["input_dir"], experiment_file))
    output_experiment_dir = ctx["exp_dirs"][experiment_name]["root"]
    run_logs_dir = ctx["exp_dirs"][experiment_name]["logs"]
    flows_dest_name = f"{experiment_name}_{routing_label}_flows.csv".replace("__","_")
    flows_dest_path = os.path.join(output_experiment_dir, flows_dest_name)

    with tempfile.TemporaryDirectory(prefix="sf_variant_") as workdir:
        metrics_dir = os.path.join(workdir, "output_metrics")
        cmd = [
            binary_path,
            "-basepath", basepath_abs,
            "-tm", tm_abs,
            "-p", f"{p}",
            "-routing", routing_arg,
            "-CC", args.cc
        ]
        if lb:
            cmd += [
                "-LB", lb,
                "-flow-explore-threshold", "44",
                "-flow-ecn-threshold", "8",
                "-flow-weight-scaling", f"{weight_scaling}",
                "-flow-sort-insert", "1",
                "-flow-small-flows-bias", "1",
                "-flow-small-flows-threshold", f"{2**19}",
                "-flow-small-flows-weight", "100.0"
            ]
        start = time.time() * 1000.0
        if args.verbose:
            print(f"[RUN] {' '.join(cmd)} (cwd={workdir})")
        log_path = os.path.join(run_logs_dir, build_output_filename(experiment_name, tag, args, p, q))
        with open(log_path, "w") as outf:
            subprocess.run(cmd, check=True, cwd=workdir, stdout=outf, stderr=subprocess.STDOUT)
        end = time.time() * 1000.0
        metrics = extract_metrics(metrics_dir)
        move_metrics(metrics_dir, flows_dest_path)
        clear_metrics(metrics_dir)
    result = {
        "p": p, "q": q,
        "routing": routing_label,
        "runtime": end - start,
        "max_FCT": metrics[0],
        "avg_FCT": metrics[1],
        "traffic_end": metrics[2],
        "new": metrics[3],
        "rtx": metrics[4],
        "ack": metrics[5]
    }
    return experiment_name, result


if __name__ == '__main__':
    args = parse_args()
    p, q = (7, 9)
    topo = f"p{p}q{q}"
    input_dir = f"experiments/sf/{topo}"
    output_dir = os.path.join(args.output_root, "sf", topo, "no_fail")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    print(f"Repo root: {repo_root}")
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
        exit(0)

    # Prepare directories per experiment
    exp_dirs = {}
    for ef in experiment_files:
        en = ef.split(".")[0]
        root_dir = os.path.join(output_dir, en)
        logs_dir = os.path.join(root_dir, "run_logs")
        os.makedirs(logs_dir, exist_ok=True)
        exp_dirs[en] = {"root": root_dir, "logs": logs_dir}

    # Build variant task list
    tasks = []
    for ef in experiment_files:
        # Basic routings
        for routing in ["MINIMAL", "VALIANT", "UGAL_L"]:
            tasks.append((
                ef,
                {
                    "routing_label": routing,
                    "routing_arg": routing,
                    "tag": routing
                },
                {
                    "args": args, "p": p, "q": q,
                    "binary_path": binary_path,
                    "basepath_abs": basepath_abs,
                    "input_dir": input_dir,
                    "exp_dirs": exp_dirs
                }
            ))
        # SOURCE with LB variants
        weight_scaling_list = [0, 0, 3, 3, 3, 0, 3]
        lb_list = ["ECMP", "OPS", "OPS", "FLICR", "FLOW_V1", "FLOW_V2", "FLOW_V2"]
        lb_names_list = ["ECMP", "OPS_U", "OPS_W", "FLICR", "FLOW_V1", "FLOW_V2_U", "FLOW_V2_W"]
        for ws, lb, lb_name in zip(weight_scaling_list, lb_list, lb_names_list):
            if args.only_lb_name and lb_name != args.only_lb_name:
                continue
            tasks.append((
                ef,
                {
                    "routing_label": f"SOURCE_{lb_name}",
                    "routing_arg": "SOURCE",
                    "lb": lb,
                    "weight_scaling": ws,
                    "lb_name": lb_name,
                    "tag": f"SOURCE_{lb_name}_ws{ws}"
                },
                {
                    "args": args, "p": p, "q": q,
                    "binary_path": binary_path,
                    "basepath_abs": basepath_abs,
                    "input_dir": input_dir,
                    "exp_dirs": exp_dirs
                }
            ))

    # Thread-safe per-experiment results accumulation
    results_lock = threading.Lock()
    results_by_exp = {ef.split(".")[0]: [] for ef in experiment_files}

    max_workers = max(1, args.parallel)
    if args.verbose:
        print(f"Scheduling {len(tasks)} variant runs with parallelism={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_variant, task): task for task in tasks}
        for fut in as_completed(future_map):
            ef, variant, _ = future_map[fut]
            try:
                experiment_name, result = fut.result()
                with results_lock:
                    results_by_exp[experiment_name].append(result)
                    # Update summary CSV after each completion
                    df_summary = pd.DataFrame(results_by_exp[experiment_name])
                    summary_path = os.path.join(exp_dirs[experiment_name]["root"], f"{experiment_name}_summary.csv")
                    df_summary.to_csv(summary_path, index=False)
                if args.verbose:
                    print(f"[DONE] {experiment_name} :: {result['routing']}")
            except Exception as e:
                print(f"[ERROR] Variant for {ef} failed: {e}")

