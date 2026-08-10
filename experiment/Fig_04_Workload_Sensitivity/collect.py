#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifests = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for item in manifests:
        source = Path(item["source_run"])
        if item["algorithm"] == "moonep":
            path = source / "algorithms/moonep/gate_load/gate_load_profile_summary.json"
        else:
            path = source / "gate_load/gate_load_profile_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing Gate profile: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))["records"]
        rank = [float(record["gate"]["rank_imbalance"]["max_mean"]) for record in records]
        server = [float(record["gate"]["server_imbalance"]["max_mean"]) for record in records]
        rows.append(
            {
                **{key: value for key, value in item.items() if key != "source_run"},
                "rank_max_mean": statistics.fmean(rank),
                "server_max_mean": statistics.fmean(server),
                "rank_max_mean_max": max(rank),
                "server_max_mean_max": max(server),
                "routing_fidelity": json.loads(path.read_text(encoding="utf-8"))["routing_provider"]["routing_fidelity"],
            }
        )
    with (args.data_dir / "gate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
