#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def first_row(path: Path, direction: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["direction"] == direction)


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifests = list(csv.DictReader(handle))
    output: list[dict[str, object]] = []
    for item in manifests:
        source = Path(item["source_run"])
        prefix = source / ("algorithms/moonep/link_load" if item["algorithm"] == "moonep" else "link_load")
        endpoints = prefix / "mprail_endpoint_load_summary.csv"
        links = prefix / "mprail_link_load_summary.csv"
        if not endpoints.is_file() or not links.is_file():
            raise FileNotFoundError(f"missing link-load summary under {prefix}")
        tx = first_row(endpoints, "output")
        rx = first_row(endpoints, "input")
        with links.open(newline="", encoding="utf-8") as handle:
            link_rows = list(csv.DictReader(handle))
        output.append(
            {
                **{key: value for key, value in item.items() if key != "source_run"},
                "endpoint_peak_utilization": max(float(tx["peak_utilization"]), float(rx["peak_utilization"])),
                "endpoint_mean_utilization": max(float(tx["mean_utilization"]), float(rx["mean_utilization"])),
                "peak_link_gbps": max(float(row["throughput_active_max_gbps"] or 0) for row in link_rows),
                "max_queue_bytes": max(int(float(row["max_queue_bytes"] or 0)) for row in link_rows),
            }
        )
    with (args.data_dir / "network_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
