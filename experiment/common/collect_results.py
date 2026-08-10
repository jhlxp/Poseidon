#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable


STANDARD_FIELDS = (
    "case_id",
    "hardware",
    "algorithm",
    "status",
    "makespan_us",
    "task_count",
    "logical_transfer_bytes",
    "remote_replicas",
    "tokens_per_rank_per_microbatch",
    "num_layers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--experiment-type", required=True)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scalar(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def merge_details(base: dict[str, object], details: dict[str, object]) -> dict[str, object]:
    row = dict(base)
    for key, value in details.items():
        if key not in row or row[key] in (None, ""):
            row[key] = scalar(value)
    return row


def expand_summary(manifest: dict[str, str]) -> list[dict[str, object]]:
    source = Path(manifest["source_run"]).resolve()
    summary_path = source / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing source summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base: dict[str, object] = {
        key: value for key, value in manifest.items() if key != "source_run"
    }
    base["source_run"] = str(source)

    rows: list[dict[str, object]] = []
    if isinstance(summary.get("results"), dict):
        for hardware, algorithms in summary["results"].items():
            for algorithm, details in algorithms.items():
                row = dict(base, hardware=hardware, algorithm=algorithm)
                row = merge_details(row, details)
                row.setdefault(
                    "tokens_per_rank_per_microbatch",
                    summary.get("tokens_per_rank_per_microbatch", ""),
                )
                row.setdefault("num_layers", summary.get("num_layers", ""))
                rows.append(row)
        return rows

    if isinstance(summary.get("algorithms"), list):
        mode = summary.get("mode", {})
        for details in summary["algorithms"]:
            row = merge_details(base, details)
            row.setdefault("hardware", manifest.get("hardware", ""))
            row.setdefault(
                "tokens_per_rank_per_microbatch",
                mode.get("tokens_per_rank", "") if isinstance(mode, dict) else "",
            )
            row.setdefault("num_layers", manifest.get("num_layers", ""))
            rows.append(row)
        return rows

    row = merge_details(base, summary)
    row.setdefault("algorithm", manifest.get("algorithm", "probeep"))
    row.setdefault("hardware", manifest.get("hardware", ""))
    rows.append(row)
    return rows


def field_order(rows: Iterable[dict[str, object]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    front = [key for key in STANDARD_FIELDS if key in keys]
    return front + [key for key in keys if key not in front]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty data: {path}")
    fields = field_order(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_observations(manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for manifest in manifest_rows:
        source = Path(manifest["source_run"]).resolve()
        path = source / "probeep_dispatch_observations.csv"
        if not path.is_file():
            continue
        dimensions = {
            key: value for key, value in manifest.items() if key != "source_run"
        }
        for record in read_csv(path):
            output.append({**dimensions, "source_run": str(source), **record})
    return output


def main() -> int:
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    if not manifest_rows:
        raise RuntimeError(f"empty source manifest: {args.manifest}")
    results = [
        row for manifest in manifest_rows for row in expand_summary(manifest)
    ]
    write_csv(args.data_dir / "results.csv", results)
    observations = collect_observations(manifest_rows)
    if observations:
        write_csv(args.data_dir / "observations.csv", observations)
    metadata = {
        "schema": "probeep_experiment_type_v1",
        "experiment_type": args.experiment_type,
        "mode": args.mode,
        "paper_eligible": args.mode == "full",
        "case_count": len(manifest_rows),
        "result_count": len(results),
        "observation_count": len(observations),
        "source_manifest": str(args.manifest.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
