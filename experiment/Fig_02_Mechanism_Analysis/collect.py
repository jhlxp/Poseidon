#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile


ALGORITHMS = ("nccl", "deepep", "eplb", "moonep", "probeep")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty mechanism data: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def total(value: object) -> int:
    if isinstance(value, dict):
        return sum(int(item) for item in value.values())
    if isinstance(value, list):
        return sum(int(item) for item in value)
    return int(value)


def main() -> int:
    options = args()
    with options.manifest.open(newline="", encoding="utf-8") as handle:
        source = Path(next(csv.DictReader(handle))["source_run"]).resolve()
    options.data_dir.mkdir(parents=True, exist_ok=True)
    load_rows: list[dict[str, object]] = []
    planning_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []

    for hardware in ("H20", "H100"):
        candidates = list(source.glob(f"{hardware}_*visualization.zip"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {hardware} visualization ZIP")
        with zipfile.ZipFile(candidates[0]) as archive:
            for algorithm in ALGORITHMS:
                prefix = f"algorithms/{algorithm}"
                gate = json.loads(
                    archive.read(f"{prefix}/gate_load/gate_load_profile_summary.json")
                )
                for record in gate["records"]:
                    for phase in ("before", "after"):
                        profile = record["profile"][phase]
                        load_rows.append(
                            {
                                "hardware": hardware,
                                "algorithm": algorithm,
                                "layer": record["layer"],
                                "micro_batch": record["micro_batch"],
                                "phase": phase,
                                "rank_max_mean": profile["rank_imbalance"]["max_mean"],
                                "server_max_mean": profile["server_imbalance"]["max_mean"],
                                "instance_max_mean": profile["instance_imbalance"]["max_mean"],
                                "assignment_digest": record["gate"]["assignment_digest_sha256"],
                            }
                        )
                    planning = record["planning"]
                    transport = record["transport"]
                    moved_routes = sum(
                        int(pair.get("moved_expert_routes", 0))
                        for pair in transport.get("server_pairs", [])
                    )
                    planning_rows.append(
                        {
                            "hardware": hardware,
                            "algorithm": algorithm,
                            "layer": record["layer"],
                            "micro_batch": record["micro_batch"],
                            "baseline_padded_routes": total(planning["baseline_server_padded_routes"]),
                            "planned_padded_routes": total(planning["planned_server_padded_routes"]),
                            "admitted_padded_routes": total(planning["admitted_server_padded_routes"]),
                            "planned_intents": planning["planned_intent_count"],
                            "admitted_intents": planning["admitted_intent_count"],
                            "deferred_intents": planning["deferred_intent_count"],
                            "remote_replicas": transport["remote_replica_count"],
                            "moved_routes": moved_routes,
                            "expert_weight_bytes": transport["expert_weight_bytes"],
                        }
                    )
                timeline = json.loads(
                    archive.read(f"{prefix}/timeline/dag_timeline_summary.json")
                )
                network_rows.append(
                    {
                        "hardware": hardware,
                        "algorithm": algorithm,
                        "makespan_us": timeline["makespan_us"],
                        "compute_active_us": timeline["selected_rank_compute_active_sum_us"],
                        "network_active_us": timeline["selected_rank_network_active_sum_us"],
                        "overlap_us": timeline["selected_rank_compute_network_overlap_sum_us"],
                        "logical_transfer_bytes": timeline["logical_transfer_bytes"],
                        "weight_rdma_bytes": timeline.get("expert_weight_rdma_bytes", 0),
                        "weight_local_bytes": timeline.get("expert_weight_local_bytes", 0),
                    }
                )
    write_csv(options.data_dir / "load_balance.csv", load_rows)
    write_csv(options.data_dir / "planning.csv", planning_rows)
    write_csv(options.data_dir / "network_cost.csv", network_rows)
    metadata = {
        "schema": "probeep_mechanism_analysis_v1",
        "mode": options.mode,
        "paper_eligible": options.mode == "full",
        "source": str(source),
        "evidence": "trace_analysis_of_packet_simulation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (options.data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
