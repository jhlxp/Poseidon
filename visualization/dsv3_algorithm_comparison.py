#!/usr/bin/env python3
"""Build a DSV3 comparison dashboard and optional visualization-only ZIP."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use ALGORITHM=CASE_DIR")
    algorithm, raw_path = value.split("=", 1)
    if not algorithm or not raw_path:
        raise argparse.ArgumentTypeError("case must use ALGORITHM=CASE_DIR")
    return algorithm, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        help="Algorithm case directory as ALGORITHM=CASE_DIR; repeat per algorithm.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zip-output",
        type=Path,
        help="Optional ZIP containing only the dashboard and visualization assets.",
    )
    parser.add_argument("--title", default="DSV3 EP32 Algorithm Comparison")
    return parser.parse_args()


def relative_url(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output.parent.resolve())).as_posix()


def load_case(algorithm: str, case_dir: Path, output: Path) -> dict[str, object]:
    timeline_dir = case_dir / "timeline"
    timeline_html = timeline_dir / "dag_gpu_timeline.html"
    timeline_summary = timeline_dir / "dag_timeline_summary.json"
    gate_dir = case_dir / "gate_load"
    gate_html = gate_dir / "gate_load_profile.html"
    gate_csv = gate_dir / "gate_load_profile.csv"
    link_image = case_dir / "link_load" / "mprail_link_load_by_layer.png"
    link_summary = case_dir / "link_load" / "mprail_link_load_summary.csv"
    endpoint_summary = case_dir / "link_load" / "mprail_endpoint_load_summary.csv"
    for path in (
        timeline_html,
        timeline_summary,
        gate_html,
        gate_csv,
        link_image,
        link_summary,
        endpoint_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{algorithm}: missing artifact {path}")
    summary = json.loads(timeline_summary.read_text(encoding="utf-8"))
    migration_rows: list[dict[str, int]] = []
    manifest_path = case_dir / "workload" / "manifest.json"
    if algorithm == "probeep" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invocations = manifest.get("metadata", {}).get(
            "micro_batch_algorithms", []
        )
        for invocation in invocations:
            planned = invocation.get("planned_migration_intents", [])
            admitted = invocation.get("admitted_migration_intents", [])
            deferred = invocation.get("deferred_migration_intents", [])
            remote_replicas = invocation.get("remote_replicas", [])
            migration_rows.append(
                {
                    "layer": int(invocation["layer"]),
                    "micro_batch": int(invocation["micro_batch"]),
                    "invocation_index": int(
                        invocation.get(
                            "invocation_index", invocation.get("sample_id", 0)
                        )
                    ),
                    "planned_experts": len(
                        {
                            (item["expert_id"], item["destination_server"])
                            for item in planned
                        }
                    ),
                    "admitted_experts": len(remote_replicas),
                    "deferred_experts": len(
                        {
                            (item["expert_id"], item["destination_server"])
                            for item in deferred
                        }
                    ),
                    "moved_routes": sum(
                        int(item["moved_route_count"]) for item in admitted
                    ),
                    "weight_bytes": int(
                        invocation.get("remote_weight_rdma_bytes", 0)
                    ),
                }
            )
    gate_summary_path = gate_dir / "gate_load_profile_summary.json"
    if (
        algorithm == "probeep"
        and not migration_rows
        and gate_summary_path.is_file()
    ):
        gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
        for index, record in enumerate(gate_summary.get("records", [])):
            planning = record.get("planning", {})
            transport = record.get("transport", {})
            server_pairs = transport.get("server_pairs", [])
            migration_rows.append(
                {
                    "layer": int(record["layer"]),
                    "micro_batch": int(record["micro_batch"]),
                    "invocation_index": index,
                    "planned_experts": int(
                        planning.get("planned_intent_count", 0)
                    ),
                    "admitted_experts": int(
                        transport.get("remote_replica_count", 0)
                    ),
                    "deferred_experts": int(
                        planning.get("deferred_intent_count", 0)
                    ),
                    "moved_routes": sum(
                        int(pair.get("moved_expert_routes", 0))
                        for pair in server_pairs
                    ),
                    "weight_bytes": int(
                        transport.get("expert_weight_bytes", 0)
                    ),
                }
            )
    bundle_files: list[tuple[Path, str]] = []
    for directory in (timeline_dir, gate_dir, case_dir / "link_load"):
        for artifact in sorted(directory.iterdir()):
            if not artifact.is_file():
                continue
            archive_name = relative_url(artifact, output)
            archive_path = PurePosixPath(archive_name)
            if archive_path.is_absolute() or ".." in archive_path.parts:
                raise ValueError(
                    f"{algorithm}: visualization artifact is outside dashboard root: "
                    f"{artifact}"
                )
            bundle_files.append((artifact, archive_name))
    return {
        "algorithm": algorithm,
        "case_dir": case_dir,
        "timeline_path": timeline_html,
        "gate_path": gate_html,
        "gate_csv_path": gate_csv,
        "link_image_path": link_image,
        "link_summary_path": link_summary,
        "endpoint_summary_path": endpoint_summary,
        "timeline_url": relative_url(timeline_html, output),
        "gate_url": relative_url(gate_html, output),
        "gate_csv_url": relative_url(gate_csv, output),
        "link_image_url": relative_url(link_image, output),
        "link_summary_url": relative_url(link_summary, output),
        "endpoint_summary_url": relative_url(endpoint_summary, output),
        "makespan_us": float(summary["makespan_us"]),
        "task_count": int(summary["task_count"]),
        "network_task_count": int(summary["network_task_count"]),
        "logical_transfer_bytes": int(summary["logical_transfer_bytes"]),
        "overlap_us": float(
            summary["selected_rank_compute_network_overlap_sum_us"]
        ),
        "migration_rows": migration_rows,
        "admitted_experts_total": sum(
            row["admitted_experts"] for row in migration_rows
        ),
        "expert_weight_bytes_total": sum(
            row["weight_bytes"] for row in migration_rows
        ),
        "bundle_files": bundle_files,
    }


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.4g} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def write_algorithm_dashboard(
    path: Path,
    comparison_path: Path,
    title: str,
    case: dict[str, object],
) -> None:
    algorithm = str(case["algorithm"])
    context_title = re.sub(
        r"\s*/\s*\d+\s+algorithms\s*$", "", title, flags=re.IGNORECASE
    )
    page_title = f"{algorithm.upper()} / {context_title}"
    gate_title = {
        "nccl": "NCCL Gate / direct expert execution",
        "deepep": "DeepEP Gate / hierarchical token transport",
        "eplb": "EPLB expert placement",
        "moonep": "MoonEP local expert replication",
        "probeep": "ProbeEP admitted expert placement",
    }.get(algorithm, f"{algorithm.upper()} Gate / expert load")
    gate_url = relative_url(Path(case["gate_path"]), path)
    gate_csv_url = relative_url(Path(case["gate_csv_path"]), path)
    timeline_url = relative_url(Path(case["timeline_path"]), path)
    link_image_url = relative_url(Path(case["link_image_path"]), path)
    link_summary_url = relative_url(Path(case["link_summary_path"]), path)
    endpoint_summary_url = relative_url(
        Path(case["endpoint_summary_path"]), path
    )
    comparison_url = relative_url(comparison_path, path)
    migration_rows = list(case["migration_rows"])
    migration_metric = ""
    migration_module = ""
    if migration_rows:
        migration_metric = (
            f"<span>{case['admitted_experts_total']} cross-server expert moves</span>"
            f"<span>{escape(format_bytes(int(case['expert_weight_bytes_total'])))} RDMA weights</span>"
        )
        table_rows = "".join(
            "<tr>"
            f"<td>{row['invocation_index']}</td>"
            f"<td>{row['layer']}</td>"
            f"<td>{row['micro_batch']}</td>"
            f"<td>{row['planned_experts']}</td>"
            f"<td>{row['admitted_experts']}</td>"
            f"<td>{row['deferred_experts']}</td>"
            f"<td>{row['moved_routes']}</td>"
            f"<td>{escape(format_bytes(row['weight_bytes']))}</td>"
            "</tr>"
            for row in migration_rows
        )
        migration_module = f"""<section class="migration"><h2>Cross-server expert migration</h2><div class="migration-content"><table><thead><tr><th>Invocation</th><th>Layer</th><th>MB</th><th>Planned experts</th><th>Admitted experts</th><th>Deferred experts</th><th>Moved routes</th><th>RDMA weights</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>"""
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(page_title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #F4F6F8; color: #17212B; }}
header {{ padding: 16px 20px; background: #FFFFFF; border-bottom: 1px solid #CBD3DC; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
h1 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
.metrics {{ display: flex; gap: 16px; margin-left: auto; color: #536170; font-size: 12px; flex-wrap: wrap; }}
.back {{ width: 100%; color: #1D5E9E; font-size: 13px; }}
main {{ padding: 14px; display: grid; gap: 12px; }}
.modules {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.module, .migration {{ background: #FFFFFF; border: 1px solid #C8D0DA; }}
.module {{ min-height: 126px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between; gap: 20px; }}
h2 {{ margin: 0; font-size: 15px; letter-spacing: 0; }}
.module-links {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; }}
.module-links a {{ color: #1D5E9E; }}
.primary {{ font-weight: 650; }}
.migration h2 {{ padding: 13px 14px; border-bottom: 1px solid #D8DEE6; }}
.migration-content {{ overflow: auto; padding: 0 12px 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 7px 9px; border-bottom: 1px solid #E1E6EB; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #EEF2F5; color: #485563; }}
@media (max-width: 760px) {{ .metrics {{ width: 100%; margin-left: 0; }} .modules {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
<h1>{escape(page_title)}</h1>
<div class="metrics"><span>{case['makespan_us']:.7g} us</span><span>{case['task_count']} tasks</span><span>{escape(format_bytes(int(case['logical_transfer_bytes'])))}</span><span>{case['overlap_us']:.7g} us GPU overlap sum</span>{migration_metric}</div>
<a class="back" href="{escape(comparison_url)}">&larr; All algorithms</a>
</header>
<main>
<div class="modules">
<section class="module"><h2>{escape(gate_title)}</h2><div class="module-links"><a class="primary" href="{escape(gate_url)}">Open Gate view &rarr;</a><a href="{escape(gate_csv_url)}">Expert load CSV</a></div></section>
<section class="module"><h2>{escape(algorithm.upper())} DAG GPU timeline</h2><div class="module-links"><a class="primary" href="{escape(timeline_url)}">Open timeline &rarr;</a></div></section>
<section class="module"><h2>{escape(algorithm.upper())} MpRail link load</h2><div class="module-links"><a class="primary" href="{escape(link_image_url)}">Open link plot &rarr;</a><a href="{escape(link_summary_url)}">Link summary CSV</a><a href="{escape(endpoint_summary_url)}">Endpoint summary CSV</a></div></section>
</div>
{migration_module}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_dashboard(path: Path, title: str, cases: list[dict[str, object]]) -> None:
    sections: list[str] = []
    for case in cases:
        algorithm = str(case["algorithm"])
        migration_summary = ""
        if case["migration_rows"]:
            migration_summary = (
                f"<span>{case['admitted_experts_total']} migrated experts</span>"
                f"<span>{escape(format_bytes(int(case['expert_weight_bytes_total'])))} RDMA weights</span>"
            )
        sections.append(
            f"""<a class="algorithm" href="{escape(str(case['algorithm_dashboard_url']))}">
<div><strong>{escape(algorithm.upper())}</strong><div class="algorithm-metrics"><span>{case['makespan_us']:.7g} us</span><span>{case['task_count']} tasks</span><span>{escape(format_bytes(int(case['logical_transfer_bytes'])))}</span><span>{case['overlap_us']:.7g} us GPU overlap sum</span>{migration_summary}</div></div><span class="arrow">&rarr;</span>
</a>"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #F4F6F8; color: #17212B; }}
header {{ padding: 18px 22px; background: #FFFFFF; border-bottom: 1px solid #CED5DE; }}
h1 {{ margin: 0; font-size: 21px; }}
main {{ padding: 14px; display: grid; gap: 12px; }}
.algorithm {{ min-height: 74px; padding: 14px 16px; background: #FFFFFF; border: 1px solid #C8D0DA; color: #17212B; text-decoration: none; display: flex; align-items: center; justify-content: space-between; gap: 20px; }}
.algorithm:hover {{ border-color: #7C8C9D; background: #FAFBFC; }}
.algorithm strong {{ display: block; margin-bottom: 8px; font-size: 15px; }}
.algorithm-metrics {{ display: flex; flex-wrap: wrap; gap: 8px 18px; color: #536170; font-size: 12px; }}
.arrow {{ color: #1D5E9E; font-size: 20px; }}
@media (max-width: 760px) {{ .algorithm {{ align-items: flex-start; }} }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1></header>
<main>{''.join(sections)}</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_visualization_zip(
    path: Path,
    dashboard: Path,
    cases: list[dict[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    members: dict[str, Path] = {}
    for case in cases:
        for artifact, archive_name in case["bundle_files"]:
            previous = members.get(archive_name)
            if previous is not None and previous != artifact:
                raise ValueError(f"duplicate ZIP member: {archive_name}")
            members[archive_name] = artifact
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(dashboard, dashboard.name)
        for archive_name, artifact in sorted(members.items()):
            archive.write(artifact, archive_name)
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"corrupt ZIP member: {broken}")
    return len(members) + 1


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    algorithms: set[str] = set()
    cases: list[dict[str, object]] = []
    for algorithm, case_dir in args.case:
        if algorithm in algorithms:
            raise SystemExit(f"duplicate algorithm: {algorithm}")
        algorithms.add(algorithm)
        cases.append(load_case(algorithm, case_dir.resolve(), output))
    for case in cases:
        dashboard_path = Path(case["case_dir"]) / "algorithm_dashboard.html"
        write_algorithm_dashboard(
            dashboard_path,
            output,
            args.title,
            case,
        )
        dashboard_url = relative_url(dashboard_path, output)
        case["algorithm_dashboard_url"] = dashboard_url
        case["bundle_files"].append((dashboard_path, dashboard_url))
    write_dashboard(output, args.title, cases)
    print(f"wrote {output}")
    if args.zip_output is not None:
        zip_output = args.zip_output.resolve()
        member_count = write_visualization_zip(zip_output, output, cases)
        print(f"wrote {zip_output} ({member_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
