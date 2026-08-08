#!/usr/bin/env python3
"""Build a DSV3 comparison dashboard and optional visualization-only ZIP."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path, PurePosixPath
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
    link_image = case_dir / "link_load" / "mprail_link_load_by_layer.png"
    link_summary = case_dir / "link_load" / "mprail_link_load_summary.csv"
    endpoint_summary = case_dir / "link_load" / "mprail_endpoint_load_summary.csv"
    for path in (
        timeline_html,
        timeline_summary,
        link_image,
        link_summary,
        endpoint_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{algorithm}: missing artifact {path}")
    summary = json.loads(timeline_summary.read_text(encoding="utf-8"))
    bundle_files: list[tuple[Path, str]] = []
    for directory in (timeline_dir, case_dir / "link_load"):
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
        "timeline_url": relative_url(timeline_html, output),
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
        "bundle_files": bundle_files,
    }


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.4g} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def write_dashboard(path: Path, title: str, cases: list[dict[str, object]]) -> None:
    sections: list[str] = []
    for index, case in enumerate(cases):
        algorithm = str(case["algorithm"])
        open_attribute = " open" if index == 0 else ""
        sections.append(
            f"""<details class="algorithm"{open_attribute}>
<summary><strong>{escape(algorithm.upper())}</strong><span>{case['makespan_us']:.7g} us</span><span>{case['task_count']} tasks</span><span>{escape(format_bytes(int(case['logical_transfer_bytes'])))}</span><span>{case['overlap_us']:.7g} us GPU overlap sum</span></summary>
<div class="algorithm-body">
<iframe src="{escape(str(case['timeline_url']))}" title="{escape(algorithm)} timeline" loading="lazy"></iframe>
<details class="link-load" open><summary>MpRail link load</summary><div class="link-content"><img src="{escape(str(case['link_image_url']))}" alt="{escape(algorithm)} MpRail link load"><div class="downloads"><a href="{escape(str(case['link_summary_url']))}">Link summary CSV</a><a href="{escape(str(case['endpoint_summary_url']))}">Endpoint summary CSV</a></div></div></details>
</div>
</details>"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
body {{ margin: 0; background: #F4F6F8; color: #17212B; }}
header {{ padding: 18px 22px; background: #FFFFFF; border-bottom: 1px solid #CED5DE; display: flex; align-items: center; gap: 14px; }}
h1 {{ margin: 0; font-size: 21px; }}
.actions {{ margin-left: auto; display: flex; gap: 7px; }}
button {{ height: 31px; padding: 0 11px; border: 1px solid #B7C0CB; background: #FFFFFF; cursor: pointer; }}
main {{ padding: 14px; display: grid; gap: 12px; }}
.algorithm {{ background: #FFFFFF; border: 1px solid #C8D0DA; }}
.algorithm > summary {{ min-height: 46px; box-sizing: border-box; padding: 12px 14px; cursor: pointer; display: flex; align-items: center; gap: 20px; list-style: none; }}
.algorithm > summary::-webkit-details-marker {{ display: none; }}
.algorithm > summary::before {{ content: ''; width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 7px solid #485563; transition: transform 120ms ease; }}
.algorithm[open] > summary::before {{ transform: rotate(90deg); }}
.algorithm > summary strong {{ min-width: 90px; font-size: 15px; }}
.algorithm > summary span {{ color: #536170; font-size: 12px; }}
.algorithm-body {{ border-top: 1px solid #D8DEE6; }}
iframe {{ display: block; width: 100%; height: 930px; border: 0; background: #FFFFFF; }}
.link-load {{ margin: 12px; border: 1px solid #CCD4DD; }}
.link-load > summary {{ padding: 10px 12px; cursor: pointer; font-weight: 650; }}
.link-content {{ border-top: 1px solid #D8DEE6; padding: 10px; }}
.link-content img {{ display: block; width: 100%; height: auto; }}
.downloads {{ display: flex; gap: 16px; padding: 10px 2px 2px; font-size: 12px; }}
.downloads a {{ color: #1D5E9E; }}
@media (max-width: 760px) {{ .algorithm > summary {{ flex-wrap: wrap; gap: 8px 14px; }} iframe {{ height: 780px; }} }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1><div class="actions"><button type="button" id="expand">Expand all</button><button type="button" id="collapse">Collapse all</button></div></header>
<main>{''.join(sections)}</main>
<script>
document.getElementById('expand').addEventListener('click', () => document.querySelectorAll('.algorithm').forEach(item => item.open = true));
document.getElementById('collapse').addEventListener('click', () => document.querySelectorAll('.algorithm').forEach(item => item.open = false));
</script>
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
    write_dashboard(output, args.title, cases)
    print(f"wrote {output}")
    if args.zip_output is not None:
        zip_output = args.zip_output.resolve()
        member_count = write_visualization_zip(zip_output, output, cases)
        print(f"wrote {zip_output} ({member_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
