#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import zipfile


LOG_NAMES = {
    "summary.json",
    "配置.json",
    "结果.json",
    "测试报告.md",
    "命令.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_direct_files(
    archive: zipfile.ZipFile,
    source: Path,
    case_id: str,
    *,
    html: bool,
) -> int:
    count = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix == ".zip":
            continue
        selected = path.suffix == ".html" if html else (
            path.suffix in {".log", ".txt"} or path.name in LOG_NAMES
        )
        if selected:
            archive.write(path, Path(case_id) / source.name / path.relative_to(source))
            count += 1
    return count


def add_nested_html(
    archive: zipfile.ZipFile, source: Path, case_id: str
) -> int:
    count = 0
    for nested_path in source.glob("*.zip"):
        with zipfile.ZipFile(nested_path) as nested:
            for member in nested.infolist():
                if member.is_dir() or not member.filename.endswith(".html"):
                    continue
                target = Path(case_id) / source.name / nested_path.stem / member.filename
                archive.writestr(str(target), nested.read(member))
                count += 1
    return count


def main() -> int:
    args = parse_args()
    rows = manifest_rows(args.manifest)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    command_logs = args.artifact_dir / "command_logs"
    with zipfile.ZipFile(
        args.artifact_dir / "logs.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as logs:
        if command_logs.is_dir():
            for path in command_logs.glob("*.log"):
                logs.write(path, Path("commands") / path.name)
        for row in rows:
            source = Path(row["source_run"]).resolve()
            add_direct_files(logs, source, row["case_id"], html=False)

    html_count = 0
    html_path = args.artifact_dir / "html.zip"
    with zipfile.ZipFile(html_path, "w", compression=zipfile.ZIP_DEFLATED) as html:
        for row in rows:
            source = Path(row["source_run"]).resolve()
            html_count += add_direct_files(html, source, row["case_id"], html=True)
            html_count += add_nested_html(html, source, row["case_id"])
    if html_count == 0:
        html_path.unlink()

    if command_logs.is_dir():
        for path in command_logs.iterdir():
            path.unlink()
        command_logs.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
