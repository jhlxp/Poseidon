from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt


FIGSIZE = (5, 4)
LABEL_SIZE = 22
TICK_SIZE = 19
LEGEND_SIZE = 20
COLORS = ("#2f6b9a", "#d07a28", "#4f8a5b", "#a64b5d", "#725a9a")
HATCHES = ("", "//", "xx", "..", "\\\\")
MARKERS = ("o", "s", "^", "D", "P", "X")


def _font_family() -> str:
    names = {item.name for item in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "Times New Roman"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": _font_family(),
            "font.size": TICK_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.2,
            "lines.markersize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figure():
    configure()
    return plt.subplots(figsize=FIGSIZE)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing plot data: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty plot data: {path}")
    return rows


def ordered(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise KeyError(f"missing numeric field {key!r}")
    return float(value)


def style_axis(ax, *, grid: str = "y") -> None:
    ax.grid(axis=grid, color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.1, length=5)


def save(fig, type_dir: Path, stem: str) -> None:
    png_dir = type_dir / "png"
    pdf_dir = type_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        pdf_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def save_legend(handles, labels, type_dir: Path, stem: str, *, ncol: int) -> None:
    configure()
    fig = plt.figure(figsize=FIGSIZE)
    fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.0,
    )
    save(fig, type_dir, stem)
