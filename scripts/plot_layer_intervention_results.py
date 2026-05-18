#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot behavior-vs-InfoRidge results from layer intervention summary.csv.")
    parser.add_argument("summary_csv", help="summary.csv from scripts/summarize_layer_interventions.py")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def f(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="#222"{transform}>{value}</text>'


def color(row: dict[str, str]) -> str:
    if row["mode"] == "bypass":
        return "#4c78a8"
    if row["mode"] == "shuffle_batch":
        return "#e45756"
    if row["mode"] == "noise":
        return "#54a24b"
    return "#999999"


def marker(row: dict[str, str], x: float, y: float) -> str:
    c = color(row)
    if row["level"] == "H":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="#222" stroke-width="0.8"/>'
    if row["level"] == "L":
        points = f"{x:.1f},{y - 7:.1f} {x - 7:.1f},{y + 6:.1f} {x + 7:.1f},{y + 6:.1f}"
        return f'<polygon points="{points}" fill="{c}" stroke="#222" stroke-width="0.8"/>'
    return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{c}"/>'


def plot_scatter(rows: list[dict[str, str]], path: Path) -> Path:
    data = [
        row
        for row in rows
        if row["mode"] != "baseline"
        and not math.isnan(f(row["mean_layer_delta_I_Z_Y"]))
        and not math.isnan(f(row["exact_accuracy_delta"]))
    ]
    width, height = 980, 660
    left, top = 90, 70
    plot_w, plot_h = 760, 430
    xs = [f(row["mean_layer_delta_I_Z_Y"]) for row in data]
    ys = [f(row["exact_accuracy_delta"]) for row in data]
    min_x, max_x = min(xs + [0.0]), max(xs + [0.0])
    min_y, max_y = min(ys + [0.0]), max(ys + [0.0])
    pad_x = (max_x - min_x) * 0.08 or 0.01
    pad_y = (max_y - min_y) * 0.08 or 0.01
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def x_for(v: float) -> float:
        return left + (v - min_x) / (max_x - min_x) * plot_w

    def y_for(v: float) -> float:
        return top + (max_y - v) / (max_y - min_y) * plot_h

    zero_x = x_for(0.0)
    zero_y = y_for(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 28, "InfoRidge Collapse vs Behavioral Drop", 18, "middle"),
        text(width / 2, 50, "Each point is one layer intervention. More negative means worse than baseline.", 12, "middle"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [min_x, 0.0, max_x]:
        x = x_for(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        parts.append(text(x, top + plot_h + 24, f"{tick:.2f}", 11, "middle"))
    for tick in [min_y, 0.0, max_y]:
        y = y_for(tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(text(left - 10, y + 4, f"{tick:.2f}", 11, "end"))
    for row in data:
        x = x_for(f(row["mean_layer_delta_I_Z_Y"]))
        y = y_for(f(row["exact_accuracy_delta"]))
        parts.append(marker(row, x, y))
        if row["mode"] in {"shuffle_batch", "noise"}:
            parts.append(text(x + 8, y - 8, f'{row["level"]}{row["layer"]}', 10))
    parts.append(text(left + plot_w / 2, height - 88, "Mean layerwise delta I_Z_Y vs baseline", 12, "middle"))
    parts.append(text(22, top + plot_h / 2, "Exact accuracy delta vs baseline", 12, "middle", -90))

    legend = [("bypass", "#4c78a8"), ("shuffle_batch", "#e45756"), ("noise", "#54a24b")]
    for idx, (label, c) in enumerate(legend):
        x = left + idx * 160
        y = height - 48
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{c}" stroke="#222"/>')
        parts.append(text(x + 12, y + 4, label, 12))
    parts.append(text(left + 520, height - 44, "circle=H layer, triangle=L layer", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def plot_accuracy_bars(rows: list[dict[str, str]], path: Path) -> Path:
    data = [row for row in rows if row["mode"] != "baseline" and not math.isnan(f(row["exact_accuracy_delta"]))]
    data.sort(key=lambda row: f(row["exact_accuracy_delta"]))
    width = 980
    row_h = 28
    left, top = 250, 68
    plot_w = 620
    height = top + row_h * len(data) + 76
    values = [f(row["exact_accuracy_delta"]) for row in data]
    min_v, max_v = min(values + [0.0]), max(values + [0.0])
    span = max_v - min_v or 1.0

    def x_for(v: float) -> float:
        return left + (v - min_v) / span * plot_w

    zero_x = x_for(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 28, "Behavioral Impact by Intervention", 18, "middle"),
        text(width / 2, 48, "Exact accuracy delta relative to baseline.", 12, "middle"),
        f'<line x1="{zero_x:.1f}" y1="{top - 8}" x2="{zero_x:.1f}" y2="{height - 54}" stroke="#333"/>',
    ]
    for idx, row in enumerate(data):
        value = f(row["exact_accuracy_delta"])
        y = top + idx * row_h
        x = x_for(value)
        bar_x = min(x, zero_x)
        bar_w = abs(zero_x - x)
        label = f'{row["level"]}{row["layer"]}-{row["mode"]}'
        parts.append(text(left - 12, y + 18, label, 12, "end"))
        parts.append(f'<rect x="{bar_x:.1f}" y="{y + 5}" width="{bar_w:.1f}" height="18" fill="{color(row)}"/>')
        parts.append(text(max(x, zero_x) + 6, y + 19, f"{value:.3f}", 11))
    parts.append(text(left + plot_w / 2, height - 18, "Exact accuracy delta", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(summary_csv)
    outputs = [
        plot_scatter(rows, output_dir / "inforidge_vs_exact_accuracy.svg"),
        plot_accuracy_bars(rows, output_dir / "exact_accuracy_drop_by_intervention.svg"),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
