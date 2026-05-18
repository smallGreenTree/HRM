#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import re
import statistics
import zipfile
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


LAYERWISE_SUFFIX = "__inforidge_layerwise.csv"
ACT_SUFFIX = "__inforidge_act_mi.csv"
LAYER_ORDER = [("H", str(i)) for i in range(4)] + [("L", str(i)) for i in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot InfoRidge summaries for HRM layer intervention sweeps.")
    parser.add_argument("input", help="Directory or ZIP containing *_inforidge_*.csv files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated CSVs and PNGs.")
    parser.add_argument(
        "--selected-cases",
        nargs="*",
        default=["baseline", "H0-shuffle_batch", "H3-shuffle_batch", "L0-shuffle_batch", "L3-shuffle_batch"],
        help="Cases to include in ACT-step comparison plots.",
    )
    return parser.parse_args()


class CsvBundle:
    def __init__(self, source: Path):
        self.source = source
        self._zip: zipfile.ZipFile | None = None
        if source.is_file() and source.suffix == ".zip":
            self._zip = zipfile.ZipFile(source)
            self.names = sorted(n for n in self._zip.namelist() if n.endswith(".csv"))
        elif source.is_dir():
            self.names = sorted(p.name for p in source.glob("*.csv"))
        else:
            raise FileNotFoundError(f"Expected a ZIP or directory: {source}")

    def read(self, name: str) -> list[dict[str, str]]:
        if self._zip is not None:
            with self._zip.open(name) as f:
                return list(csv.DictReader(io.TextIOWrapper(f)))
        with (self.source / name).open() as f:
            return list(csv.DictReader(f))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def num(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def case_sort_key(case: str) -> tuple[int, str, int, str]:
    if case == "baseline":
        return (0, "", -1, "")
    match = re.match(r"([HL])(\d+)-(bypass|shuffle_batch)$", case)
    if not match:
        return (9, case, 0, "")
    module, layer, intervention = match.groups()
    return (1 if module == "H" else 2, intervention, int(layer), module)


def load_layerwise(bundle: CsvBundle) -> dict[str, dict[tuple[str, str], dict[str, float | None]]]:
    data: dict[str, dict[tuple[str, str], dict[str, float | None]]] = {}
    for name in bundle.names:
        if not name.endswith(LAYERWISE_SUFFIX):
            continue
        case = name[: -len(LAYERWISE_SUFFIX)]
        rows = bundle.read(name)
        data[case] = {
            (row["layer_type"], row["layer_index"]): {
                "I_Z_Y": num(row.get("I_Z_Y")),
                "I_dZ_Y": num(row.get("I_dZ_Y")),
            }
            for row in rows
        }
    return data


def load_act(bundle: CsvBundle) -> dict[str, list[dict[str, float | int | None]]]:
    data: dict[str, list[dict[str, float | int | None]]] = {}
    for name in bundle.names:
        if not name.endswith(ACT_SUFFIX):
            continue
        case = name[: -len(ACT_SUFFIX)]
        rows = bundle.read(name)
        data[case] = [
            {
                "act_step": int(row["act_step"]),
                "I_Z_Y": num(row.get("I_Z_Y")),
                "I_dZ_Y": num(row.get("I_dZ_Y")),
            }
            for row in rows
        ]
    return data


def write_inforidge_summary(
    output_dir: Path,
    layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]],
    act: dict[str, list[dict[str, float | int | None]]],
) -> Path:
    baseline_layer = layerwise["baseline"]
    baseline_act = {row["act_step"]: row for row in act["baseline"]}
    rows: list[dict[str, str | float]] = []

    for case in sorted(layerwise, key=case_sort_key):
        layer_deltas_izy: list[float] = []
        layer_deltas_idzy: list[float] = []
        for key, values in layerwise[case].items():
            base_values = baseline_layer[key]
            for metric, target in [("I_Z_Y", layer_deltas_izy), ("I_dZ_Y", layer_deltas_idzy)]:
                value = values[metric]
                base_value = base_values[metric]
                if value is not None and base_value is not None:
                    target.append(value - base_value)

        act_deltas_izy: list[float] = []
        act_deltas_idzy: list[float] = []
        for row in act.get(case, []):
            base_row = baseline_act.get(row["act_step"])
            if base_row is None:
                continue
            for metric, target in [("I_Z_Y", act_deltas_izy), ("I_dZ_Y", act_deltas_idzy)]:
                value = row[metric]
                base_value = base_row[metric]
                if isinstance(value, float) and isinstance(base_value, float):
                    target.append(value - base_value)

        match = re.match(r"([HL])(\d+)-(bypass|shuffle_batch)$", case)
        rows.append(
            {
                "case": case,
                "target_module": match.group(1) if match else ("baseline" if case == "baseline" else ""),
                "target_layer": match.group(2) if match else "",
                "intervention": match.group(3) if match else ("baseline" if case == "baseline" else ""),
                "mean_layer_delta_I_Z_Y": statistics.mean(layer_deltas_izy) if layer_deltas_izy else 0.0,
                "min_layer_delta_I_Z_Y": min(layer_deltas_izy) if layer_deltas_izy else 0.0,
                "max_layer_delta_I_Z_Y": max(layer_deltas_izy) if layer_deltas_izy else 0.0,
                "mean_layer_delta_I_dZ_Y": statistics.mean(layer_deltas_idzy) if layer_deltas_idzy else 0.0,
                "min_layer_delta_I_dZ_Y": min(layer_deltas_idzy) if layer_deltas_idzy else 0.0,
                "max_layer_delta_I_dZ_Y": max(layer_deltas_idzy) if layer_deltas_idzy else 0.0,
                "mean_act_delta_I_Z_Y": statistics.mean(act_deltas_izy) if act_deltas_izy else 0.0,
                "min_act_delta_I_Z_Y": min(act_deltas_izy) if act_deltas_izy else 0.0,
                "max_act_delta_I_Z_Y": max(act_deltas_izy) if act_deltas_izy else 0.0,
                "mean_act_delta_I_dZ_Y": statistics.mean(act_deltas_idzy) if act_deltas_idzy else 0.0,
                "min_act_delta_I_dZ_Y": min(act_deltas_idzy) if act_deltas_idzy else 0.0,
                "max_act_delta_I_dZ_Y": max(act_deltas_idzy) if act_deltas_idzy else 0.0,
            }
        )

    path = output_dir / "inforidge_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_heatmap(output_dir: Path, layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]]) -> Path:
    baseline = layerwise["baseline"]
    cases = [case for case in sorted(layerwise, key=case_sort_key) if case != "baseline"]
    matrix = []
    for case in cases:
        row = []
        for key in LAYER_ORDER:
            value = layerwise[case][key]["I_Z_Y"]
            base_value = baseline[key]["I_Z_Y"]
            row.append((value or 0.0) - (base_value or 0.0))
        matrix.append(row)

    if plt is None:
        return write_heatmap_svg(output_dir / "layer_intervention_izy_heatmap.svg", cases, matrix)

    fig, ax = plt.subplots(figsize=(11, 8))
    max_abs = max(abs(value) for row in matrix for value in row)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_title("Layer Intervention Effect on InfoRidge I_Z_Y")
    ax.set_xticks(range(len(LAYER_ORDER)))
    ax.set_xticklabels([f"{kind}{idx}" for kind, idx in LAYER_ORDER])
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases)
    ax.set_xlabel("Measured Layer")
    ax.set_ylabel("Intervention")
    fig.colorbar(image, ax=ax, label="Delta I_Z_Y vs baseline")
    fig.tight_layout()
    path = output_dir / "layer_intervention_izy_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sensitivity_bars(output_dir: Path, summary_path: Path) -> Path:
    with summary_path.open() as f:
        rows = [row for row in csv.DictReader(f) if row["case"] != "baseline"]
    rows.sort(key=lambda row: float(row["mean_layer_delta_I_Z_Y"]))

    if plt is None:
        return write_bar_svg(output_dir / "intervention_sensitivity_bar.svg", rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    cases = [row["case"] for row in rows]
    values = [float(row["mean_layer_delta_I_Z_Y"]) for row in rows]
    colors = ["#4c78a8" if case.startswith("H") else "#f58518" for case in cases]
    ax.bar(range(len(cases)), values, color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("Mean Layerwise InfoRidge Drop by Intervention")
    ax.set_ylabel("Mean delta I_Z_Y vs baseline")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "intervention_sensitivity_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_act_curves(output_dir: Path, act: dict[str, list[dict[str, float | int | None]]], selected_cases: list[str]) -> Path:
    if plt is None:
        return write_act_svg(output_dir / "act_step_inforidge_curves.svg", act, selected_cases)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for case in selected_cases:
        if case not in act:
            continue
        rows = sorted(act[case], key=lambda row: int(row["act_step"]))
        steps = [int(row["act_step"]) for row in rows]
        axes[0].plot(steps, [row["I_Z_Y"] for row in rows], marker="o", linewidth=1.4, label=case)
        axes[1].plot(steps, [row["I_dZ_Y"] for row in rows], marker="o", linewidth=1.4, label=case)

    axes[0].set_title("ACT-Step I_Z_Y")
    axes[1].set_title("ACT-Step I_dZ_Y")
    for ax in axes:
        ax.set_xlabel("ACT step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("InfoRidge")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = output_dir / "act_step_inforidge_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_baseline_profile(output_dir: Path, layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]]) -> Path:
    baseline = layerwise["baseline"]
    labels = [f"{kind}{idx}" for kind, idx in LAYER_ORDER]
    iz = [baseline[key]["I_Z_Y"] for key in LAYER_ORDER]
    idz = [baseline[key]["I_dZ_Y"] for key in LAYER_ORDER]

    if plt is None:
        return write_profile_svg(output_dir / "baseline_layerwise_profile.svg", labels, iz, idz)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(labels)))
    ax.plot(x, iz, marker="o", label="I_Z_Y")
    ax.plot(x, idz, marker="o", label="I_dZ_Y")
    ax.set_title("Baseline Layerwise InfoRidge Profile")
    ax.set_ylabel("InfoRidge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "baseline_layerwise_profile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def layer_delta(
    layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]],
    case: str,
    metric: str,
    module: str | None = None,
) -> list[float]:
    baseline = layerwise["baseline"]
    values: list[float] = []
    for key in LAYER_ORDER:
        if module is not None and key[0] != module:
            continue
        value = layerwise[case][key][metric]
        base_value = baseline[key][metric]
        if value is not None and base_value is not None:
            values.append(value - base_value)
    return values


def case_parts(case: str) -> tuple[str, str, str] | None:
    match = re.match(r"([HL])(\d+)-(bypass|shuffle_batch)$", case)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def write_ranked_bar_svg(
    path: Path,
    layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]],
) -> Path:
    rows = []
    for case in sorted((c for c in layerwise if c != "baseline"), key=case_sort_key):
        rows.append((case, mean(layer_delta(layerwise, case, "I_Z_Y"))))
    rows.sort(key=lambda item: item[1])

    width = 980
    row_h = 28
    left = 230
    top = 64
    plot_w = 650
    height = top + row_h * len(rows) + 70
    min_v = min(v for _, v in rows)
    max_v = max(0.0, max(v for _, v in rows))
    span = max_v - min_v or 1.0

    def x_for(v: float) -> float:
        return left + (v - min_v) / span * plot_w

    zero_x = x_for(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Which intervention collapses information most?", 18, "middle"),
        svg_text(width / 2, 48, "Mean layerwise delta I_Z_Y vs baseline. More negative means stronger collapse.", 12, "middle"),
        f'<line x1="{zero_x:.1f}" y1="{top - 8}" x2="{zero_x:.1f}" y2="{height - 52}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [-0.55, -0.4, -0.2, 0.0]:
        if min_v <= tick <= max_v:
            x = x_for(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{height - 50}" x2="{x:.1f}" y2="{height - 45}" stroke="#333"/>')
            parts.append(svg_text(x, height - 28, f"{tick:.2f}", 11, "middle"))
    for i, (case, value) in enumerate(rows):
        y = top + i * row_h
        color = "#4c78a8" if case.startswith("H") else "#f58518"
        x = x_for(value)
        bar_x = min(x, zero_x)
        bar_w = abs(zero_x - x)
        parts.append(svg_text(left - 12, y + 18, case, 12, "end"))
        parts.append(f'<rect x="{bar_x:.1f}" y="{y + 5:.1f}" width="{bar_w:.1f}" height="18" fill="{color}"/>')
        parts.append(svg_text(max(x, zero_x) + 6, y + 19, f"{value:.3f}", 11))
    parts.append(svg_text(left + plot_w / 2, height - 8, "Mean delta I_Z_Y", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def write_grouped_module_svg(
    path: Path,
    layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]],
) -> Path:
    groups = {
        "H bypass": [],
        "H shuffle": [],
        "L bypass": [],
        "L shuffle": [],
    }
    for case in layerwise:
        parts = case_parts(case)
        if parts is None:
            continue
        module, _layer, intervention = parts
        key = f"{module} {'shuffle' if intervention == 'shuffle_batch' else 'bypass'}"
        groups[key].append(mean(layer_delta(layerwise, case, "I_Z_Y")))
    rows = [(key, mean(values)) for key, values in groups.items()]

    width, height = 760, 430
    left, top = 100, 70
    plot_w, plot_h = 560, 250
    min_v = min(v for _, v in rows)
    max_v = 0.0
    span = max_v - min_v or 1.0

    def y_for(v: float) -> float:
        return top + (max_v - v) / span * plot_h

    zero_y = y_for(0.0)
    step = plot_w / len(rows)
    bar_w = step * 0.55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "H vs L Sensitivity", 18, "middle"),
        svg_text(width / 2, 48, "Average information drop over all measured layers.", 12, "middle"),
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#333"/>',
    ]
    for tick in [-0.55, -0.4, -0.2, 0.0]:
        if min_v <= tick <= max_v:
            y = y_for(tick)
            parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
            parts.append(svg_text(left - 10, y + 4, f"{tick:.2f}", 11, "end"))
    for idx, (label, value) in enumerate(rows):
        x = left + idx * step + (step - bar_w) / 2
        y = y_for(value)
        color = "#4c78a8" if label.startswith("H") else "#f58518"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{zero_y - y:.1f}" fill="{color}"/>')
        parts.append(svg_text(x + bar_w / 2, zero_y + 24, label, 12, "middle"))
        parts.append(svg_text(x + bar_w / 2, y - 8, f"{value:.3f}", 12, "middle"))
    parts.append(svg_text(28, top + plot_h / 2, "Mean delta I_Z_Y", 12, "middle", -90))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def write_module_collapse_heatmap_svg(
    path: Path,
    layerwise: dict[str, dict[tuple[str, str], dict[str, float | None]]],
) -> Path:
    cases = [case for case in sorted(layerwise, key=case_sort_key) if case != "baseline"]
    rows = [(case, mean(layer_delta(layerwise, case, "I_Z_Y", "H")), mean(layer_delta(layerwise, case, "I_Z_Y", "L"))) for case in cases]
    cell_w, cell_h = 130, 28
    left, top = 220, 72
    width = left + cell_w * 2 + 60
    height = top + cell_h * len(rows) + 70
    max_abs = max(abs(v) for _, h, l in rows for v in (h, l))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Where does information collapse?", 18, "middle"),
        svg_text(width / 2, 48, "Mean delta I_Z_Y in measured H layers vs measured L layers.", 12, "middle"),
        svg_text(left + cell_w / 2, top - 12, "Measured H avg", 12, "middle"),
        svg_text(left + cell_w * 1.5, top - 12, "Measured L avg", 12, "middle"),
    ]
    for i, (case, h_value, l_value) in enumerate(rows):
        y = top + i * cell_h
        parts.append(svg_text(left - 12, y + 18, case, 12, "end"))
        for j, value in enumerate([h_value, l_value]):
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color_for_delta(value, max_abs)}" stroke="#ffffff"/>')
            parts.append(svg_text(x + cell_w / 2, y + 18, f"{value:.3f}", 11, "middle"))
    parts.append(svg_text(left, height - 22, "Blue = information drop vs baseline.", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def write_act_delta_svg(
    path: Path,
    act: dict[str, list[dict[str, float | int | None]]],
    selected_cases: list[str],
) -> Path:
    cases = [case for case in selected_cases if case in act and case != "baseline"]
    baseline = {int(row["act_step"]): row for row in act["baseline"]}
    width, height = 1120, 560
    left, top = 80, 74
    panel_w, panel_h = 450, 310
    colors = ["#4c78a8", "#54a24b", "#f58518", "#e45756", "#b279a2"]

    def series(metric: str) -> dict[str, list[tuple[int, float]]]:
        out: dict[str, list[tuple[int, float]]] = {}
        for case in cases:
            vals = []
            for row in sorted(act[case], key=lambda r: int(r["act_step"])):
                step = int(row["act_step"])
                value = row[metric]
                base_value = baseline[step][metric]
                if isinstance(value, float) and isinstance(base_value, float):
                    vals.append((step, value - base_value))
            out[case] = vals
        return out

    all_series = {"Delta I_Z_Y": series("I_Z_Y"), "Delta I_dZ_Y": series("I_dZ_Y")}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "When does the intervention change information?", 18, "middle"),
        svg_text(width / 2, 50, "ACT-step delta against baseline. Negative values mean less information than baseline.", 12, "middle"),
    ]
    for panel_idx, (title, values_by_case) in enumerate(all_series.items()):
        x0 = left + panel_idx * (panel_w + 110)
        all_vals = [v for vals in values_by_case.values() for _, v in vals]
        min_v = min(all_vals + [0.0])
        max_v = max(all_vals + [0.0])
        pad = (max_v - min_v) * 0.08 or 0.001
        min_v -= pad
        max_v += pad
        span = max_v - min_v

        def x_for(step: int) -> float:
            return x0 + (step - 1) / 15 * panel_w

        def y_for(v: float) -> float:
            return top + (max_v - v) / span * panel_h

        zero_y = y_for(0.0)
        parts.append(f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#bbbbbb"/>')
        parts.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x0 + panel_w}" y2="{zero_y:.1f}" stroke="#333" stroke-width="1"/>')
        parts.append(svg_text(x0 + panel_w / 2, top - 14, title, 14, "middle"))
        for tick in [1, 4, 8, 12, 16]:
            x = x_for(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{top + panel_h}" x2="{x:.1f}" y2="{top + panel_h + 5}" stroke="#333"/>')
            parts.append(svg_text(x, top + panel_h + 22, str(tick), 11, "middle"))
        for tick in [min_v, 0.0, max_v]:
            y = y_for(tick)
            parts.append(f'<line x1="{x0 - 5}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#333"/>')
            parts.append(svg_text(x0 - 10, y + 4, f"{tick:.3f}", 10, "end"))
        for idx, (case, vals) in enumerate(values_by_case.items()):
            pts = [(x_for(step), y_for(value)) for step, value in vals]
            parts.append(polyline(pts, colors[idx % len(colors)]))
        parts.append(svg_text(x0 + panel_w / 2, top + panel_h + 46, "ACT step", 12, "middle"))
    legend_y = height - 92
    for idx, case in enumerate(cases):
        x = left + (idx % 3) * 330
        y = legend_y + (idx // 3) * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{colors[idx % len(colors)]}" stroke-width="3"/>')
        parts.append(svg_text(x + 36, y + 4, case, 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="#222"{transform}>{text}</text>'


def color_for_delta(value: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "#f7f7f7"
    t = max(-1.0, min(1.0, value / max_abs))
    if t < 0:
        a = abs(t)
        r = int(247 * (1 - a) + 49 * a)
        g = int(247 * (1 - a) + 130 * a)
        b = int(247 * (1 - a) + 189 * a)
    else:
        a = t
        r = int(247 * (1 - a) + 202 * a)
        g = int(247 * (1 - a) + 0 * a)
        b = int(247 * (1 - a) + 32 * a)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(path: Path, cases: list[str], matrix: list[list[float]]) -> Path:
    cell_w, cell_h = 64, 28
    left, top = 180, 70
    width = left + cell_w * len(LAYER_ORDER) + 40
    height = top + cell_h * len(cases) + 80
    max_abs = max(abs(value) for row in matrix for value in row)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Layer Intervention Effect on InfoRidge I_Z_Y", 18, "middle"),
    ]
    for j, (kind, idx) in enumerate(LAYER_ORDER):
        parts.append(svg_text(left + j * cell_w + cell_w / 2, top - 16, f"{kind}{idx}", 12, "middle"))
    for i, case in enumerate(cases):
        y = top + i * cell_h
        parts.append(svg_text(left - 10, y + 18, case, 11, "end"))
        for j, value in enumerate(matrix[i]):
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color_for_delta(value, max_abs)}" stroke="#ffffff"/>')
            parts.append(svg_text(x + cell_w / 2, y + 18, f"{value:.2f}", 9, "middle"))
    parts.append(svg_text(left, height - 24, "Cell value = delta I_Z_Y vs baseline. Blue is drop, red is increase.", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def write_bar_svg(path: Path, rows: list[dict[str, str]]) -> Path:
    width, height = 1120, 560
    left, top, bottom = 80, 60, 150
    plot_h = height - top - bottom
    plot_w = width - left - 30
    values = [float(row["mean_layer_delta_I_Z_Y"]) for row in rows]
    min_v = min(values + [0.0])
    max_v = max(values + [0.0])
    scale = max_v - min_v or 1.0

    def y_for(v: float) -> float:
        return top + (max_v - v) / scale * plot_h

    bar_w = plot_w / len(rows) * 0.78
    step = plot_w / len(rows)
    zero_y = y_for(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Mean Layerwise InfoRidge Drop by Intervention", 18, "middle"),
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - 30}" y2="{zero_y:.1f}" stroke="#222" stroke-width="1"/>',
    ]
    for i, row in enumerate(rows):
        v = float(row["mean_layer_delta_I_Z_Y"])
        x = left + i * step + (step - bar_w) / 2
        y = min(y_for(v), zero_y)
        h = abs(zero_y - y_for(v))
        color = "#4c78a8" if row["case"].startswith("H") else "#f58518"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(svg_text(x + bar_w / 2, height - 30, row["case"], 10, "end", -45))
    parts.append(svg_text(20, top + plot_h / 2, "Mean delta I_Z_Y", 12, "middle", -90))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'


def write_act_svg(path: Path, act: dict[str, list[dict[str, float | int | None]]], selected_cases: list[str]) -> Path:
    width, height = 1120, 460
    left, top = 70, 60
    panel_w, panel_h = 460, 300
    colors = ["#222222", "#4c78a8", "#54a24b", "#f58518", "#e45756", "#b279a2"]

    series = {case: sorted(act[case], key=lambda row: int(row["act_step"])) for case in selected_cases if case in act}
    all_vals = [float(row[m]) for rows in series.values() for row in rows for m in ("I_Z_Y", "I_dZ_Y") if isinstance(row[m], float)]
    min_v, max_v = min(all_vals), max(all_vals)
    scale = max_v - min_v or 1.0

    def points(rows: list[dict[str, float | int | None]], metric: str, x0: int) -> list[tuple[float, float]]:
        pts = []
        for row in rows:
            step = int(row["act_step"])
            value = float(row[metric] or 0.0)
            x = x0 + (step - 1) / 15 * panel_w
            y = top + (max_v - value) / scale * panel_h
            pts.append((x, y))
        return pts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "ACT-Step InfoRidge Curves", 18, "middle"),
    ]
    for panel_idx, metric in enumerate(["I_Z_Y", "I_dZ_Y"]):
        x0 = left + panel_idx * (panel_w + 95)
        parts.append(f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>')
        parts.append(svg_text(x0 + panel_w / 2, top - 12, metric, 14, "middle"))
        for idx, (case, rows) in enumerate(series.items()):
            parts.append(polyline(points(rows, metric, x0), colors[idx % len(colors)]))
    legend_x = left
    for idx, case in enumerate(series):
        y = top + panel_h + 42 + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{colors[idx % len(colors)]}" stroke-width="3"/>')
        parts.append(svg_text(legend_x + 32, y + 4, case, 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def write_profile_svg(path: Path, labels: list[str], iz: list[float | None], idz: list[float | None]) -> Path:
    width, height = 820, 420
    left, top = 70, 60
    plot_w, plot_h = 680, 260
    values = [float(v) for v in iz + idz if v is not None]
    min_v, max_v = min(values), max(values)
    scale = max_v - min_v or 1.0

    def points(vals: list[float | None]) -> list[tuple[float, float]]:
        pts = []
        for idx, value in enumerate(vals):
            x = left + idx / (len(vals) - 1) * plot_w
            y = top + (max_v - float(value or 0.0)) / scale * plot_h
            pts.append((x, y))
        return pts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Baseline Layerwise InfoRidge Profile", 18, "middle"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>',
        polyline(points(iz), "#4c78a8"),
        polyline(points(idz), "#f58518"),
    ]
    for idx, label in enumerate(labels):
        x = left + idx / (len(labels) - 1) * plot_w
        parts.append(svg_text(x, top + plot_h + 22, label, 11, "middle"))
    parts.append(f'<line x1="{left}" y1="{height - 48}" x2="{left + 24}" y2="{height - 48}" stroke="#4c78a8" stroke-width="3"/>')
    parts.append(svg_text(left + 32, height - 44, "I_Z_Y", 12))
    parts.append(f'<line x1="{left + 110}" y1="{height - 48}" x2="{left + 134}" y2="{height - 48}" stroke="#f58518" stroke-width="3"/>')
    parts.append(svg_text(left + 142, height - 44, "I_dZ_Y", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = CsvBundle(Path(args.input).resolve())
    try:
        layerwise = load_layerwise(bundle)
        act = load_act(bundle)
    finally:
        bundle.close()

    if "baseline" not in layerwise or "baseline" not in act:
        raise ValueError("Input must include baseline__inforidge_layerwise.csv and baseline__inforidge_act_mi.csv")

    outputs = [
        write_inforidge_summary(output_dir, layerwise, act),
        write_ranked_bar_svg(output_dir / "ranked_intervention_information_drop.svg", layerwise),
        write_grouped_module_svg(output_dir / "module_intervention_summary.svg", layerwise),
        write_module_collapse_heatmap_svg(output_dir / "h_vs_l_collapse_heatmap.svg", layerwise),
        write_act_delta_svg(output_dir / "act_step_information_delta.svg", act, args.selected_cases),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
