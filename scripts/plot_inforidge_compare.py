#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two InfoRidge runs.")
    parser.add_argument("--small", required=True, help="Layerwise CSV for the small-batch run.")
    parser.add_argument("--full", required=True, help="Layerwise CSV for the full-test run.")
    parser.add_argument("--small-act", default=None, help="ACT-step CSV for the small-batch run.")
    parser.add_argument("--full-act", default=None, help="ACT-step CSV for the full-test run.")
    parser.add_argument("--output", required=True, help="PNG output path.")
    return parser.parse_args()


def read_layerwise_csv(path: Path) -> dict[str, tuple[list[int], list[float], list[float | None]]]:
    rows: dict[str, tuple[list[int], list[float], list[float | None]]] = {
        "H": ([], [], []),
        "L": ([], [], []),
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_type = row["layer_type"]
            x, iz, idz = rows[layer_type]
            x.append(int(row["layer_index"]))
            iz.append(float(row["I_Z_Y"]))
            idz.append(None if row["I_dZ_Y"] == "" else float(row["I_dZ_Y"]))
    return rows


def _series_without_none(xs: list[int], ys: list[float | None]) -> tuple[list[int], list[float]]:
    out_x: list[int] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        if y is not None:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def read_act_csv(path: Path) -> tuple[list[int], list[float], list[float | None]]:
    x: list[int] = []
    iz: list[float] = []
    idz: list[float | None] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(int(row["act_step"]))
            iz.append(float(row["I_Z_Y"]))
            idz.append(None if row["I_dZ_Y"] == "" else float(row["I_dZ_Y"]))
    return x, iz, idz


def main() -> None:
    args = parse_args()
    small_path = Path(args.small).resolve()
    full_path = Path(args.full).resolve()
    output_path = Path(args.output).resolve()
    small_act_path = Path(args.small_act).resolve() if args.small_act else None
    full_act_path = Path(args.full_act).resolve() if args.full_act else None

    small = read_layerwise_csv(small_path)
    full = read_layerwise_csv(full_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x_offset = 0.06

    for layer_type, color in (("H", "#1f77b4"), ("L", "#d62728")):
        xs_small, iz_small, idz_small = small[layer_type]
        xs_full, iz_full, idz_full = full[layer_type]
        xs_small_shift = [x - x_offset for x in xs_small]
        xs_full_shift = [x + x_offset for x in xs_full]

        axes[0, 0].plot(xs_small_shift, iz_small, marker="s", linestyle="--", color=color, alpha=0.85, linewidth=2, label=f"{layer_type} small")
        axes[0, 0].plot(xs_full_shift, iz_full, marker="o", linestyle="-", color=color, linewidth=2, label=f"{layer_type} full")

        x_small_d, y_small_d = _series_without_none(xs_small, idz_small)
        x_full_d, y_full_d = _series_without_none(xs_full, idz_full)
        axes[0, 1].plot([x - x_offset for x in x_small_d], y_small_d, marker="s", linestyle="--", color=color, alpha=0.85, linewidth=2, label=f"{layer_type} small")
        axes[0, 1].plot([x + x_offset for x in x_full_d], y_full_d, marker="o", linestyle="-", color=color, linewidth=2, label=f"{layer_type} full")

    axes[0, 0].set_title("Layerwise Predictive MI")
    axes[0, 0].set_xlabel("Layer Depth")
    axes[0, 0].set_ylabel("I(Z;Y)")
    axes[0, 0].set_xticks(np.arange(0, 4, 1))
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("Layerwise Incremental MI")
    axes[0, 1].set_xlabel("Layer Depth")
    axes[0, 1].set_ylabel("I(dZ;Y)")
    axes[0, 1].set_xticks(np.arange(0, 4, 1))
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    if small_act_path is not None and full_act_path is not None:
        small_act_x, small_act_iz, small_act_idz = read_act_csv(small_act_path)
        full_act_x, full_act_iz, full_act_idz = read_act_csv(full_act_path)

        axes[1, 0].plot([x - x_offset for x in small_act_x], small_act_iz, marker="s", linestyle="--", color="#2ca02c", alpha=0.85, linewidth=2, label="small")
        axes[1, 0].plot([x + x_offset for x in full_act_x], full_act_iz, marker="o", linestyle="-", color="#2ca02c", linewidth=2, label="full")
        axes[1, 0].set_title("ACT Predictive MI")
        axes[1, 0].set_xlabel("ACT Step")
        axes[1, 0].set_ylabel("I(Z;Y)")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()

        small_act_dx, small_act_dy = _series_without_none(small_act_x, small_act_idz)
        full_act_dx, full_act_dy = _series_without_none(full_act_x, full_act_idz)
        axes[1, 1].plot([x - x_offset for x in small_act_dx], small_act_dy, marker="s", linestyle="--", color="#9467bd", alpha=0.85, linewidth=2, label="small")
        axes[1, 1].plot([x + x_offset for x in full_act_dx], full_act_dy, marker="o", linestyle="-", color="#9467bd", linewidth=2, label="full")
        axes[1, 1].set_title("ACT Incremental MI")
        axes[1, 1].set_xlabel("ACT Step")
        axes[1, 1].set_ylabel("I(dZ;Y)")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        if small_act_iz == full_act_iz and small_act_idz == full_act_idz:
            axes[1, 0].text(0.02, 0.98, "small == full\n(current ACT path uses first batch only)", transform=axes[1, 0].transAxes, va="top", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))
            axes[1, 1].text(0.02, 0.98, "small == full\n(current ACT path uses first batch only)", transform=axes[1, 1].transAxes, va="top", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
