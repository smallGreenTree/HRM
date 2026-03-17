#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot InfoRidge CSV outputs.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing inforidge_step_*.csv and optionally inforidge_act_mi_step_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs. Defaults to <input-dir>/plots.",
    )
    parser.add_argument(
        "--step-file",
        default=None,
        help="Specific layerwise CSV filename. Defaults to the latest inforidge_step_*.csv in input-dir.",
    )
    parser.add_argument(
        "--act-file",
        default=None,
        help="Specific ACT-step CSV filename. Defaults to the latest inforidge_act_mi_step_*.csv in input-dir if present.",
    )
    return parser.parse_args()


def latest_matching(input_dir: Path, pattern: str) -> Path | None:
    matches = sorted(input_dir.glob(pattern))
    return matches[-1] if matches else None


def read_layerwise_csv(path: Path) -> tuple[list[int], list[float], list[float | None], list[int], list[float], list[float | None]]:
    h_x: list[int] = []
    h_iz: list[float] = []
    h_idz: list[float | None] = []
    l_x: list[int] = []
    l_iz: list[float] = []
    l_idz: list[float | None] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_type = row["layer_type"]
            layer_index = int(row["layer_index"])
            i_z_y = float(row["I_Z_Y"])
            i_dz_y = None if row["I_dZ_Y"] == "" else float(row["I_dZ_Y"])

            if layer_type == "H":
                h_x.append(layer_index)
                h_iz.append(i_z_y)
                h_idz.append(i_dz_y)
            elif layer_type == "L":
                l_x.append(layer_index)
                l_iz.append(i_z_y)
                l_idz.append(i_dz_y)
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")

    return h_x, h_iz, h_idz, l_x, l_iz, l_idz


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


def plot_layerwise(path: Path, output_path: Path) -> None:
    h_x, h_iz, h_idz, l_x, l_iz, l_idz = read_layerwise_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(h_x, h_iz, marker="o", label="H")
    axes[0].plot(l_x, l_iz, marker="o", label="L")
    axes[0].set_title("Predictive MI")
    axes[0].set_xlabel("Layer Depth")
    axes[0].set_ylabel("I(Z;Y)")
    axes[0].set_xticks(sorted(set(h_x + l_x)))
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    h_idz_x = [x for x, y in zip(h_x, h_idz) if y is not None]
    h_idz_y = [y for y in h_idz if y is not None]
    l_idz_x = [x for x, y in zip(l_x, l_idz) if y is not None]
    l_idz_y = [y for y in l_idz if y is not None]

    axes[1].plot(h_idz_x, h_idz_y, marker="o", label="H")
    axes[1].plot(l_idz_x, l_idz_y, marker="o", label="L")
    axes[1].set_title("Incremental MI")
    axes[1].set_xlabel("Layer Depth")
    axes[1].set_ylabel("I(dZ;Y)")
    axes[1].set_xticks(sorted(set(h_x + l_x)))
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(path.name)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_act(path: Path, output_path: Path) -> None:
    x, iz, idz = read_act_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(x, iz, marker="o")
    axes[0].set_title("ACT Predictive MI")
    axes[0].set_xlabel("ACT Step")
    axes[0].set_ylabel("I(Z;Y)")
    axes[0].grid(alpha=0.3)

    idz_x = [step for step, value in zip(x, idz) if value is not None]
    idz_y = [value for value in idz if value is not None]
    axes[1].plot(idz_x, idz_y, marker="o")
    axes[1].set_title("ACT Incremental MI")
    axes[1].set_xlabel("ACT Step")
    axes[1].set_ylabel("I(dZ;Y)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(path.name)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_file = Path(args.step_file) if args.step_file else latest_matching(input_dir, "inforidge_step_*.csv")
    if step_file is not None and not step_file.is_absolute():
        step_file = input_dir / step_file

    act_file = Path(args.act_file) if args.act_file else latest_matching(input_dir, "inforidge_act_mi_step_*.csv")
    if act_file is not None and not act_file.is_absolute():
        act_file = input_dir / act_file

    if step_file is None or not step_file.exists():
        raise FileNotFoundError("No layerwise InfoRidge CSV found.")

    layerwise_png = output_dir / "inforidge_layerwise.png"
    plot_layerwise(step_file, layerwise_png)
    print(f"Wrote {layerwise_png}")

    if act_file is not None and act_file.exists():
        act_png = output_dir / "inforidge_act_mi.png"
        plot_act(act_file, act_png)
        print(f"Wrote {act_png}")


if __name__ == "__main__":
    main()
