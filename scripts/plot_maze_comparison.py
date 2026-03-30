#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot old vs corrected maze run comparisons.")
    parser.add_argument("--old-eval-csv", required=True)
    parser.add_argument("--new-eval-csv", required=True)
    parser.add_argument("--old-inforidge-csv", required=True)
    parser.add_argument("--old-full-inforidge-csv", required=True)
    parser.add_argument("--new-inforidge-csv", required=True)
    parser.add_argument("--old-act-csv", required=True)
    parser.add_argument("--old-full-act-csv", required=True)
    parser.add_argument("--new-act-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_eval_metrics(path: Path) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                metrics.setdefault(key, [])
                metrics[key].append(float(value))
    return metrics


def read_layerwise(path: Path) -> dict[str, dict[str, list[float]]]:
    rows = list(csv.DictReader(path.open(newline="")))
    out: dict[str, dict[str, list[float]]] = {}
    for layer_type in ("H", "L"):
        layer_rows = [row for row in rows if row["layer_type"] == layer_type]
        out[layer_type] = {
            "layer_index": [float(row["layer_index"]) for row in layer_rows],
            "I_Z_Y": [float(row["I_Z_Y"]) for row in layer_rows],
            "I_dZ_Y": [float("nan") if row["I_dZ_Y"] == "" else float(row["I_dZ_Y"]) for row in layer_rows],
        }
    return out


def read_act(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(newline="")))
    return {
        "act_step": [float(row["act_step"]) for row in rows],
        "I_Z_Y": [float(row["I_Z_Y"]) for row in rows],
        "I_dZ_Y": [float("nan") if row["I_dZ_Y"] == "" else float(row["I_dZ_Y"]) for row in rows],
    }


def plot_eval(old_eval: dict[str, list[float]], new_eval: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = [
        ("all/accuracy", "Eval Accuracy", axes[0, 0]),
        ("all/exact_accuracy", "Eval Exact Accuracy", axes[0, 1]),
        ("all/lm_loss", "Eval LM Loss", axes[1, 0]),
        ("all/q_halt_accuracy", "Eval Halt Accuracy", axes[1, 1]),
    ]

    for key, title, ax in series:
        ax.plot(old_eval["step"], old_eval[key], marker="o", linewidth=2, label="Old run")
        ax.plot(new_eval["step"], new_eval[key], marker="s", linewidth=2, label="Corrected run")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Maze Eval Comparison: Old vs Corrected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layerwise(
    old_small: dict[str, dict[str, list[float]]],
    old_full: dict[str, dict[str, list[float]]],
    new: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("H", "I_Z_Y", axes[0, 0], "H Predictive MI"),
        ("L", "I_Z_Y", axes[0, 1], "L Predictive MI"),
        ("H", "I_dZ_Y", axes[1, 0], "H Incremental MI"),
        ("L", "I_dZ_Y", axes[1, 1], "L Incremental MI"),
    ]

    for layer_type, key, ax, title in panels:
        ax.plot(old_small[layer_type]["layer_index"], old_small[layer_type][key], marker="o", linewidth=2, label="Old small")
        ax.plot(old_full[layer_type]["layer_index"], old_full[layer_type][key], marker="s", linewidth=2, label="Old full")
        ax.plot(new[layer_type]["layer_index"], new[layer_type][key], marker="^", linewidth=2, label="Corrected")
        ax.set_title(title)
        ax.set_xlabel("Layer Index")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Maze InfoRidge Comparison: Layerwise")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_act(
    old_small: dict[str, list[float]],
    old_full: dict[str, list[float]],
    new: dict[str, list[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    panels = [
        ("I_Z_Y", axes[0], "ACT Predictive MI"),
        ("I_dZ_Y", axes[1], "ACT Incremental MI"),
    ]

    for key, ax, title in panels:
        ax.plot(old_small["act_step"], old_small[key], marker="o", linewidth=2, label="Old small")
        ax.plot(old_full["act_step"], old_full[key], marker="s", linewidth=2, label="Old full")
        ax.plot(new["act_step"], new[key], marker="^", linewidth=2, label="Corrected")
        ax.set_title(title)
        ax.set_xlabel("ACT Step")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Maze InfoRidge Comparison: ACT Steps")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    old_eval = read_eval_metrics(Path(args.old_eval_csv).resolve())
    new_eval = read_eval_metrics(Path(args.new_eval_csv).resolve())
    old_small = read_layerwise(Path(args.old_inforidge_csv).resolve())
    old_full = read_layerwise(Path(args.old_full_inforidge_csv).resolve())
    new = read_layerwise(Path(args.new_inforidge_csv).resolve())
    old_act = read_act(Path(args.old_act_csv).resolve())
    old_full_act = read_act(Path(args.old_full_act_csv).resolve())
    new_act = read_act(Path(args.new_act_csv).resolve())

    plot_eval(old_eval, new_eval, output_dir / "eval_comparison.png")
    plot_layerwise(old_small, old_full, new, output_dir / "layerwise_inforidge_comparison.png")
    plot_act(old_act, old_full_act, new_act, output_dir / "act_inforidge_comparison.png")

    print(f"Wrote {output_dir / 'eval_comparison.png'}")
    print(f"Wrote {output_dir / 'layerwise_inforidge_comparison.png'}")
    print(f"Wrote {output_dir / 'act_inforidge_comparison.png'}")


if __name__ == "__main__":
    main()
