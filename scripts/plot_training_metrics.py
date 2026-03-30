#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HRM training/eval metrics from CSV.")
    parser.add_argument("--eval-csv", required=True, help="Path to eval_metrics.csv")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument(
        "--title",
        default="HRM Evaluation Metrics",
        help="Figure title",
    )
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


def main() -> None:
    args = parse_args()
    eval_csv = Path(args.eval_csv).resolve()
    output_path = Path(args.output).resolve()

    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV does not exist: {eval_csv}")

    data = read_eval_metrics(eval_csv)
    steps = data["step"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(steps, data["all/accuracy"], marker="o", linewidth=2)
    axes[0, 0].set_title("Eval Accuracy")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, data["all/exact_accuracy"], marker="o", linewidth=2)
    axes[0, 1].set_title("Eval Exact Accuracy")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("exact_accuracy")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, data["all/lm_loss"], marker="o", linewidth=2)
    axes[1, 0].set_title("Eval LM Loss")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("lm_loss")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, data["all/q_halt_loss"], marker="o", linewidth=2, label="q_halt_loss")
    if "all/q_halt_accuracy" in data:
        axes[1, 1].plot(steps, data["all/q_halt_accuracy"], marker="s", linewidth=2, label="q_halt_accuracy")
    axes[1, 1].set_title("Eval Halt Head")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
