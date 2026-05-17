#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log representative hidden-dynamics example plots to W&B.")
    parser.add_argument("--examples-csv", required=True, help="hidden_dynamics_examples.csv")
    parser.add_argument("--project", default="maze-hrm")
    parser.add_argument("--entity")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--output-dir", help="Optional directory to save PNGs locally.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def by_example(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["example_index"]), []).append(row)
    for example_rows in grouped.values():
        example_rows.sort(key=lambda row: int(row["act_step"]))
    return grouped


def select_examples(grouped: dict[int, list[dict[str, str]]]) -> list[tuple[str, int]]:
    scored = []
    for idx, rows in grouped.items():
        success = int(rows[0]["success"])
        final_h_l_distance = float(rows[-1]["h_l_distance"])
        total_l_delta = sum(float(row["l_delta"]) for row in rows[1:])
        total_h_delta = sum(float(row["h_delta"]) for row in rows[1:])
        difficulty = total_h_delta + total_l_delta + final_h_l_distance
        scored.append((idx, success, difficulty))

    successes = sorted((x for x in scored if x[1] == 1), key=lambda x: x[2])
    failures = sorted((x for x in scored if x[1] == 0), key=lambda x: x[2], reverse=True)

    selected: list[tuple[str, int]] = []
    if successes:
        selected.append(("easy_success", successes[0][0]))
        selected.append(("hard_success", successes[-1][0]))
        if len(successes) > 2:
            selected.append(("typical_success", successes[len(successes) // 2][0]))
    if failures:
        selected.append(("failure", failures[0][0]))
    return selected


def series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def plot_example(rows: list[dict[str, str]], title: str, output_path: Path) -> None:
    steps = [int(row["act_step"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(steps, series(rows, "h_norm"), marker="o", label="H norm")
    axes[0, 0].plot(steps, series(rows, "l_norm"), marker="s", label="L norm")
    axes[0, 0].set_title("Hidden State Norms")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, series(rows, "h_delta"), marker="o", label="delta H")
    axes[0, 1].plot(steps, series(rows, "l_delta"), marker="s", label="delta L")
    axes[0, 1].set_title("Update Magnitudes")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, series(rows, "h_l_cosine"), marker="o", color="tab:purple")
    axes[1, 0].set_title("H/L Cosine Similarity")

    axes[1, 1].plot(steps, series(rows, "h_l_distance"), marker="o", color="tab:red")
    axes[1, 1].set_title("H/L Distance")

    for ax in axes.flat:
        ax.set_xlabel("ACT step")
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    examples_csv = Path(args.examples_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else examples_csv.parent / "example_plots"
    rows = read_rows(examples_csv)
    grouped = by_example(rows)
    selected = select_examples(grouped)
    if not selected:
        raise ValueError("No examples available to plot.")

    run = wandb.init(entity=args.entity, project=args.project, name=args.run_name, job_type="hidden-dynamics-plot")
    log_payload = {}
    table = wandb.Table(columns=["label", "example_index", "success"])

    for label, example_idx in selected:
        example_rows = grouped[example_idx]
        success = int(example_rows[0]["success"])
        title = f"{args.checkpoint_label} | {label} | example {example_idx} | success={success}"
        output_path = output_dir / f"{label}_example_{example_idx}.png"
        plot_example(example_rows, title, output_path)
        log_payload[f"hidden_dynamics/{label}"] = wandb.Image(str(output_path), caption=title)
        table.add_data(label, example_idx, success)

    log_payload["hidden_dynamics/selected_examples"] = table
    run.log(log_payload)
    run.finish()
    print(f"Logged {len(selected)} example plot(s) to W&B.")
    print(f"Saved plots in {output_dir}")


if __name__ == "__main__":
    main()
