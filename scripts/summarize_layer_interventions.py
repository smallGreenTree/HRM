#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize layer intervention analysis_only outputs.")
    parser.add_argument("sweep_dir", help="Directory produced by scripts/run_layer_intervention_sweep.sh")
    parser.add_argument("--output-csv", help="Optional path to write the summary table.")
    return parser.parse_args()


def read_last_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    row = rows[-1]
    return {k: float(v) for k, v in row.items() if v != ""}


def parse_case(name: str) -> tuple[str, str, str]:
    parts = name.split("-")
    if parts[-1] == "baseline":
        return "baseline", "", "baseline"
    mode = parts[-1]
    layer = parts[-2]
    if len(layer) >= 2 and layer[0] in {"H", "L"}:
        return layer[0], layer[1:], mode
    return "", "", mode


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")

    rows: list[dict[str, str | float]] = []
    baseline = None

    for metrics_path in sorted(sweep_dir.glob("*/eval_metrics.csv")):
        case_name = metrics_path.parent.name
        metrics = read_last_metrics(metrics_path)
        level, layer, mode = parse_case(case_name)
        row: dict[str, str | float] = {
            "case": case_name,
            "level": level,
            "layer": layer,
            "mode": mode,
            "step": metrics.get("step", 0.0),
            "accuracy": metrics.get("all/accuracy", float("nan")),
            "exact_accuracy": metrics.get("all/exact_accuracy", float("nan")),
            "steps": metrics.get("all/steps", float("nan")),
            "lm_loss": metrics.get("all/lm_loss", float("nan")),
        }
        rows.append(row)
        if mode == "baseline":
            baseline = row

    if baseline is None:
        raise ValueError(f"No baseline eval_metrics.csv found under {sweep_dir}")

    base_acc = float(baseline["accuracy"])
    base_exact = float(baseline["exact_accuracy"])
    for row in rows:
        row["accuracy_delta"] = float(row["accuracy"]) - base_acc
        row["exact_accuracy_delta"] = float(row["exact_accuracy"]) - base_exact

    rows.sort(key=lambda row: (float(row["exact_accuracy_delta"]), float(row["accuracy_delta"])))

    fieldnames = [
        "case",
        "level",
        "layer",
        "mode",
        "step",
        "accuracy",
        "accuracy_delta",
        "exact_accuracy",
        "exact_accuracy_delta",
        "steps",
        "lm_loss",
    ]

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fieldnames))

    if args.output_csv:
        output_path = Path(args.output_csv).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
