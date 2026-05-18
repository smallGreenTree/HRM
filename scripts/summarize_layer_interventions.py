#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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


def maybe_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_layerwise(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (row["layer_type"], row["layer_index"]): {
            "I_Z_Y": maybe_float(row.get("I_Z_Y")),
            "I_dZ_Y": maybe_float(row.get("I_dZ_Y")),
        }
        for row in rows
    }


def read_act(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["act_step"]: {
            "I_Z_Y": maybe_float(row.get("I_Z_Y")),
            "I_dZ_Y": maybe_float(row.get("I_dZ_Y")),
        }
        for row in rows
    }


def latest_file(case_dir: Path, pattern: str) -> Path | None:
    matches = sorted(case_dir.glob(pattern))
    return matches[-1] if matches else None


def mean_delta(current: dict, baseline: dict, metric: str, module: str | None = None) -> float:
    deltas = []
    for key, values in current.items():
        if module is not None:
            if not isinstance(key, tuple) or key[0] != module:
                continue
        base = baseline.get(key)
        if base is None:
            continue
        value = values.get(metric, float("nan"))
        base_value = base.get(metric, float("nan"))
        if not math.isnan(value) and not math.isnan(base_value):
            deltas.append(value - base_value)
    return sum(deltas) / len(deltas) if deltas else float("nan")


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

    rows: list[dict[str, str | float | Path | None]] = []
    baseline = None
    baseline_layerwise = None
    baseline_act = None

    for metrics_path in sorted(sweep_dir.glob("*/eval_metrics.csv")):
        case_name = metrics_path.parent.name
        case_dir = metrics_path.parent
        metrics = read_last_metrics(metrics_path)
        level, layer, mode = parse_case(case_name)
        layerwise_path = latest_file(case_dir, "inforidge_step_*.csv")
        act_path = latest_file(case_dir, "inforidge_act_mi_step_*.csv")
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
            "q_halt_accuracy": metrics.get("all/q_halt_accuracy", float("nan")),
            "q_halt_loss": metrics.get("all/q_halt_loss", float("nan")),
        }
        row["_layerwise_path"] = layerwise_path
        row["_act_path"] = act_path
        rows.append(row)
        if mode == "baseline":
            baseline = row
            if layerwise_path is not None:
                baseline_layerwise = read_layerwise(layerwise_path)
            if act_path is not None:
                baseline_act = read_act(act_path)

    if baseline is None:
        raise ValueError(f"No baseline eval_metrics.csv found under {sweep_dir}")

    base_acc = float(baseline["accuracy"])
    base_exact = float(baseline["exact_accuracy"])
    for row in rows:
        row["accuracy_delta"] = float(row["accuracy"]) - base_acc
        row["exact_accuracy_delta"] = float(row["exact_accuracy"]) - base_exact
        layerwise_path = row.pop("_layerwise_path")
        act_path = row.pop("_act_path")
        if layerwise_path is not None and baseline_layerwise is not None:
            layerwise = read_layerwise(layerwise_path)
            row["mean_layer_delta_I_Z_Y"] = mean_delta(layerwise, baseline_layerwise, "I_Z_Y")
            row["mean_layer_delta_I_dZ_Y"] = mean_delta(layerwise, baseline_layerwise, "I_dZ_Y")
            row["mean_H_delta_I_Z_Y"] = mean_delta(layerwise, baseline_layerwise, "I_Z_Y", module="H")
            row["mean_L_delta_I_Z_Y"] = mean_delta(layerwise, baseline_layerwise, "I_Z_Y", module="L")
        else:
            row["mean_layer_delta_I_Z_Y"] = float("nan")
            row["mean_layer_delta_I_dZ_Y"] = float("nan")
            row["mean_H_delta_I_Z_Y"] = float("nan")
            row["mean_L_delta_I_Z_Y"] = float("nan")

        if act_path is not None and baseline_act is not None:
            act = read_act(act_path)
            row["mean_act_delta_I_Z_Y"] = mean_delta(act, baseline_act, "I_Z_Y")
            row["mean_act_delta_I_dZ_Y"] = mean_delta(act, baseline_act, "I_dZ_Y")
        else:
            row["mean_act_delta_I_Z_Y"] = float("nan")
            row["mean_act_delta_I_dZ_Y"] = float("nan")

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
        "q_halt_accuracy",
        "q_halt_loss",
        "mean_layer_delta_I_Z_Y",
        "mean_layer_delta_I_dZ_Y",
        "mean_H_delta_I_Z_Y",
        "mean_L_delta_I_Z_Y",
        "mean_act_delta_I_Z_Y",
        "mean_act_delta_I_dZ_Y",
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
