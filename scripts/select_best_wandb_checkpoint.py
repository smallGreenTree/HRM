#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NON_CRASHED_STATES = {"finished"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the best non-crashed W&B run/step and print the corresponding checkpoint path."
    )
    parser.add_argument("--entity", help="W&B entity/team. Optional if your wandb config has a default entity.")
    parser.add_argument("--project", default="maze-hrm", help="W&B project name.")
    parser.add_argument("--metric", default="all.exact_accuracy", help="Metric to maximize.")
    parser.add_argument(
        "--metric-alias",
        action="append",
        dest="metric_aliases",
        help="Additional metric key to try. Repeatable.",
    )
    parser.add_argument("--state", action="append", dest="states", help="Allowed run state. Repeatable. Defaults to finished.")
    parser.add_argument("--run-name-contains", help="Only consider runs whose display name contains this text.")
    parser.add_argument("--exclude-run-name-contains", action="append", default=[], help="Skip runs whose name contains this text.")
    parser.add_argument("--training-only", action="store_true", help="Skip analysis_only runs according to W&B config.")
    parser.add_argument("--max-runs", type=int, default=200, help="Maximum recent runs to inspect.")
    parser.add_argument(
        "--checkpoint-root",
        help="Override checkpoint root. If set, checkpoint is <root>/<run-name>/step_<best-step><extension>.",
    )
    parser.add_argument("--checkpoint-extension", default=".pt", help="Checkpoint filename extension.")
    parser.add_argument("--output-json", help="Optional path to write selection details.")
    parser.add_argument("--print-layer-sweep-command", action="store_true", help="Print a ready layer sweep command.")
    return parser.parse_args()


def run_checkpoint_path(run: Any, step: int, args: argparse.Namespace) -> str:
    if args.checkpoint_root:
        base = Path(args.checkpoint_root) / run.name
    else:
        checkpoint_path = run.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                f"Run {run.name!r} has no checkpoint_path in W&B config. "
                "Pass --checkpoint-root to construct paths from run names."
            )
        base = Path(str(checkpoint_path))

    suffix = args.checkpoint_extension
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    return str(base / f"step_{step}{suffix}")


def metric_candidates(metric: str, aliases: list[str] | None) -> list[str]:
    candidates = [metric]
    if "/" in metric:
        candidates.append(metric.replace("/", "."))
    if "." in metric:
        candidates.append(metric.replace(".", "/"))
    candidates.extend(aliases or [])
    return list(dict.fromkeys(candidates))


def best_metric_row(run: Any, metrics: list[str]) -> tuple[str, int, float] | None:
    best_step = None
    best_value = None
    best_metric = None
    for metric in metrics:
        for row in run.scan_history(keys=["_step", metric], page_size=1000):
            if metric not in row or row[metric] is None:
                continue
            value = float(row[metric])
            step = int(row.get("_step", 0))
            if best_value is None or value > best_value:
                best_value = value
                best_step = step
                best_metric = metric
    if best_step is None or best_value is None:
        return None
    return str(best_metric), best_step, best_value


def main() -> None:
    args = parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it with `pip install wandb` in this environment.") from exc

    allowed_states = set(args.states or NON_CRASHED_STATES)
    metrics = metric_candidates(args.metric, args.metric_aliases)
    api = wandb.Api(timeout=60)
    path = f"{args.entity}/{args.project}" if args.entity else args.project

    candidates = []
    for run in api.runs(path, per_page=min(args.max_runs, 100)):
        if len(candidates) >= args.max_runs:
            break
        if run.state not in allowed_states:
            continue
        if args.run_name_contains and args.run_name_contains not in run.name:
            continue
        if any(skip in run.name for skip in args.exclude_run_name_contains):
            continue
        if args.training_only and bool(run.config.get("analysis_only", False)):
            continue

        best = best_metric_row(run, metrics)
        if best is None:
            continue
        metric_key, step, value = best
        candidates.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "metric_key": metric_key,
                "best_step": step,
                "best_metric": value,
                "checkpoint_path": run_checkpoint_path(run, step, args),
                "url": run.url,
            }
        )

    if not candidates:
        raise SystemExit(
            f"No runs matched project={path!r}, states={sorted(allowed_states)!r}, metrics={metrics!r}."
        )

    candidates.sort(key=lambda item: item["best_metric"], reverse=True)
    winner = candidates[0]

    print("Best non-crashed run")
    print(f"  run: {winner['run_name']} ({winner['run_id']})")
    print(f"  state: {winner['state']}")
    print(f"  metric: {winner['metric_key']} = {winner['best_metric']}")
    print(f"  step: {winner['best_step']}")
    print(f"  checkpoint: {winner['checkpoint_path']}")
    print(f"  url: {winner['url']}")

    if args.print_layer_sweep_command:
        print("\nLayer sweep command")
        print(f"CHECKPOINT_PATH={winner['checkpoint_path']} \\")
        print("OUTPUT_ROOT=/workspace/HRM/checkpoints/layer-intervention-sweep-best \\")
        print("bash scripts/run_layer_intervention_sweep.sh")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"winner": winner, "candidates": candidates}
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
