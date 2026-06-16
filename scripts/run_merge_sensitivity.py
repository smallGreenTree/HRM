#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from inforidge.behavior import per_example_behavior
from models.losses import IGNORE_LABEL_ID
from pretrain import PretrainConfig, create_dataloader, init_train_state, maybe_load_train_state


VALID_SOURCES = {
    "L": {"own", "H", "input"},
    "H": {"own", "L"},
}
METRICS = ["exact_accuracy", "cell_accuracy", "target_nll", "target_margin", "h_act_update_l2", "l_act_update_l2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure HRM sensitivity to fixed H/L/input merge weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/maze-30x30-hard-1k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--sources", nargs="+", default=["L:own", "L:H", "L:input", "H:own", "H:L"])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.25, 1.5])
    parser.add_argument("--act-steps", nargs="*", type=int, default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="merge-sensitivity-step24304")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PretrainConfig:
    config_dir = str((ROOT_DIR / "config").resolve())
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(
            config_name="cfg_pretrain",
            overrides=[
                f"data_path={args.data_path}",
                f"global_batch_size={args.global_batch_size}",
                f"load_checkpoint_path={args.checkpoint}",
                "analysis_only=true",
                f"+seed={args.seed}",
                "project_name=maze-hrm-merge-sensitivity",
                f"run_name={args.run_name}",
                "checkpoint_path=null",
                "+arch.layer_intervention_mode=none",
                "+arch.merge_intervention_level=null",
                "+arch.merge_intervention_source=null",
                "+arch.merge_intervention_scale=1.0",
                "+arch.merge_intervention_act_steps=[]",
            ],
        )
    return PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def parse_sources(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        try:
            level, source = value.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid source {value!r}; expected LEVEL:SOURCE, for example L:H") from exc
        level = level.upper()
        canonical = next((item for item in VALID_SOURCES.get(level, set()) if item.lower() == source.lower()), None)
        if canonical is None:
            allowed = ", ".join(f"{name}:{item}" for name, items in VALID_SOURCES.items() for item in sorted(items))
            raise ValueError(f"Invalid source {value!r}; allowed values: {allowed}")
        parsed.append((level, canonical))
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_example_rows(labels: torch.Tensor) -> torch.Tensor:
    return torch.nonzero((labels != IGNORE_LABEL_ID).any(dim=1), as_tuple=False).squeeze(-1)


def cache_eval_batches(eval_loader: Any, max_examples: int) -> list[dict[str, Any]]:
    cached = []
    set_offsets: dict[str, int] = defaultdict(int)
    example_count = 0
    for set_name, fixed_batch, _global_batch_size in eval_loader:
        if example_count >= max_examples:
            break
        rows = valid_example_rows(fixed_batch["labels"])[: max_examples - example_count]
        if rows.numel() == 0:
            continue
        batch = {key: value[rows].clone() for key, value in fixed_batch.items()}
        example_ids = [f"{set_name}:{set_offsets[set_name] + index:06d}" for index in range(rows.numel())]
        set_offsets[set_name] += rows.numel()
        example_count += rows.numel()
        cached.append({"set_name": set_name, "example_ids": example_ids, "batch": batch})
    return cached


def act_state_update(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    labels: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    if previous is None:
        return torch.full((current.shape[0],), float("nan"), device=current.device)
    token_l2 = (current[:, prefix_len:].float() - previous[:, prefix_len:].float()).norm(dim=-1)
    valid = (labels != IGNORE_LABEL_ID).to(token_l2.dtype)
    return (token_l2 * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)


def condition_name(level: str | None, source: str | None, scale: float) -> str:
    if level is None or source is None:
        return "baseline"
    scale_text = f"{scale:g}".replace("-", "neg").replace(".", "p")
    return f"{level}-{source}-scale-{scale_text}"


def set_condition(model_inner: Any, level: str | None, source: str | None, scale: float, act_steps: list[int]) -> None:
    model_inner.config.merge_intervention_level = level
    model_inner.config.merge_intervention_source = source
    model_inner.config.merge_intervention_scale = scale
    model_inner.config.merge_intervention_act_steps = list(act_steps)


def run_condition(
    *,
    model_core: Any,
    model_inner: Any,
    cached_batches: list[dict[str, Any]],
    level: str | None,
    source: str | None,
    scale: float,
    act_steps: list[int],
) -> list[dict[str, Any]]:
    set_condition(model_inner, level, source, scale, act_steps)
    name = condition_name(level, source, scale)
    source_label = "baseline" if level is None else f"{level}:{source}"
    rows = []

    with torch.inference_mode():
        for cached in cached_batches:
            batch = {key: value.cuda(non_blocking=True) for key, value in cached["batch"].items()}
            labels = batch["labels"].to(torch.int64)
            with torch.device("cuda"):
                carry = model_core.initial_carry(batch)
            previous_h = None
            previous_l = None
            act_step = 0

            while True:
                act_step += 1
                carry, outputs = model_core(carry=carry, batch=batch, return_z=True)
                behavior = per_example_behavior(outputs["logits"], labels)
                h_update = act_state_update(outputs["z_H"], previous_h, labels, model_inner.puzzle_emb_len)
                l_update = act_state_update(outputs["z_L"], previous_l, labels, model_inner.puzzle_emb_len)

                for index, example_id in enumerate(cached["example_ids"]):
                    rows.append(
                        {
                            "condition": name,
                            "merge_source": source_label,
                            "level": level or "",
                            "source": source or "",
                            "scale": scale,
                            "set_name": cached["set_name"],
                            "example_id": example_id,
                            "act_step": act_step,
                            "exact_accuracy": int(behavior["exact_accuracy"][index].item()),
                            "cell_accuracy": float(behavior["cell_accuracy"][index].item()),
                            "target_nll": float(behavior["target_nll"][index].item()),
                            "target_margin": float(behavior["target_margin"][index].item()),
                            "h_act_update_l2": float(h_update[index].item()),
                            "l_act_update_l2": float(l_update[index].item()),
                        }
                    )

                if bool(carry.halted.all().item()):
                    break
                previous_h = outputs["z_H"]
                previous_l = outputs["z_L"]

            del carry, outputs, batch
    return rows


def mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def bootstrap_interval(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    tensor = torch.tensor([value for value in values if not math.isnan(value)], dtype=torch.float64)
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, tensor.numel(), (samples, tensor.numel()), generator=generator)
    means = tensor[indices].mean(dim=1)
    return float(torch.quantile(means, 0.025).item()), float(torch.quantile(means, 0.975).item())


def add_paired_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = {
        (row["example_id"], row["act_step"]): row
        for row in rows
        if row["condition"] == "baseline"
    }
    for row in rows:
        base = baseline[(row["example_id"], row["act_step"])]
        for metric in METRICS:
            row[f"delta_{metric}"] = float(row[metric]) - float(base[metric])


def summarize(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["act_step"])].append(row)

    output = []
    for (condition, act_step), group in sorted(grouped.items()):
        first = group[0]
        summary = {
            "condition": condition,
            "merge_source": first["merge_source"],
            "level": first["level"],
            "source": first["source"],
            "scale": first["scale"],
            "act_step": act_step,
            "examples": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            deltas = [float(row[f"delta_{metric}"]) for row in group]
            low, high = bootstrap_interval(deltas, bootstrap_samples, seed + act_step)
            summary[metric] = mean(values)
            summary[f"delta_{metric}"] = mean(deltas)
            summary[f"delta_{metric}_ci_low"] = low
            summary[f"delta_{metric}_ci_high"] = high
        output.append(summary)
    return output


def final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["condition"], row["example_id"])
        if key not in final or row["act_step"] > final[key]["act_step"]:
            final[key] = row
    return list(final.values())


def preferred_scales(
    final: list[dict[str, Any]], sources: list[tuple[str, str]], scales: list[float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {row["example_id"]: row for row in final if row["condition"] == "baseline"}
    by_condition = {(row["condition"], row["example_id"]): row for row in final}
    choices = []
    counts = []

    for level, source in sources:
        source_label = f"{level}:{source}"
        counter: Counter[float] = Counter()
        for example_id, base in baseline.items():
            candidates = [(1.0, float(base["target_nll"]))]
            for scale in scales:
                row = by_condition[(condition_name(level, source, scale), example_id)]
                candidates.append((scale, float(row["target_nll"])))
            best_scale, best_nll = min(candidates, key=lambda item: item[1])
            counter[best_scale] += 1
            choices.append(
                {
                    "merge_source": source_label,
                    "example_id": example_id,
                    "preferred_scale": best_scale,
                    "preferred_target_nll": best_nll,
                    "baseline_target_nll": float(base["target_nll"]),
                    "nll_improvement": float(base["target_nll"]) - best_nll,
                }
            )
        for scale in sorted({1.0, *scales}):
            counts.append(
                {
                    "merge_source": source_label,
                    "scale": scale,
                    "preferred_examples": counter[scale],
                    "fraction": counter[scale] / max(len(baseline), 1),
                }
            )
    return choices, counts


def plot_final_effects(summary_rows: list[dict[str, Any]], sources: list[tuple[str, str]], output_dir: Path) -> None:
    final_step = max(int(row["act_step"]) for row in summary_rows)
    final = [row for row in summary_rows if int(row["act_step"]) == final_step]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for level, source in sources:
        label = f"{level}:{source}"
        data = [row for row in final if row["merge_source"] == label]
        points = [(1.0, 0.0, 0.0)] + [
            (float(row["scale"]), float(row["delta_target_nll"]), float(row["delta_exact_accuracy"]))
            for row in data
        ]
        points.sort()
        axes[0].plot([x[0] for x in points], [x[1] for x in points], marker="o", label=label)
        axes[1].plot([x[0] for x in points], [x[2] for x in points], marker="o", label=label)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[0].set(title=f"Final target NLL sensitivity (ACT {final_step})", xlabel="Source scale", ylabel="Paired delta NLL")
    axes[1].set(title=f"Final exact-accuracy sensitivity (ACT {final_step})", xlabel="Source scale", ylabel="Paired delta exact accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "final_merge_sensitivity.png", dpi=180)
    plt.close(fig)


def plot_best_scale_by_act(summary_rows: list[dict[str, Any]], sources: list[tuple[str, str]], output_dir: Path) -> None:
    baseline = {int(row["act_step"]): float(row["target_nll"]) for row in summary_rows if row["condition"] == "baseline"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for level, source in sources:
        label = f"{level}:{source}"
        data = [row for row in summary_rows if row["merge_source"] == label]
        by_step: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in data:
            by_step[int(row["act_step"])].append((float(row["scale"]), float(row["target_nll"])))
        steps = sorted(by_step)
        best = []
        for step in steps:
            candidates = [(1.0, baseline[step]), *by_step[step]]
            best.append(min(candidates, key=lambda item: item[1])[0])
        ax.plot(steps, best, marker="o", label=label)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="original scale")
    ax.set(title="Descriptive NLL-optimal fixed scale by ACT step", xlabel="ACT step", ylabel="Scale with lowest mean target NLL")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "best_scale_by_act.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Merge sensitivity requires a CUDA GPU for the trained HRM checkpoint.")
    if args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    sources = parse_sources(args.sources)
    scales = sorted(set(args.scales))
    if 1.0 in scales:
        scales.remove(1.0)

    os.environ.setdefault("DISABLE_COMPILE", "1")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    _train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    eval_loader, _eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    cached_batches = cache_eval_batches(eval_loader, args.max_examples)
    example_count = sum(len(batch["example_ids"]) for batch in cached_batches)
    if example_count == 0:
        raise RuntimeError("No valid evaluation examples were loaded")

    train_state = init_train_state(config, train_metadata, world_size=1)
    maybe_load_train_state(config, train_state)
    train_state.model.eval()
    model_core = train_state.model.model  # type: ignore[attr-defined]
    model_inner = model_core.inner

    all_rows = run_condition(
        model_core=model_core,
        model_inner=model_inner,
        cached_batches=cached_batches,
        level=None,
        source=None,
        scale=1.0,
        act_steps=args.act_steps,
    )
    print(f"Completed baseline ({example_count} examples)", flush=True)

    for level, source in sources:
        for scale in scales:
            all_rows.extend(
                run_condition(
                    model_core=model_core,
                    model_inner=model_inner,
                    cached_batches=cached_batches,
                    level=level,
                    source=source,
                    scale=scale,
                    act_steps=args.act_steps,
                )
            )
            print(f"Completed {level}:{source} scale={scale:g}", flush=True)

    add_paired_deltas(all_rows)
    summary_rows = summarize(all_rows, args.bootstrap_samples, args.seed)
    final = final_rows(all_rows)
    choices, counts = preferred_scales(final, sources, scales)

    write_csv(output_dir / "merge_sensitivity_by_example_act.csv", all_rows)
    write_csv(output_dir / "merge_sensitivity_summary_by_act.csv", summary_rows)
    write_csv(output_dir / "merge_sensitivity_final_by_example.csv", final)
    write_csv(output_dir / "preferred_scale_by_example.csv", choices)
    write_csv(output_dir / "preferred_scale_counts.csv", counts)
    plot_final_effects(summary_rows, sources, output_dir)
    plot_best_scale_by_act(summary_rows, sources, output_dir)

    manifest = {
        "experiment": "fixed merge sensitivity",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "examples": example_count,
        "sources": [f"{level}:{source}" for level, source in sources],
        "scales": [1.0, *scales],
        "act_steps": args.act_steps or "all",
        "merge_definitions": {
            "L": "scale one source in z_L + z_H + input_embeddings before the L blocks",
            "H": "scale one source in z_H + z_L before the H blocks",
        },
        "hypothesis": "If useful mixture weights differ by source, ACT stage, or maze, learned H/L gating is motivated.",
        "decision_rule": "Proceed to learned gating only if non-unit scales improve paired NLL or accuracy systematically, or preferred scales vary reproducibly across stages or maze groups.",
        "behavior_metrics": ["target_nll", "target_margin", "cell_accuracy", "exact_accuracy"],
        "dynamics_metrics": ["h_act_update_l2", "l_act_update_l2"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Merge sensitivity complete: {output_dir}")


if __name__ == "__main__":
    main()
