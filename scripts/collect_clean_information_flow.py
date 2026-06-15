#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
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
from inforidge.inforidge import l2_normalize, matrix_mutual_information
from models.losses import IGNORE_LABEL_ID
from pretrain import PretrainConfig, create_dataloader, init_train_state, maybe_load_train_state


STATE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the clean HRM information-flow baseline.")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint .pt file.")
    parser.add_argument("--data-path", default="data/maze-30x30-hard-1k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--state-dtype", choices=sorted(STATE_DTYPES), default="float16")
    parser.add_argument("--kernel-sigma", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="clean-information-flow")
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
                "project_name=clean-information-flow",
                f"run_name={args.run_name}",
                "checkpoint_path=null",
                "+arch.layer_intervention_mode=none",
                "+arch.layer_intervention_level=null",
                "+arch.layer_intervention_layers=[]",
                "arch.layer_intervention_act_steps=[]",
            ],
        )
    return PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


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


def last_valid_positions(labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE_LABEL_ID
    reverse_index = torch.argmax(torch.flip(valid, dims=[1]).to(torch.int64), dim=1)
    return labels.shape[1] - 1 - reverse_index


def pack_boundaries(inputs: list[torch.Tensor], after: list[torch.Tensor], trace_name: str) -> torch.Tensor:
    """Pack one module call as [before block 0, after block 0, ..., after block N]."""
    if len(inputs) != len(after) or not after:
        raise ValueError(f"Invalid {trace_name}: {len(inputs)} inputs, {len(after)} outputs")
    for block_index in range(1, len(inputs)):
        if not torch.equal(inputs[block_index], after[block_index - 1]):
            raise RuntimeError(f"{trace_name} block {block_index} input does not equal the preceding block output")
    return torch.stack([inputs[0], *after], dim=1)


def occurrence_boundaries(outputs: dict[str, Any], level: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    traces = outputs[f"block_traces_{level}"]
    boundaries = []
    metadata = []
    for occurrence_index, trace in enumerate(traces):
        boundaries.append(pack_boundaries(trace["inputs"], trace["outputs"], f"{level} occurrence {occurrence_index}"))
        metadata.append(
            {
                "occurrence_index": occurrence_index,
                "h_cycle": int(trace["h_cycle"]),
                "l_cycle": "" if trace["l_cycle"] is None else int(trace["l_cycle"]),
                "phase": str(trace["phase"]),
            }
        )
    return torch.stack(boundaries, dim=1), metadata


def masked_update_magnitude(boundaries: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean tokenwise L2 update, returned as [batch, occurrence, block]."""
    updates = boundaries[:, :, 1:].float() - boundaries[:, :, :-1].float()
    token_l2 = updates.norm(dim=-1)
    mask = (labels != IGNORE_LABEL_ID).to(token_l2.dtype).unsqueeze(1).unsqueeze(1)
    return (token_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def act_state_update_magnitude(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    labels: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    if previous is None:
        return torch.full((current.shape[0],), float("nan"), device=current.device)
    token_l2 = (current[:, prefix_len:].float() - previous[:, prefix_len:].float()).norm(dim=-1)
    mask = (labels != IGNORE_LABEL_ID).to(token_l2.dtype)
    return (token_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def target_vectors(model_inner: Any, batch: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the current second-pass, last-valid-token InfoRidge target."""
    positions = last_valid_positions(labels)
    rows = torch.arange(labels.shape[0], device=labels.device)
    second_inputs = batch["inputs"].clone()
    second_inputs[rows, positions] = labels[rows, positions].to(second_inputs.dtype)
    embedding = model_inner._input_embeddings(second_inputs.to(torch.int32), batch["puzzle_identifiers"])
    prefix_len = int(getattr(model_inner, "prefix_len", getattr(model_inner, "puzzle_emb_len", 0)))
    targets = embedding[rows, positions + prefix_len].float()
    return l2_normalize(targets), positions + prefix_len


def append_inforidge_vectors(
    store: dict[tuple[int, str, int, int, str], list[torch.Tensor]],
    level: str,
    act_step: int,
    boundaries: torch.Tensor,
    positions_with_prefix: torch.Tensor,
) -> None:
    gather_index = positions_with_prefix.view(-1, 1, 1, 1, 1).expand(
        -1, boundaries.shape[1], boundaries.shape[2], 1, boundaries.shape[4]
    )
    selected = boundaries.gather(dim=3, index=gather_index).squeeze(3).float()
    selected = l2_normalize(selected)
    for occurrence_index in range(boundaries.shape[1]):
        for block_index in range(boundaries.shape[2] - 1):
            before = selected[:, occurrence_index, block_index]
            after = selected[:, occurrence_index, block_index + 1]
            update = l2_normalize(after - before)
            store[(act_step, level, occurrence_index, block_index, "before")].append(before.cpu())
            store[(act_step, level, occurrence_index, block_index, "after")].append(after.cpu())
            store[(act_step, level, occurrence_index, block_index, "update")].append(update.cpu())


def analyze_solve_trajectory(step_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        by_example[str(row["example_id"])].append(row)

    result: dict[str, dict[str, int]] = {}
    for example_id, rows in by_example.items():
        rows.sort(key=lambda row: int(row["act_step"]))
        solved_steps = [int(row["act_step"]) for row in rows if int(row["exact_accuracy"]) == 1]
        first_solved = solved_steps[0] if solved_steps else 0
        regressed = int(first_solved > 0 and any(int(row["exact_accuracy"]) == 0 for row in rows[first_solved:]))
        result[example_id] = {
            "first_solved_step": first_solved,
            "regressed_after_solve": regressed,
            "final_exact_accuracy": int(rows[-1]["exact_accuracy"]),
        }
    return result


def current_inforidge_rows(
    vectors: dict[tuple[int, str, int, int, str], list[torch.Tensor]],
    targets: list[torch.Tensor],
    occurrence_metadata: dict[str, list[dict[str, Any]]],
    sigma: float,
) -> list[dict[str, Any]]:
    y = torch.cat(targets, dim=0)
    rows: list[dict[str, Any]] = []
    keys = sorted({(act, level, occurrence, block) for act, level, occurrence, block, _phase in vectors})
    for act_step, level, occurrence_index, block_index in keys:
        values = {}
        for phase in ("before", "after", "update"):
            z = torch.cat(vectors[(act_step, level, occurrence_index, block_index, phase)], dim=0)
            values[phase] = float(matrix_mutual_information(z, y, sigma=sigma).item())
        rows.append(
            {
                "act_step": act_step,
                "level": level,
                **occurrence_metadata[level][occurrence_index],
                "block_index": block_index,
                "sample_count": y.shape[0],
                "kernel_sigma": sigma,
                "target_mode": "last_valid_token_second_pass",
                "I_before_Y": values["before"],
                "I_after_Y": values["after"],
                "delta_marginal_I": values["after"] - values["before"],
                "I_update_Y": values["update"],
            }
        )
    return rows


def summarize_behavior(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    steps = sorted({int(row["act_step"]) for row in rows})
    metrics = ["exact_accuracy", "cell_accuracy", "target_nll", "target_margin"]
    summary: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)
    for step in steps:
        step_rows = [row for row in rows if int(row["act_step"]) == step]
        output: dict[str, Any] = {"act_step": step, "count": len(step_rows)}
        for metric in metrics + ["h_act_update_l2", "l_act_update_l2"]:
            values = torch.tensor(
                [float(row[metric]) for row in step_rows if math.isfinite(float(row[metric]))],
                dtype=torch.float64,
            )
            if values.numel() == 0:
                output[f"{metric}_mean"] = float("nan")
                output[f"{metric}_ci_low"] = float("nan")
                output[f"{metric}_ci_high"] = float("nan")
                continue
            sample_indices = torch.randint(
                values.numel(),
                (bootstrap_samples, values.numel()),
                generator=generator,
            )
            bootstrap_means = values[sample_indices].mean(dim=1)
            output[f"{metric}_mean"] = float(values.mean().item())
            output[f"{metric}_ci_low"] = float(torch.quantile(bootstrap_means, 0.025).item())
            output[f"{metric}_ci_high"] = float(torch.quantile(bootstrap_means, 0.975).item())
        summary.append(output)
    return summary


def plot_behavior(summary: list[dict[str, Any]], output_path: Path) -> None:
    steps = [int(row["act_step"]) for row in summary]
    metrics = ["exact_accuracy", "cell_accuracy", "target_nll", "target_margin"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        means = [float(row[f"{metric}_mean"]) for row in summary]
        lows = [float(row[f"{metric}_ci_low"]) for row in summary]
        highs = [float(row[f"{metric}_ci_high"]) for row in summary]
        axis.plot(steps, means, marker="o", linewidth=1.5)
        axis.fill_between(steps, lows, highs, alpha=0.2)
        axis.set(title=metric.replace("_", " ").title(), xlabel="ACT step")
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, Any]], value_key: str, output_path: Path, title: str) -> None:
    groups = sorted({(str(row["level"]), int(row["occurrence_index"]), str(row["phase"])) for row in rows})
    columns = 3
    row_count = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(row_count, columns, figsize=(6 * columns, 4.5 * row_count), squeeze=False, constrained_layout=True)
    for axis, (level, occurrence_index, phase) in zip(axes.flat, groups):
        group_rows = [row for row in rows if row["level"] == level and int(row["occurrence_index"]) == occurrence_index]
        steps = sorted({int(row["act_step"]) for row in group_rows})
        blocks = sorted({int(row["block_index"]) for row in group_rows})
        values = torch.tensor(
            [[next(float(row[value_key]) for row in group_rows if int(row["act_step"]) == step and int(row["block_index"]) == block)
              for block in blocks] for step in steps]
        )
        image = axis.imshow(values.numpy(), aspect="auto", origin="lower")
        axis.set(title=f"{level} occurrence {occurrence_index} ({phase}): {title}", xlabel="Block", ylabel="ACT step")
        axis.set_xticks(range(len(blocks)), blocks)
        axis.set_yticks(range(len(steps)), steps)
        fig.colorbar(image, ax=axis, shrink=0.8)
    for axis in axes.flat[len(groups):]:
        axis.set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 1 requires a CUDA GPU for the trained HRM checkpoint.")

    os.environ.setdefault("DISABLE_COMPILE", "1")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    _train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    train_state = init_train_state(config, train_metadata, world_size=1)
    maybe_load_train_state(config, train_state)
    train_state.model.eval()

    model_core = train_state.model.model  # type: ignore[attr-defined]
    model_inner = model_core.inner
    prefix_len = int(getattr(model_inner, "prefix_len", getattr(model_inner, "puzzle_emb_len", 0)))
    state_dtype = STATE_DTYPES[args.state_dtype]

    behavior_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    target_store: list[torch.Tensor] = []
    info_store: dict[tuple[int, str, int, int, str], list[torch.Tensor]] = defaultdict(list)
    occurrence_metadata: dict[str, list[dict[str, Any]]] = {}
    set_offsets: dict[str, int] = defaultdict(int)
    example_count = 0
    shard_index = 0
    act_steps_seen = 0

    with torch.inference_mode():
        for set_name, fixed_batch, _global_batch_size in eval_loader:
            if example_count >= args.max_examples:
                break
            labels_all = fixed_batch["labels"]
            valid_rows = valid_example_rows(labels_all)
            remaining = args.max_examples - example_count
            valid_rows = valid_rows[:remaining]
            if valid_rows.numel() == 0:
                continue
            fixed_batch = {key: value[valid_rows] for key, value in fixed_batch.items()}
            batch = {key: value.cuda(non_blocking=True) for key, value in fixed_batch.items()}
            labels = batch["labels"].to(torch.int64)
            batch_size = labels.shape[0]
            example_ids = [f"{set_name}:{set_offsets[set_name] + i:06d}" for i in range(batch_size)]
            set_offsets[set_name] += batch_size

            with torch.device("cuda"):
                carry = model_core.initial_carry(batch)

            h_steps: list[torch.Tensor] = []
            l_steps: list[torch.Tensor] = []
            y, positions_with_prefix = target_vectors(model_inner, batch, labels)
            target_store.append(y.cpu())
            act_step = 0
            previous_z_h = None
            previous_z_l = None

            while True:
                act_step += 1
                carry, outputs = model_core(carry=carry, batch=batch, return_block_traces=True, return_z=True)
                h_boundaries, h_occurrences = occurrence_boundaries(outputs, "H")
                l_boundaries, l_occurrences = occurrence_boundaries(outputs, "L")
                for level, metadata in (("H", h_occurrences), ("L", l_occurrences)):
                    if level in occurrence_metadata and occurrence_metadata[level] != metadata:
                        raise RuntimeError(f"{level} occurrence metadata changed between ACT steps")
                    occurrence_metadata[level] = metadata

                behavior = per_example_behavior(outputs["logits"], labels)
                h_act_update = act_state_update_magnitude(outputs["z_H"], previous_z_h, labels, prefix_len)
                l_act_update = act_state_update_magnitude(outputs["z_L"], previous_z_l, labels, prefix_len)
                h_update = masked_update_magnitude(h_boundaries[:, :, :, prefix_len:], labels)
                l_update = masked_update_magnitude(l_boundaries[:, :, :, prefix_len:], labels)
                append_inforidge_vectors(info_store, "H", act_step, h_boundaries, positions_with_prefix)
                append_inforidge_vectors(info_store, "L", act_step, l_boundaries, positions_with_prefix)

                h_steps.append(h_boundaries[:, :, :, prefix_len:].to(dtype=state_dtype).cpu())
                l_steps.append(l_boundaries[:, :, :, prefix_len:].to(dtype=state_dtype).cpu())

                for row_index, example_id in enumerate(example_ids):
                    behavior_rows.append(
                        {
                            "example_id": example_id,
                            "set_name": set_name,
                            "act_step": act_step,
                            "cell_accuracy": float(behavior["cell_accuracy"][row_index].item()),
                            "exact_accuracy": int(behavior["exact_accuracy"][row_index].item()),
                            "target_nll": float(behavior["target_nll"][row_index].item()),
                            "target_margin": float(behavior["target_margin"][row_index].item()),
                            "q_halt_logit": float(outputs["q_halt_logits"][row_index].item()),
                            "q_continue_logit": float(outputs["q_continue_logits"][row_index].item()),
                            "h_act_update_l2": float(h_act_update[row_index].item()),
                            "l_act_update_l2": float(l_act_update[row_index].item()),
                        }
                    )
                    for level, magnitudes, metadata in (("H", h_update, h_occurrences), ("L", l_update, l_occurrences)):
                        for occurrence_index in range(magnitudes.shape[1]):
                            for block_index in range(magnitudes.shape[2]):
                                update_rows.append(
                                    {
                                        "example_id": example_id,
                                        "set_name": set_name,
                                        "act_step": act_step,
                                        "level": level,
                                        **metadata[occurrence_index],
                                        "block_index": block_index,
                                        "mean_token_update_l2": float(magnitudes[row_index, occurrence_index, block_index].item()),
                                    }
                                )

                if bool(carry.halted.all().item()):
                    break
                previous_z_h = outputs["z_H"]
                previous_z_l = outputs["z_L"]

            act_steps_seen = max(act_steps_seen, act_step)
            h_tensor = torch.stack(h_steps, dim=1)
            l_tensor = torch.stack(l_steps, dim=1)
            shard_path = activation_dir / f"batch_{shard_index:04d}.pt"
            torch.save(
                {
                    "format_version": 1,
                    "example_ids": example_ids,
                    "set_name": set_name,
                    "inputs": fixed_batch["inputs"].to(torch.int32),
                    "labels": fixed_batch["labels"].to(torch.int32),
                    "puzzle_identifiers": fixed_batch["puzzle_identifiers"].to(torch.int32),
                    "H_boundary_states": h_tensor,
                    "L_boundary_states": l_tensor,
                    "dimensions": ["example", "act_step", "occurrence", "boundary", "token", "hidden"],
                    "H_occurrences": occurrence_metadata["H"],
                    "L_occurrences": occurrence_metadata["L"],
                    "boundary_meaning": ["before_block_0"] + [f"after_block_{index}" for index in range(h_tensor.shape[3] - 1)],
                },
                shard_path,
            )

            for level, tensor in (("H", h_tensor), ("L", l_tensor)):
                occurrence_count = tensor.shape[2]
                block_count = tensor.shape[3] - 1
                for step in range(1, tensor.shape[1] + 1):
                    for occurrence_index in range(occurrence_count):
                        for block_index in range(block_count):
                            index_rows.append(
                                {
                                    "shard": str(shard_path.relative_to(output_dir)),
                                    "set_name": set_name,
                                    "first_example_id": example_ids[0],
                                    "last_example_id": example_ids[-1],
                                    "example_count": batch_size,
                                    "level": level,
                                    "act_step": step,
                                    **occurrence_metadata[level][occurrence_index],
                                    "block_index": block_index,
                                    "tensor_key": f"{level}_boundary_states",
                                    "before_boundary_index": block_index,
                                    "after_boundary_index": block_index + 1,
                                    "dtype": str(tensor.dtype).removeprefix("torch."),
                                    "token_prefix_removed": prefix_len,
                                }
                            )

            example_count += batch_size
            shard_index += 1
            print(f"Saved {shard_path} ({example_count}/{args.max_examples} examples)", flush=True)
            del h_tensor, l_tensor, h_steps, l_steps, carry, outputs, batch

    if example_count == 0:
        raise RuntimeError("No valid test examples were collected.")

    trajectories = analyze_solve_trajectory(behavior_rows)
    for row in behavior_rows:
        row.update(trajectories[str(row["example_id"])])

    info_rows = current_inforidge_rows(info_store, target_store, occurrence_metadata, sigma=args.kernel_sigma)
    behavior_summary = summarize_behavior(behavior_rows, args.bootstrap_samples, args.seed)
    mean_updates: list[dict[str, Any]] = []
    update_keys = sorted({(int(row["act_step"]), str(row["level"]), int(row["occurrence_index"]), int(row["block_index"])) for row in update_rows})
    for act_step, level, occurrence_index, block_index in update_keys:
        values = [float(row["mean_token_update_l2"]) for row in update_rows
                  if int(row["act_step"]) == act_step and row["level"] == level
                  and int(row["occurrence_index"]) == occurrence_index and int(row["block_index"]) == block_index]
        mean_updates.append({"act_step": act_step, "level": level, **occurrence_metadata[level][occurrence_index], "block_index": block_index,
                             "mean_token_update_l2": sum(values) / len(values)})

    write_csv(output_dir / "behavior_by_example_act.csv", behavior_rows)
    write_csv(output_dir / "behavior_summary_by_act.csv", behavior_summary)
    write_csv(output_dir / "block_updates_by_example_act.csv", update_rows)
    write_csv(output_dir / "activation_index.csv", index_rows)
    write_csv(output_dir / "current_inforidge_by_block_act.csv", info_rows)
    write_csv(output_dir / "mean_block_updates.csv", mean_updates)

    plot_behavior(behavior_summary, output_dir / "per_step_behavior.png")
    plot_heatmaps(mean_updates, "mean_token_update_l2", output_dir / "block_update_heatmaps.png", "Mean token update L2")
    plot_heatmaps(info_rows, "I_after_Y", output_dir / "current_inforidge_heatmaps.png", "Current I(after; Y)")

    bytes_per_value = torch.tensor([], dtype=state_dtype).element_size()
    boundaries_per_act = (
        len(occurrence_metadata["H"]) * (model_inner.config.H_layers + 1)
        + len(occurrence_metadata["L"]) * (model_inner.config.L_layers + 1)
    )
    estimated_state_bytes = example_count * act_steps_seen * boundaries_per_act * eval_metadata.seq_len * model_inner.config.hidden_size * bytes_per_value
    manifest = {
        "experiment": "clean_information_flow_baseline",
        "format_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "examples": example_count,
        "activation_shards": shard_index,
        "act_steps": act_steps_seen,
        "levels": ["H", "L"],
        "blocks_per_level": {"H": model_inner.config.H_layers, "L": model_inner.config.L_layers},
        "activation_dimensions": ["example", "act_step", "occurrence", "boundary", "token", "hidden"],
        "occurrences_per_act": {level: occurrence_metadata[level] for level in ("H", "L")},
        "boundary_storage": "boundary i is block i input; boundary i+1 is block i output",
        "puzzle_embedding_prefix_removed": prefix_len,
        "state_dtype": args.state_dtype,
        "estimated_activation_bytes": estimated_state_bytes,
        "behavior_metrics": ["target_nll", "target_margin", "cell_accuracy", "exact_accuracy"],
        "act_state_update_metrics": ["h_act_update_l2", "l_act_update_l2"],
        "bootstrap_samples": args.bootstrap_samples,
        "information_measure_status": "current estimator, not yet validated",
        "information_target_mode": "last_valid_token_second_pass",
        "kernel_sigma": args.kernel_sigma,
        "intervention": "none",
        "seed": args.seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Clean information-flow baseline complete: {output_dir}")


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
