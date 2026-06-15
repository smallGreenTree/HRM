#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
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
import torch.nn.functional as F
from omegaconf import OmegaConf

from inforidge.behavior import per_example_behavior
from pretrain import PretrainConfig, create_dataloader, init_train_state, maybe_load_train_state


IGNORE_LABEL_ID = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HRM H/L hidden-state dynamics across ACT steps.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint .pt file.")
    parser.add_argument("--data-path", default="data/maze-30x30-hard-1k", help="Dataset path.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSVs and plots.")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="hidden-dynamics")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PretrainConfig:
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(
            config_name="cfg_pretrain",
            overrides=[
                f"data_path={args.data_path}",
                f"global_batch_size={args.global_batch_size}",
                f"load_checkpoint_path={args.checkpoint}",
                "analysis_only=true",
                f"+seed={args.seed}",
                "project_name=hidden-dynamics",
                f"run_name={args.run_name}",
                "checkpoint_path=null",
            ],
        )
    return PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def pool_valid_tokens(z: torch.Tensor, labels: torch.Tensor, prefix_len: int) -> torch.Tensor:
    if prefix_len > 0:
        z = z[:, prefix_len:]
    labels = labels[:, : z.shape[1]]
    mask = labels != IGNORE_LABEL_ID
    mask_f = mask.to(z.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (z * mask_f).sum(dim=1) / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def tensor_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.std(unbiased=False).item())


def collect_rows(config: PretrainConfig, max_batches: int) -> list[dict[str, Any]]:
    os.environ.setdefault("DISABLE_COMPILE", "1")
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    eval_loader, _eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    train_state = init_train_state(config, train_metadata, world_size=1)
    maybe_load_train_state(config, train_state)
    train_state.model.eval()

    model_core = train_state.model.model  # type: ignore[attr-defined]
    prefix_len = getattr(model_core.inner, "prefix_len", getattr(model_core.inner, "puzzle_emb_len", 0))

    rows: list[dict[str, Any]] = []
    example_offset = 0
    batches = 0

    with torch.inference_mode():
        for _set_name, fixed_batch, _global_batch_size in eval_loader:
            if batches >= max_batches:
                break
            batches += 1
            batch = {k: v.cuda() for k, v in fixed_batch.items()}
            labels = batch["labels"].to(torch.int64)
            with torch.device("cuda"):
                carry = model_core.initial_carry(batch)

            step_records = []
            while True:
                carry, outputs = model_core(carry=carry, batch=batch, return_z=True)
                z_h = outputs["z_H"].detach()
                z_l = outputs["z_L"].detach()
                logits = outputs["logits"].detach()
                behavior = per_example_behavior(logits, labels)
                step_records.append(
                    (
                        z_h,
                        z_l,
                        behavior,
                        outputs["q_halt_logits"].detach(),
                        outputs["q_continue_logits"].detach(),
                    )
                )
                if bool(carry.halted.all().item()):
                    break

            exact_by_step = torch.stack([record[2]["exact_accuracy"] for record in step_records])
            success = exact_by_step[-1]
            first_solved_step = torch.zeros(labels.shape[0], dtype=torch.int64, device=labels.device)
            regressed_after_solve = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            for i in range(labels.shape[0]):
                solved_steps = torch.nonzero(exact_by_step[:, i], as_tuple=False).squeeze(-1)
                if solved_steps.numel() == 0:
                    continue
                first_idx = int(solved_steps[0].item())
                first_solved_step[i] = first_idx + 1
                regressed_after_solve[i] = bool((~exact_by_step[first_idx:, i]).any().item())

            prev_h = None
            prev_l = None
            prev_dh = None
            prev_dl = None
            for step_idx, (z_h, z_l, behavior, q_halt, q_continue) in enumerate(step_records, start=1):
                h = pool_valid_tokens(z_h, labels, prefix_len)
                l = pool_valid_tokens(z_l, labels, prefix_len)
                h_norm = h.float().norm(dim=-1)
                l_norm = l.float().norm(dim=-1)
                h_l_distance = (h.float() - l.float()).norm(dim=-1)
                h_l_cosine = cosine(h, l)

                if prev_h is None:
                    h_delta = torch.zeros_like(h_norm)
                    l_delta = torch.zeros_like(l_norm)
                    h_step_cosine = torch.ones_like(h_norm)
                    l_step_cosine = torch.ones_like(l_norm)
                    update_cosine = torch.zeros_like(h_norm)
                else:
                    dh = h.float() - prev_h.float()
                    dl = l.float() - prev_l.float()
                    h_delta = dh.norm(dim=-1)
                    l_delta = dl.norm(dim=-1)
                    h_step_cosine = cosine(h, prev_h)
                    l_step_cosine = cosine(l, prev_l)
                    update_cosine = cosine(dh, dl)
                    prev_dh = dh
                    prev_dl = dl

                if prev_dh is None or prev_dl is None:
                    update_alignment_to_previous = torch.zeros_like(h_norm)
                else:
                    update_alignment_to_previous = cosine(prev_dh, prev_dl)

                for i in range(labels.shape[0]):
                    rows.append(
                        {
                            "batch_index": batches - 1,
                            "example_index": example_offset + i,
                            "act_step": step_idx,
                            "success": int(bool(success[i].item())),
                            "cell_accuracy": float(behavior["cell_accuracy"][i].item()),
                            "exact_accuracy": int(bool(behavior["exact_accuracy"][i].item())),
                            "target_nll": float(behavior["target_nll"][i].item()),
                            "target_margin": float(behavior["target_margin"][i].item()),
                            "q_halt_logit": float(q_halt[i].item()),
                            "q_continue_logit": float(q_continue[i].item()),
                            "first_solved_step": int(first_solved_step[i].item()),
                            "is_first_solved_step": int(first_solved_step[i].item() == step_idx),
                            "regressed_after_solve": int(bool(regressed_after_solve[i].item())),
                            "h_norm": float(h_norm[i].item()),
                            "l_norm": float(l_norm[i].item()),
                            "h_delta": float(h_delta[i].item()),
                            "l_delta": float(l_delta[i].item()),
                            "h_step_cosine": float(h_step_cosine[i].item()),
                            "l_step_cosine": float(l_step_cosine[i].item()),
                            "h_l_cosine": float(h_l_cosine[i].item()),
                            "h_l_distance": float(h_l_distance[i].item()),
                            "update_cosine": float(update_cosine[i].item()),
                            "update_alignment_to_previous": float(update_alignment_to_previous[i].item()),
                        }
                    )

                prev_h = h.detach()
                prev_l = l.detach()

            example_offset += labels.shape[0]

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "cell_accuracy",
        "exact_accuracy",
        "target_nll",
        "target_margin",
        "q_halt_logit",
        "q_continue_logit",
        "h_norm",
        "l_norm",
        "h_delta",
        "l_delta",
        "h_step_cosine",
        "l_step_cosine",
        "h_l_cosine",
        "h_l_distance",
        "update_cosine",
    ]
    groups = [("all", None), ("success", 1), ("failure", 0)]
    summary = []
    steps = sorted({int(row["act_step"]) for row in rows})
    for step in steps:
        step_rows = [row for row in rows if int(row["act_step"]) == step]
        for group_name, success_value in groups:
            group_rows = step_rows if success_value is None else [row for row in step_rows if int(row["success"]) == success_value]
            if not group_rows:
                continue
            out: dict[str, Any] = {"act_step": step, "group": group_name, "count": len(group_rows)}
            out["first_solved_count"] = sum(int(row["is_first_solved_step"]) for row in group_rows)
            out["regressed_after_solve_count"] = sum(int(row["regressed_after_solve"]) for row in group_rows)
            for metric in metrics:
                values = [float(row[metric]) for row in group_rows]
                out[f"{metric}_mean"] = sum(values) / len(values)
                out[f"{metric}_std"] = tensor_std(values)
            summary.append(out)
    return summary


def plot_summary(summary: list[dict[str, Any]], output_path: Path) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_group.setdefault(str(row["group"]), []).append(row)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panels = [
        ("target_nll_mean", "Target NLL"),
        ("exact_accuracy_mean", "Exact Accuracy"),
        ("h_delta_mean", "H Update Magnitude"),
        ("l_delta_mean", "L Update Magnitude"),
        ("h_l_cosine_mean", "H/L Cosine Similarity"),
        ("h_l_distance_mean", "H/L Distance"),
    ]
    colors = {"all": "tab:blue", "success": "tab:green", "failure": "tab:red"}
    for ax, (metric, title) in zip(axes.flat, panels):
        for group, rows in by_group.items():
            rows = sorted(rows, key=lambda r: int(r["act_step"]))
            ax.plot(
                [int(r["act_step"]) for r in rows],
                [float(r[metric]) for r in rows],
                marker="o",
                linewidth=2,
                label=group,
                color=colors.get(group),
            )
        ax.set_title(title)
        ax.set_xlabel("ACT step")
        ax.grid(alpha=0.3)
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    config = build_config(args)
    rows = collect_rows(config, max_batches=args.max_batches)
    if not rows:
        raise RuntimeError("No dynamics rows collected.")

    summary_rows = summarize(rows)
    write_csv(output_dir / "hidden_dynamics_examples.csv", rows)
    write_csv(output_dir / "hidden_dynamics_summary.csv", summary_rows)
    plot_summary(summary_rows, output_dir / "hidden_dynamics_summary.png")

    success_count = len({row["example_index"] for row in rows if int(row["success"]) == 1})
    failure_count = len({row["example_index"] for row in rows if int(row["success"]) == 0})
    print(f"Wrote {output_dir}")
    print(f"success_examples={success_count} failure_examples={failure_count}")


if __name__ == "__main__":
    main()
