#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch


CHARSET = "# SGo"
ID_TO_CHAR = {0: "·", **{idx + 1: ch for idx, ch in enumerate(CHARSET)}}
CHAR_TO_COLOR = {
    "#": "#1f1f1f",
    " ": "#f5f5f5",
    "S": "#2ca02c",
    "G": "#d62728",
    "o": "#1f77b4",
    "·": "#ffffff",
}
COLOR_ORDER = ["·", "#", " ", "S", "G", "o"]
VALUE_TO_COLOR_INDEX = {value: COLOR_ORDER.index(char) for value, char in ID_TO_CHAR.items() if char in COLOR_ORDER}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Maze test prediction from saved eval outputs.")
    parser.add_argument("--pred-file", required=True, help="Path to step_*_all_preds.* file produced by evaluate/pretrain.")
    parser.add_argument("--output", required=True, help="PNG output path.")
    parser.add_argument("--example-index", type=int, default=0, help="Index of the example to render.")
    parser.add_argument(
        "--mode",
        choices=["index", "first-failed", "first-correct"],
        default="index",
        help="How to choose the example to render.",
    )
    return parser.parse_args()


def _as_grid(seq: torch.Tensor) -> torch.Tensor:
    seq_len = int(seq.numel())
    side = int(math.isqrt(seq_len))
    if side * side != seq_len:
        raise ValueError(f"Sequence length {seq_len} is not a square grid.")
    return seq.view(side, side)


def _decode_grid(seq: torch.Tensor) -> list[list[str]]:
    grid = _as_grid(seq)
    decoded: list[list[str]] = []
    for row in grid.tolist():
        decoded.append([ID_TO_CHAR.get(int(value), "?") if int(value) >= 0 else "·" for value in row])
    return decoded


def _color_grid(decoded: list[list[str]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in decoded:
        rows.append([COLOR_ORDER.index(char if char in COLOR_ORDER else "·") for char in row])
    return rows


def _token_accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    valid = labels >= 0
    if not valid.any():
        return 0.0
    return float((preds[valid] == labels[valid]).to(torch.float32).mean().item())


def _exact_match(labels: torch.Tensor, preds: torch.Tensor) -> bool:
    valid = labels >= 0
    if not valid.any():
        return False
    return bool(torch.equal(preds[valid], labels[valid]))


def _select_index(labels: torch.Tensor, preds: torch.Tensor, mode: str, requested_index: int) -> int:
    if mode == "index":
        return requested_index

    for idx in range(labels.shape[0]):
        exact = _exact_match(labels[idx], preds[idx])
        if mode == "first-failed" and not exact:
            return idx
        if mode == "first-correct" and exact:
            return idx

    raise ValueError(f"No example found for mode={mode}.")


def _draw_panel(ax, title: str, seq: torch.Tensor) -> None:
    decoded = _decode_grid(seq)
    color_grid = _color_grid(decoded)
    cmap = ListedColormap([CHAR_TO_COLOR[ch] for ch in COLOR_ORDER])

    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=len(COLOR_ORDER) - 1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    for r, row in enumerate(decoded):
        for c, ch in enumerate(row):
            text_color = "#ffffff" if ch == "#" else "#111111"
            ax.text(c, r, ch if ch != " " else "·", ha="center", va="center", fontsize=8, color=text_color)


def main() -> None:
    args = parse_args()
    pred_file = Path(args.pred_file).resolve()
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {pred_file}")

    data = torch.load(pred_file, map_location="cpu")
    required = {"inputs", "labels", "logits"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Prediction file missing required keys: {sorted(missing)}")

    inputs = data["inputs"]
    labels = data["labels"]
    preds = torch.argmax(data["logits"], dim=-1)

    index = _select_index(labels, preds, args.mode, args.example_index)
    if index < 0 or index >= inputs.shape[0]:
        raise IndexError(f"example-index {index} out of range for {inputs.shape[0]} examples.")

    input_seq = inputs[index].to(torch.int64)
    label_seq = labels[index].to(torch.int64)
    pred_seq = preds[index].to(torch.int64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _draw_panel(axes[0], "Input Maze", input_seq)
    _draw_panel(axes[1], "Gold Path", torch.where(label_seq >= 0, label_seq, torch.zeros_like(label_seq)))
    _draw_panel(axes[2], "Predicted Path", pred_seq)

    token_acc = _token_accuracy(label_seq, pred_seq)
    exact = _exact_match(label_seq, pred_seq)
    fig.suptitle(f"Example {index} | token_acc={token_acc:.3f} | exact={exact}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
