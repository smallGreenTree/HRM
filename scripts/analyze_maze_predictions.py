#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


CHARSET = "# SGo"
ID_TO_CHAR = {0: "PAD", **{idx + 1: ch for idx, ch in enumerate(CHARSET)}}
CHAR_TO_ID = {ch: idx for idx, ch in ID_TO_CHAR.items()}
PATH_ID = CHAR_TO_ID["o"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved maze predictions from HRM evaluation outputs.")
    parser.add_argument(
        "--pred-file",
        required=True,
        help="Path to step_*_all_preds.* file produced by evaluate/pretrain.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to save aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--examples-csv",
        help="Optional path to save per-example metrics as CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst examples to print, ranked by path F1 then token accuracy.",
    )
    return parser.parse_args()


def _safe_ratio(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom


def _tensor_bool_count(x: torch.Tensor) -> int:
    return int(x.to(torch.int64).sum().item())


def _build_example_rows(labels: torch.Tensor, preds: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(labels.shape[0]):
        gold = labels[idx]
        pred = preds[idx]

        valid = gold >= 0
        gold_path = gold == PATH_ID
        pred_path = pred == PATH_ID
        valid_gold_path = valid & gold_path
        valid_non_path = valid & (~gold_path)

        correct = valid & (gold == pred)
        tp = _tensor_bool_count(valid_gold_path & pred_path)
        fp = _tensor_bool_count(valid_non_path & pred_path)
        fn = _tensor_bool_count(valid_gold_path & (~pred_path))

        token_acc = _safe_ratio(_tensor_bool_count(correct), _tensor_bool_count(valid))
        path_precision = _safe_ratio(tp, tp + fp)
        path_recall = _safe_ratio(tp, tp + fn)
        path_f1 = _safe_ratio(2 * path_precision * path_recall, path_precision + path_recall)
        path_iou = _safe_ratio(tp, tp + fp + fn)

        row = {
            "example_index": idx,
            "token_accuracy": token_acc,
            "exact_match": int(bool(torch.equal(pred[valid], gold[valid]))),
            "valid_cells": _tensor_bool_count(valid),
            "total_errors": _tensor_bool_count(valid & (gold != pred)),
            "gold_path_cells": _tensor_bool_count(valid_gold_path),
            "pred_path_cells": _tensor_bool_count(valid & pred_path),
            "path_true_positive": tp,
            "path_false_positive": fp,
            "path_false_negative": fn,
            "path_precision": path_precision,
            "path_recall": path_recall,
            "path_f1": path_f1,
            "path_iou": path_iou,
            "path_exact": int(bool(torch.equal(pred_path[valid], gold_path[valid]))),
        }
        rows.append(row)
    return rows


def _aggregate_summary(labels: torch.Tensor, preds: torch.Tensor, example_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = labels >= 0
    correct = valid & (labels == preds)
    gold_path = valid & (labels == PATH_ID)
    pred_path = valid & (preds == PATH_ID)
    non_path = valid & (labels != PATH_ID)

    tp = _tensor_bool_count(gold_path & pred_path)
    fp = _tensor_bool_count(non_path & pred_path)
    fn = _tensor_bool_count(gold_path & (~pred_path))

    exact_count = sum(int(row["exact_match"]) for row in example_rows)
    path_exact_count = sum(int(row["path_exact"]) for row in example_rows)

    per_label: dict[str, Any] = {}
    for value, char in ID_TO_CHAR.items():
        if value == 0:
            continue
        mask = valid & (labels == value)
        support = _tensor_bool_count(mask)
        if support == 0:
            continue
        per_label[char] = {
            "support": support,
            "accuracy": _safe_ratio(_tensor_bool_count(mask & (preds == value)), support),
        }

    summary = {
        "num_examples": labels.shape[0],
        "seq_len": labels.shape[1],
        "token_accuracy": _safe_ratio(_tensor_bool_count(correct), _tensor_bool_count(valid)),
        "exact_accuracy": _safe_ratio(exact_count, len(example_rows)),
        "path_exact_accuracy": _safe_ratio(path_exact_count, len(example_rows)),
        "avg_errors_per_example": _safe_ratio(_tensor_bool_count(valid & (labels != preds)), len(example_rows)),
        "path_cell_rate": _safe_ratio(_tensor_bool_count(gold_path), _tensor_bool_count(valid)),
        "path_precision": _safe_ratio(tp, tp + fp),
        "path_recall": _safe_ratio(tp, tp + fn),
        "path_f1": _safe_ratio(2 * _safe_ratio(tp, tp + fp) * _safe_ratio(tp, tp + fn), _safe_ratio(tp, tp + fp) + _safe_ratio(tp, tp + fn)),
        "path_iou": _safe_ratio(tp, tp + fp + fn),
        "non_path_false_positive_rate": _safe_ratio(_tensor_bool_count(non_path & pred_path), _tensor_bool_count(non_path)),
        "per_label_accuracy": per_label,
    }
    return summary


def _write_examples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary: dict[str, Any], worst_rows: list[dict[str, Any]]) -> None:
    print("Aggregate metrics")
    print(f"  examples: {summary['num_examples']}")
    print(f"  seq_len: {summary['seq_len']}")
    print(f"  token_accuracy: {summary['token_accuracy']:.6f}")
    print(f"  exact_accuracy: {summary['exact_accuracy']:.6f}")
    print(f"  path_exact_accuracy: {summary['path_exact_accuracy']:.6f}")
    print(f"  avg_errors_per_example: {summary['avg_errors_per_example']:.3f}")
    print(f"  path_cell_rate: {summary['path_cell_rate']:.6f}")
    print(f"  path_precision: {summary['path_precision']:.6f}")
    print(f"  path_recall: {summary['path_recall']:.6f}")
    print(f"  path_f1: {summary['path_f1']:.6f}")
    print(f"  path_iou: {summary['path_iou']:.6f}")
    print(f"  non_path_false_positive_rate: {summary['non_path_false_positive_rate']:.6f}")

    print("\nPer-label accuracy")
    for char, metrics in summary["per_label_accuracy"].items():
        print(f"  {char!r}: support={metrics['support']} accuracy={metrics['accuracy']:.6f}")

    print("\nWorst examples")
    for row in worst_rows:
        print(
            "  "
            f"idx={row['example_index']} "
            f"exact={row['exact_match']} "
            f"token_acc={row['token_accuracy']:.4f} "
            f"path_f1={row['path_f1']:.4f} "
            f"errors={row['total_errors']} "
            f"gold_path={row['gold_path_cells']} "
            f"pred_path={row['pred_path_cells']}"
        )


def main() -> None:
    args = parse_args()
    pred_file = Path(args.pred_file).resolve()
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {pred_file}")

    data = torch.load(pred_file, map_location="cpu")
    required = {"labels", "logits"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Prediction file missing required keys: {sorted(missing)}")

    labels = data["labels"].to(torch.int64)
    logits = data["logits"]
    preds = torch.argmax(logits, dim=-1).to(torch.int64)

    example_rows = _build_example_rows(labels, preds)
    summary = _aggregate_summary(labels, preds, example_rows)

    worst_rows = sorted(
        example_rows,
        key=lambda row: (row["path_f1"], row["token_accuracy"], -row["total_errors"]),
    )[: max(args.top_k, 0)]
    _print_summary(summary, worst_rows)

    if args.summary_json:
        output_path = Path(args.summary_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nWrote {output_path}")

    if args.examples_csv:
        output_path = Path(args.examples_csv).resolve()
        _write_examples_csv(output_path, example_rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
