from __future__ import annotations

import torch

from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy


def per_example_behavior(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return graded maze-solving metrics for each example in a batch."""
    valid = labels != IGNORE_LABEL_ID
    valid_count = valid.sum(dim=-1).clamp_min(1)
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels)).to(torch.long)

    predictions = torch.argmax(logits, dim=-1)
    correct = valid & (predictions == labels)
    cell_accuracy = correct.sum(dim=-1).to(torch.float32) / valid_count
    exact_accuracy = correct.sum(dim=-1) == valid_count

    losses = stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
    target_nll = losses.sum(dim=-1) / valid_count

    target_logits = torch.gather(logits.float(), dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    incorrect_logits = logits.float().clone()
    incorrect_logits.scatter_(dim=-1, index=safe_labels.unsqueeze(-1), value=float("-inf"))
    strongest_incorrect = incorrect_logits.max(dim=-1).values
    cell_margin = target_logits - strongest_incorrect
    target_margin = torch.where(valid, cell_margin, 0).sum(dim=-1) / valid_count

    return {
        "cell_accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "target_nll": target_nll,
        "target_margin": target_margin,
    }
