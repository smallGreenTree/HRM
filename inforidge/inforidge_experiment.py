from __future__ import annotations

import csv
import math
import os
from typing import Any, Optional, TYPE_CHECKING

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from inforidge.inforidge import (
    compute_layerwise_info_metrics,
    l2_normalize,
    matrix_mutual_information,
)

if TYPE_CHECKING:
    from pretrain import PretrainConfig, TrainState


def run_inforidge_analysis(
    config: "PretrainConfig",
    train_state: "TrainState",
    eval_loader: DataLoader,
    rank: int,
):
    if not config.inforidge_config.enabled or rank != 0:
        return None

    max_batches = max(1, config.inforidge_config.max_batches)
    max_tokens = max(1, config.inforidge_config.max_tokens)
    sample_mode = config.inforidge_config.sample_mode
    target_embedding_mode = config.inforidge_config.target_embedding_mode
    normalize_vectors = config.inforidge_config.normalize_vectors
    kernel_sigma = config.inforidge_config.kernel_sigma
    base_seed = config.inforidge_config.sample_seed + train_state.step
    include_h = config.inforidge_config.include_H
    include_l = config.inforidge_config.include_L

    h_predictive_sum = None
    h_incremental_sum = None
    h_weight = 0
    l_predictive_sum = None
    l_incremental_sum = None
    l_weight = 0
    batches = 0
    model_inner = train_state.model.model.inner  # type: ignore
    embed_tokens = model_inner.embed_tokens
    prefix_len = getattr(model_inner, "prefix_len", getattr(model_inner, "puzzle_emb_len", 0))

    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            if batches >= max_batches:
                break
            batches += 1

            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            while True:
                carry, _, _metrics, preds, all_finish = train_state.model(
                    carry=carry,
                    batch=batch,
                    return_keys=["inforidge"],
                )
                if all_finish:
                    break

            info = preds.get("inforidge") if preds is not None else None
            if info is None:
                continue

            labels = info.get("labels")
            if labels is None:
                continue

            if include_h and info.get("layer_states_H") is not None:
                h_metrics = compute_layerwise_info_metrics(
                    info["layer_states_H"],
                    labels,
                    embed_tokens,
                    prefix_len=prefix_len,
                    max_tokens=max_tokens,
                    sample_seed=base_seed + batches,
                    sigma=kernel_sigma,
                    sample_mode=sample_mode,
                    target_embedding_mode=target_embedding_mode,
                    inputs=batch.get("inputs"),
                    puzzle_identifiers=batch.get("puzzle_identifiers"),
                    input_embeddings_fn=getattr(model_inner, "_input_embeddings", None),
                    normalize_vectors=normalize_vectors,
                )
                if h_metrics is not None:
                    h_pred = h_metrics.predictive_mi.detach().cpu() * h_metrics.sample_count
                    h_inc = h_metrics.incremental_mi.detach().cpu() * h_metrics.sample_count
                    h_predictive_sum = h_pred if h_predictive_sum is None else h_predictive_sum + h_pred
                    h_incremental_sum = h_inc if h_incremental_sum is None else h_incremental_sum + h_inc
                    h_weight += h_metrics.sample_count

            if include_l and info.get("layer_states_L") is not None:
                l_metrics = compute_layerwise_info_metrics(
                    info["layer_states_L"],
                    labels,
                    embed_tokens,
                    prefix_len=prefix_len,
                    max_tokens=max_tokens,
                    sample_seed=base_seed + batches,
                    sigma=kernel_sigma,
                    sample_mode=sample_mode,
                    target_embedding_mode=target_embedding_mode,
                    inputs=batch.get("inputs"),
                    puzzle_identifiers=batch.get("puzzle_identifiers"),
                    input_embeddings_fn=getattr(model_inner, "_input_embeddings", None),
                    normalize_vectors=normalize_vectors,
                )
                if l_metrics is not None:
                    l_pred = l_metrics.predictive_mi.detach().cpu() * l_metrics.sample_count
                    l_inc = l_metrics.incremental_mi.detach().cpu() * l_metrics.sample_count
                    l_predictive_sum = l_pred if l_predictive_sum is None else l_predictive_sum + l_pred
                    l_incremental_sum = l_inc if l_incremental_sum is None else l_incremental_sum + l_inc
                    l_weight += l_metrics.sample_count

    if (h_weight + l_weight) == 0:
        return None

    return {
        "H_predictive_sum": h_predictive_sum,
        "H_incremental_sum": h_incremental_sum,
        "H_weight": h_weight,
        "L_predictive_sum": l_predictive_sum,
        "L_incremental_sum": l_incremental_sum,
        "L_weight": l_weight,
    }


def log_inforidge_results(
    config: "PretrainConfig",
    train_state: "TrainState",
    results: Optional[dict[str, Any]],
):
    if results is None:
        return

    rows = []
    if results["H_predictive_sum"] is not None and results["H_weight"] > 0:
        h_predictive = (results["H_predictive_sum"] / results["H_weight"]).tolist()
        h_incremental = (results["H_incremental_sum"] / results["H_weight"]).tolist()
        for i, mi in enumerate(h_predictive):
            dmi = h_incremental[i]
            rows.append(("H", i, float(mi), (None if math.isnan(dmi) else float(dmi))))
    if results["L_predictive_sum"] is not None and results["L_weight"] > 0:
        l_predictive = (results["L_predictive_sum"] / results["L_weight"]).tolist()
        l_incremental = (results["L_incremental_sum"] / results["L_weight"]).tolist()
        for i, mi in enumerate(l_predictive):
            dmi = l_incremental[i]
            rows.append(("L", i, float(mi), (None if math.isnan(dmi) else float(dmi))))

    if config.inforidge_config.save_csv and config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        csv_path = os.path.join(config.checkpoint_path, f"inforidge_step_{train_state.step}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_type", "layer_index", "I_Z_Y", "I_dZ_Y"])
            for layer_type, layer_index, mi, dmi in rows:
                writer.writerow([layer_type, layer_index, mi, dmi if dmi is not None else ""])

    if config.inforidge_config.log_wandb and wandb.run is not None:
        for layer_type, layer_index, mi, dmi in rows:
            wandb.log({f"inforidge/{layer_type}/I_Z_Y_layer_{layer_index}": mi}, step=train_state.step)
            if dmi is not None:
                wandb.log({f"inforidge/{layer_type}/I_dZ_Y_layer_{layer_index}": dmi}, step=train_state.step)


def _pool_valid(z: torch.Tensor, labels: torch.Tensor, prefix_len: int) -> torch.Tensor:
    if prefix_len > 0:
        z = z[:, prefix_len:]
        labels = labels[:, : z.shape[1]]
    mask = labels != -100
    mask_f = mask.to(z.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pooled = (z * mask_f).sum(dim=1) / denom
    return pooled


def _pool_label_emb(labels: torch.Tensor, embed_tokens: nn.Module, prefix_len: int) -> torch.Tensor:
    if prefix_len > 0:
        labels = labels[:, : -prefix_len]
    mask = labels != -100
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    emb = embed_tokens(safe_labels.to(torch.int32))
    mask_f = mask.to(emb.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pooled = (emb * mask_f).sum(dim=1) / denom
    return pooled


def run_inforidge_act_mi(
    config: "PretrainConfig",
    train_state: "TrainState",
    eval_loader: DataLoader,
    rank: int,
) -> Optional[list[tuple[int, float, Optional[float]]]]:
    if not config.inforidge_act_mi.enabled or rank != 0:
        return None

    fixed_batch = None
    for _set_name, batch, _global_batch_size in eval_loader:
        fixed_batch = batch
        break

    if fixed_batch is None:
        return None

    with torch.inference_mode():
        batch = {k: v.cuda() for k, v in fixed_batch.items()}
        with torch.device("cuda"):
            carry = train_state.model.initial_carry(batch)  # type: ignore

        zh_steps = []
        labels = batch["labels"]
        prefix_len = getattr(train_state.model.model.inner, "prefix_len", getattr(train_state.model.model.inner, "puzzle_emb_len", 0))  # type: ignore
        embed_tokens = train_state.model.model.inner.embed_tokens  # type: ignore

        while True:
            carry, _, _metrics, preds, all_finish = train_state.model(
                carry=carry,
                batch=batch,
                return_keys=["inforidge_act_mi"],
            )
            if preds is None or "inforidge_act_mi" not in preds:
                return None
            zh_steps.append(preds["inforidge_act_mi"]["z_H"])
            if all_finish:
                break

        y = _pool_label_emb(labels, embed_tokens, prefix_len)
        y = l2_normalize(y)

        results = []
        prev = None
        for step_idx, z in enumerate(zh_steps, start=1):
            z_pooled = _pool_valid(z, labels, prefix_len)
            z_pooled = l2_normalize(z_pooled)
            mi = matrix_mutual_information(z_pooled, y).detach().cpu().item()
            if prev is None:
                dmi = None
            else:
                dz = z_pooled - prev
                dz = l2_normalize(dz)
                dmi = matrix_mutual_information(dz, y).detach().cpu().item()
            prev = z_pooled
            results.append((step_idx, mi, dmi))

    return results


def log_inforidge_act_mi(
    config: "PretrainConfig",
    train_state: "TrainState",
    results: Optional[list[tuple[int, float, Optional[float]]]],
):
    if results is None:
        return

    if config.inforidge_act_mi.save_csv and config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        csv_path = os.path.join(config.checkpoint_path, f"inforidge_act_mi_step_{train_state.step}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["act_step", "I_Z_Y", "I_dZ_Y"])
            for step_idx, mi, dmi in results:
                writer.writerow([step_idx, mi, dmi if dmi is not None else ""])

    if config.inforidge_act_mi.log_wandb and wandb.run is not None:
        for step_idx, mi, dmi in results:
            wandb.log({f"inforidge_act_mi/I_Z_Y_step_{step_idx}": mi}, step=train_state.step)
            if dmi is not None:
                wandb.log({f"inforidge_act_mi/I_dZ_Y_step_{step_idx}": dmi}, step=train_state.step)
