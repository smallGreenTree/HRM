from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from torch import nn


@dataclass
class LayerwiseInfoMetrics:
    predictive_mi: torch.Tensor
    incremental_mi: torch.Tensor
    sample_count: int


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def gram_matrix(u: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Matrix entropy relies on eigendecomposition, which is not implemented for
    # bfloat16 on CUDA. Upcast once here so all MI callers share the stable path.
    u = u.to(torch.float64)
    diff = u.unsqueeze(1) - u.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)
    g = torch.exp(-dist2 / (2 * sigma * sigma))
    return g / (torch.trace(g) + 1e-12)


def matrix_entropy(g: torch.Tensor) -> torch.Tensor:
    eigvals = torch.linalg.eigvalsh(g)
    eigvals = torch.clamp(eigvals.real, min=1e-12)
    return -torch.sum(eigvals * torch.log(eigvals))


def matrix_mutual_information(u: torch.Tensor, v: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    gu = gram_matrix(u, sigma=sigma)
    gv = gram_matrix(v, sigma=sigma)
    h_u = matrix_entropy(gu)
    h_v = matrix_entropy(gv)
    g_uv = gu * gv
    g_uv = g_uv / (torch.trace(g_uv) + 1e-12)
    h_uv = matrix_entropy(g_uv)
    return h_u + h_v - h_uv


def _subsample_indices(indices: torch.Tensor, max_tokens: Optional[int], seed: int) -> torch.Tensor:
    if max_tokens is None or indices.numel() <= max_tokens:
        return indices

    g = torch.Generator(device=indices.device)
    g.manual_seed(seed)
    perm = torch.randperm(indices.numel(), generator=g, device=indices.device)
    return indices[perm[:max_tokens]]


def _last_valid_positions(labels: torch.Tensor, ignore_label_id: int) -> torch.Tensor:
    valid = labels != ignore_label_id
    rev_idx = torch.argmax(torch.flip(valid, dims=[1]).to(torch.int64), dim=1)
    return labels.shape[1] - 1 - rev_idx


def compute_layerwise_info_metrics(
    layer_states: List[torch.Tensor],
    labels: torch.Tensor,
    embed_tokens: nn.Module,
    *,
    prefix_len: int = 0,
    puzzle_emb_len: Optional[int] = None,
    ignore_label_id: int = -100,
    max_tokens: Optional[int] = 256,
    sample_seed: int = 0,
    sigma: float = 1.0,
    sample_mode: str = "last_token",
    target_embedding_mode: str = "lookup",
    inputs: Optional[torch.Tensor] = None,
    puzzle_identifiers: Optional[torch.Tensor] = None,
    input_embeddings_fn: Optional[Callable[..., torch.Tensor]] = None,
    normalize_vectors: bool = True,
) -> Optional[LayerwiseInfoMetrics]:
    if len(layer_states) == 0:
        return None

    # Backward-compatible alias for older call sites.
    if puzzle_emb_len is not None:
        prefix_len = puzzle_emb_len

    if target_embedding_mode not in {"lookup", "second_pass"}:
        raise ValueError(f"Unsupported target_embedding_mode: {target_embedding_mode}")

    def build_target_embeddings(rows: torch.Tensor, cols: torch.Tensor, y_tokens: torch.Tensor) -> torch.Tensor:
        if target_embedding_mode == "lookup":
            y = embed_tokens(y_tokens.to(torch.int32)).to(torch.float64)
            if normalize_vectors:
                y = l2_normalize(y)
            return y

        if inputs is None or input_embeddings_fn is None:
            raise ValueError("target_embedding_mode='second_pass' requires `inputs` and `input_embeddings_fn`.")

        second_inputs = inputs.clone()
        second_inputs[rows, cols] = y_tokens.to(second_inputs.dtype)

        if puzzle_identifiers is None:
            emb = input_embeddings_fn(second_inputs.to(torch.int32))  # type: ignore[misc]
        else:
            emb = input_embeddings_fn(second_inputs.to(torch.int32), puzzle_identifiers)  # type: ignore[misc]

        y = emb[rows, cols + prefix_len].to(torch.float64)
        if normalize_vectors:
            y = l2_normalize(y)
        return y

    seq_len = layer_states[0].shape[1] - max(prefix_len, 0)
    trimmed_labels = labels[:, :seq_len]
    if sample_mode == "all_tokens":
        valid_mask = (trimmed_labels != ignore_label_id).reshape(-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_indices = _subsample_indices(valid_indices, max_tokens=max_tokens, seed=sample_seed)
        if valid_indices.numel() == 0:
            return None

        rows = valid_indices // seq_len
        cols = valid_indices % seq_len
        flat_labels = trimmed_labels.reshape(-1)[valid_indices]
        y = build_target_embeddings(rows, cols, flat_labels)

        def gather_layer(layer: torch.Tensor) -> torch.Tensor:
            z = layer[:, prefix_len: prefix_len + seq_len]
            z = z.reshape(-1, z.shape[-1])[valid_indices].to(torch.float64)
            if normalize_vectors:
                z = l2_normalize(z)
            return z

        sample_count = int(valid_indices.numel())
    elif sample_mode == "last_token":
        valid = trimmed_labels != ignore_label_id
        has_valid = valid.any(dim=1)
        valid_rows = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
        valid_rows = _subsample_indices(valid_rows, max_tokens=max_tokens, seed=sample_seed)
        if valid_rows.numel() == 0:
            return None

        all_last_pos = _last_valid_positions(trimmed_labels, ignore_label_id=ignore_label_id)
        last_pos = all_last_pos[valid_rows]
        y_tokens = trimmed_labels[valid_rows, last_pos]
        y = build_target_embeddings(valid_rows, last_pos, y_tokens)

        def gather_layer(layer: torch.Tensor) -> torch.Tensor:
            z = layer[:, prefix_len: prefix_len + seq_len]
            z = z[valid_rows, last_pos].to(torch.float64)
            if normalize_vectors:
                z = l2_normalize(z)
            return z

        sample_count = int(valid_rows.numel())
    else:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    predictive_vals = []
    incremental_vals = []
    prev_z = None
    for layer in layer_states:
        z = gather_layer(layer)

        predictive_vals.append(matrix_mutual_information(z, y, sigma=sigma))

        if prev_z is None:
            incremental_vals.append(torch.tensor(float("nan"), dtype=torch.float64, device=z.device))
        else:
            dz = z - prev_z
            if normalize_vectors:
                dz = l2_normalize(dz)
            incremental_vals.append(matrix_mutual_information(dz, y, sigma=sigma))
        prev_z = z

    return LayerwiseInfoMetrics(
        predictive_mi=torch.stack(predictive_vals),
        incremental_mi=torch.stack(incremental_vals),
        sample_count=sample_count,
    )
