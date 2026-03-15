import unittest
from pathlib import Path

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from models.losses import ACTLossHead
from inforidge.inforidge import (
    compute_layerwise_info_metrics,
    matrix_mutual_information,
)


def _median_heuristic_sigma(x: torch.Tensor) -> float:
    distances = torch.cdist(x, x)
    nonzero = distances[distances > 0]
    if nonzero.numel() == 0:
        return 1.0
    return float(nonzero.median().item())


def _cross_kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)
    return torch.exp(-dist2 / (2 * sigma * sigma))


def _summary_stats(x: torch.Tensor) -> dict:
    return {
        "min": float(x.min().item()),
        "p10": float(torch.quantile(x, 0.10).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


class InfoRidgeMathTests(unittest.TestCase):
    def test_matrix_mutual_information_tracks_alignment(self):
        torch.manual_seed(7)
        u = 3.7 * torch.randn(128, 8, dtype=torch.float64)
        v_aligned = u + 0.1 * torch.randn_like(u)
        sigma = _median_heuristic_sigma(torch.cat([u, v_aligned], dim=0))

        mi_aligned = matrix_mutual_information(u, v_aligned, sigma=sigma).item()
        shuffled_mis = []
        for _ in range(5):
            v_shuffled = v_aligned[torch.randperm(v_aligned.shape[0])]
            shuffled_mis.append(matrix_mutual_information(u, v_shuffled, sigma=sigma).item())

        self.assertGreater(mi_aligned, max(shuffled_mis))

    def test_matrix_mutual_information_visual_debug(self):

        torch.manual_seed(7)
        u = 3 * torch.randn(128, 8, dtype=torch.float64)
        v_aligned = u + 0.1 * torch.randn_like(u)
        v_shuffled = v_aligned[torch.randperm(v_aligned.shape[0])]
        sigma = _median_heuristic_sigma(torch.cat([u, v_aligned], dim=0))
        print(sigma)
        mi_aligned = matrix_mutual_information(u, v_aligned, sigma=sigma).item()
        mi_shuffled = matrix_mutual_information(u, v_shuffled, sigma=sigma).item()
        k_aligned = _cross_kernel(u, v_aligned, sigma=sigma)
        k_shuffled = _cross_kernel(u, v_shuffled, sigma=sigma)
        d_aligned = torch.norm(u - v_aligned, dim=1)
        d_shuffled = torch.norm(u - v_shuffled, dim=1)

        # PCA to 2D via SVD for visual intuition.
        x = torch.cat([u, v_aligned, v_shuffled], dim=0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
        x2 = x_centered @ vh[:2].T
        n = u.shape[0]
        u2, v2 = x2[:n], x2[n:2 * n]

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].scatter(u2[:, 0], u2[:, 1], s=10, alpha=0.8, label="u")
        ax[0, 0].scatter(v2[:, 0], v2[:, 1], s=10, alpha=0.7, label="v_aligned")
        for i in range(20):
            ax[0, 0].plot([u2[i, 0], v2[i, 0]], [u2[i, 1], v2[i, 1]], "k-", alpha=0.2, lw=0.8)
        ax[0, 0].set_title("PCA view of aligned pairs")
        ax[0, 0].legend()

        ax[0, 1].imshow(k_aligned.cpu(), cmap="viridis", aspect="auto")
        ax[0, 1].set_title("K(u, v_aligned)")

        ax[1, 0].imshow(k_shuffled.cpu(), cmap="viridis", aspect="auto")
        ax[1, 0].set_title("K(u, v_shuffled)")

        diag_aligned = k_aligned.diag().cpu()
        diag_shuffled = k_shuffled.diag().cpu()
        combined = torch.cat([diag_aligned, diag_shuffled])
        bins = torch.linspace(combined.min(), combined.max(), 40)

        # Render shuffled as translucent fill and aligned as thick outline so both stay visible when overlapping.
        ax[1, 1].hist(diag_shuffled, bins=bins, alpha=0.35, color="tab:orange", label="diag shuffled")
        ax[1, 1].hist(diag_aligned, bins=bins, histtype="step", linewidth=2.2, color="tab:blue", label="diag aligned")
        ax[1, 1].axvline(diag_aligned.mean().item(), color="tab:blue", linestyle="--", linewidth=1.3, alpha=0.9)
        ax[1, 1].axvline(diag_shuffled.mean().item(), color="tab:orange", linestyle="--", linewidth=1.3, alpha=0.9)
        ax[1, 1].set_xlim(float(combined.min().item()), float(combined.max().item()))
        ax[1, 1].set_title(f"Diag K values | MI aligned={mi_aligned:.4f} / shuffled={mi_shuffled:.4f}")
        ax[1, 1].legend()

        out_path = Path("./inforidge_alignment_debug.png")
        diag_path = Path("./inforidge_alignment_debug.txt")
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        with diag_path.open("w") as f:
            f.write("InfoRidge debug diagnostics\n")
            f.write(f"sigma={sigma:.8f}\n")
            f.write(f"mi_aligned={mi_aligned:.8f}\n")
            f.write(f"mi_shuffled={mi_shuffled:.8f}\n")
            f.write("\n")

            for name, values in (
                ("distance_aligned", d_aligned),
                ("distance_shuffled", d_shuffled),
                ("distance_over_sigma_aligned", d_aligned / sigma),
                ("distance_over_sigma_shuffled", d_shuffled / sigma),
                ("diag_kernel_aligned", diag_aligned),
                ("diag_kernel_shuffled", diag_shuffled),
            ):
                stats = _summary_stats(values)
                f.write(f"[{name}]\n")
                for k in ("min", "p10", "p50", "p90", "max", "mean", "std"):
                    f.write(f"{k}={stats[k]:.8f}\n")
                f.write("\n")

        self.assertTrue(out_path.exists())
        self.assertTrue(diag_path.exists())

    def test_layerwise_metrics_capture_signal_addition(self):
        torch.manual_seed(11)
        batch_size = 64
        seq_len = 4
        hidden_dim = 8
        vocab_size = 64

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        embed = nn.Embedding(vocab_size, hidden_dim)

        with torch.no_grad():
            y = embed(labels.to(torch.int32)).to(torch.float64)

        layer0 = torch.randn(batch_size, seq_len, hidden_dim)  # weak signal
        layer1 = y.to(torch.float32)  # strong signal injection (exact label embedding)
        layer2 = layer1 + 0.005 * torch.randn(batch_size, seq_len, hidden_dim)  # tiny residual update
        sigma = _median_heuristic_sigma(y.reshape(-1, hidden_dim))

        metrics = compute_layerwise_info_metrics(
            [layer0, layer1, layer2],
            labels,
            embed,
            max_tokens=256,
            sample_seed=123,
            sigma=sigma,
        )

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.predictive_mi.shape[0], 3)
        self.assertEqual(metrics.incremental_mi.shape[0], 3)
        self.assertTrue(torch.isnan(metrics.incremental_mi[0]).item())
        self.assertGreater(metrics.predictive_mi[1].item(), metrics.predictive_mi[0].item())
        self.assertGreater(metrics.incremental_mi[1].item(), metrics.incremental_mi[2].item())

    def test_ignore_mask_and_prefix_and_sampling_cap(self):
        torch.manual_seed(3)
        labels = torch.tensor([
            [1, 2, -100, 3],
            [4, -100, -100, 5],
        ])
        embed = nn.Embedding(10, 8)
        prefix_len = 2

        # Sequence length includes a non-predictive prefix segment.
        layer_a = torch.randn(2, 6, 8)
        layer_b = layer_a + 0.1 * torch.randn(2, 6, 8)

        metrics = compute_layerwise_info_metrics(
            [layer_a, layer_b],
            labels,
            embed,
            prefix_len=prefix_len,
            max_tokens=3,
            sample_seed=9,
            sample_mode="all_tokens",
        )

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.sample_count, 3)
        self.assertEqual(metrics.predictive_mi.shape[0], 2)
        self.assertTrue(torch.isnan(metrics.incremental_mi[0]).item())

    def test_last_token_mode_uses_one_sample_per_sequence(self):
        torch.manual_seed(5)
        labels = torch.tensor([
            [1, -100, 2, 3],      # last valid position = 3
            [4, 5, -100, -100],   # last valid position = 1
            [-100, -100, -100, -100],  # filtered out
        ])
        embed = nn.Embedding(10, 8)
        layer_a = torch.randn(3, 4, 8)
        layer_b = layer_a + 0.1 * torch.randn(3, 4, 8)

        metrics = compute_layerwise_info_metrics(
            [layer_a, layer_b],
            labels,
            embed,
            max_tokens=10,
            sample_seed=0,
            sample_mode="last_token",
        )

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.predictive_mi.shape[0], 2)
        self.assertTrue(torch.isnan(metrics.incremental_mi[0]).item())

    def test_no_valid_tokens_returns_none(self):
        labels = torch.full((2, 3), -100)
        embed = nn.Embedding(10, 8)
        layer = torch.randn(2, 3, 8)

        metrics = compute_layerwise_info_metrics([layer], labels, embed, max_tokens=16)
        self.assertIsNone(metrics)

    def test_second_pass_target_embedding_mode_matches_lookup(self):
        torch.manual_seed(17)
        labels = torch.tensor([
            [1, 2, 3, -100],
            [4, 5, -100, -100],
            [6, 7, 8, 9],
        ], dtype=torch.int64)
        inputs = torch.zeros_like(labels)
        puzzle_identifiers = torch.zeros((labels.shape[0],), dtype=torch.int32)
        embed = nn.Embedding(16, 8)

        layer_a = torch.randn(3, 4, 8)
        layer_b = layer_a + 0.1 * torch.randn(3, 4, 8)
        sigma = _median_heuristic_sigma(layer_a.reshape(-1, layer_a.shape[-1]).to(torch.float64))

        def fake_input_embeddings(input_ids: torch.Tensor, _puzzle_ids: torch.Tensor) -> torch.Tensor:
            return embed(input_ids.to(torch.int32))

        metrics_lookup = compute_layerwise_info_metrics(
            [layer_a, layer_b],
            labels,
            embed,
            max_tokens=10,
            sample_seed=0,
            sigma=sigma,
            sample_mode="last_token",
            target_embedding_mode="lookup",
        )
        metrics_second_pass = compute_layerwise_info_metrics(
            [layer_a, layer_b],
            labels,
            embed,
            max_tokens=10,
            sample_seed=0,
            sigma=sigma,
            sample_mode="last_token",
            target_embedding_mode="second_pass",
            inputs=inputs,
            puzzle_identifiers=puzzle_identifiers,
            input_embeddings_fn=fake_input_embeddings,
        )

        self.assertIsNotNone(metrics_lookup)
        self.assertIsNotNone(metrics_second_pass)
        assert metrics_lookup is not None
        assert metrics_second_pass is not None
        self.assertEqual(metrics_lookup.sample_count, metrics_second_pass.sample_count)
        self.assertTrue(
            torch.allclose(
                metrics_lookup.predictive_mi,
                metrics_second_pass.predictive_mi,
                atol=1e-10,
                rtol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics_lookup.incremental_mi[1:],
                metrics_second_pass.incremental_mi[1:],
                atol=1e-10,
                rtol=1e-7,
            )
        )


class _DummyCarry:
    def __init__(self, labels: torch.Tensor):
        batch_size = labels.shape[0]
        self.current_data = {"labels": labels}
        self.halted = torch.ones(batch_size, dtype=torch.bool)
        self.steps = torch.ones(batch_size, dtype=torch.int32)


class _DummyInner(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, puzzle_emb_len: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.puzzle_emb_len = puzzle_emb_len


class _DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 12, puzzle_emb_len: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.inner = _DummyInner(vocab_size, hidden_dim, puzzle_emb_len)

    def initial_carry(self, batch):
        return _DummyCarry(batch["labels"])

    def forward(self, carry, batch, return_layer_states: bool = False, return_z: bool = False):
        labels = batch["labels"]
        batch_size, seq_len = labels.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        outputs = {
            "logits": logits,
            "q_halt_logits": torch.zeros(batch_size),
            "q_continue_logits": torch.zeros(batch_size),
        }

        if return_layer_states:
            total_seq_len = seq_len + self.inner.puzzle_emb_len
            h0 = torch.randn(batch_size, total_seq_len, self.hidden_dim)
            h1 = h0 + 0.1 * torch.randn_like(h0)
            l0 = torch.randn(batch_size, total_seq_len, self.hidden_dim)
            l1 = l0 + 0.1 * torch.randn_like(l0)
            outputs["layer_states_H"] = [h0, h1]
            outputs["layer_states_L"] = [l0, l1]

        return _DummyCarry(labels), outputs


class InfoRidgeIntegrationTests(unittest.TestCase):
    def test_act_loss_head_returns_raw_inforidge_payload(self):
        torch.manual_seed(13)
        model = _DummyModel()
        head = ACTLossHead(model, loss_type="softmax_cross_entropy")

        labels = torch.tensor([
            [1, 2, 3, -100],
            [4, 5, -100, -100],
        ])
        batch = {
            "labels": labels,
            "inputs": torch.zeros_like(labels),
            "puzzle_identifiers": torch.zeros((labels.shape[0],), dtype=torch.int32),
        }
        carry = head.initial_carry(batch)

        _, _, _, detached_outputs, _ = head(return_keys=["inforidge"], carry=carry, batch=batch)
        self.assertIn("inforidge", detached_outputs)
        info = detached_outputs["inforidge"]

        self.assertIn("labels", info)
        self.assertIn("layer_states_H", info)
        self.assertIn("layer_states_L", info)
        self.assertEqual(len(info["layer_states_H"]), 2)
        self.assertEqual(len(info["layer_states_L"]), 2)


if __name__ == "__main__":
    unittest.main()
