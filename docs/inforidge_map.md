# InfoRidge Implementation Guide

This document explains, in plain language, how the current codebase implements the InfoRidge idea. It is written as a practical map: where the logic lives, how the data moves, which paper claims are already implemented, and which ones are still pending.

## 1. Where to Start Reading

The best entry point is `pretrain.py`, because this is where the training loop decides when to run InfoRidge. In the evaluation phase, the script calls `run_inforidge_analysis(...)` and `log_inforidge_results(...)`, and then optionally calls `run_inforidge_act_mi(...)` and `log_inforidge_act_mi(...)`. You can see these calls in `pretrain.py` around lines 459 to 469.

The important point is separation of concerns: `pretrain.py` now orchestrates the workflow, while the InfoRidge-specific computation and logging are contained in `utils/inforidge_experiment.py`.

## 2. What Each File Is Responsible For

`utils/inforidge_experiment.py` contains the experiment runtime. This includes configuration classes, the logic that collects model outputs and computes layer-wise metrics, and the logic that writes those metrics to CSV and Weights & Biases.

`utils/inforidge.py` contains the mathematical core. This is where Gram matrices, matrix entropy, matrix-based mutual information, and layer-wise sampling/aggregation are defined. If you want to validate equations against code, this is the most important file.

`models/losses.py` and `models/hrm/hrm_act_v1.py` are the source of the intermediate states used by InfoRidge. The loss head exposes detached tensors for analysis (`layer_states_H`, `layer_states_L`, and ACT states), and the HRM model produces those tensors during forward passes.

`tests/test_inforidge.py` is a focused test suite for InfoRidge math behavior and payload wiring.

## 3. How the Current Implementation Matches the Paper

The paper’s main claim is that predictive information can be tracked across depth using mutual information between hidden representations and the target token embedding. In code, this is implemented in `compute_layerwise_info_metrics(...)` in `utils/inforidge.py`.

Predictive information \(I(Z_\ell; Y)\) is computed for each layer by applying `matrix_mutual_information(z, y, sigma)` to the current layer representation and target embedding. Incremental information gain \(I(\Delta Z_\ell; Y)\) is computed by first forming \(\Delta Z_\ell = Z_\ell - Z_{\ell-1}\), then applying the same MI estimator to \(\Delta Z_\ell\) and \(Y\).

The mathematical estimator itself follows the matrix-based formulation: a trace-normalized Gram matrix is built from pairwise distances, entropy is computed from eigenvalues, and MI is formed as \(H(U) + H(V) - H(U \circ V)\), where \(\circ\) is the Hadamard product. These steps are implemented in `gram_matrix(...)`, `matrix_entropy(...)`, and `matrix_mutual_information(...)`.

## 4. How Representations Are Extracted in Practice

The model wrapper `ACTLossHead.forward(...)` in `models/losses.py` controls which extra tensors are returned during a forward call. When `return_keys` includes `"inforidge"`, it returns `labels`, `layer_states_H`, and `layer_states_L`. When `return_keys` includes `"inforidge_act_mi"`, it returns ACT states (`z_H`, `z_L`) for per-step analysis.

Inside the model implementation (`models/hrm/hrm_act_v1.py`), these tensors are produced during forward execution and attached to the output dictionary. InfoRidge then consumes them from `utils/inforidge_experiment.py`.

For target extraction, the code embeds the selected label tokens with the model’s embedding table. For state extraction, the code supports two modes: all valid tokens or last valid token per sequence. The default is last-token mode, which corresponds to the paper’s emphasis on next-token prediction at the final relevant position.

## 5. What “Generalization Ridge” Means in This Code

The hypothesis in your text defines a ridge layer \(\ell^* = \arg\max_\ell I(Z_\ell; Y)\). The current implementation computes and logs \(I(Z_\ell; Y)\) for every layer, so the ridge is observable from outputs. However, there is no dedicated helper yet that explicitly computes and stores \(\ell^*\) as a first-class metric.

In practical terms, you can identify the ridge today by taking the maximum `I_Z_Y` across layers in the output CSV or W&B plots.

## 6. What Is Implemented Exactly vs What Is Still Missing

The MI-based layer-wise tracking is implemented and operational. The ACT-step variant is also implemented and logs predictive and incremental MI across ACT iterations.

The following ideas from your pasted text are not yet implemented in this repository: Wasserstein-distance analysis, explicit minimum-Wasserstein generalization bound calculations, alternate runtime kernels (Laplacian and Polynomial), and a strict two-pass `x` then `x+y` extraction pipeline for target embeddings.

So the current code captures the core InfoRidge MI methodology, but not the full set of auxiliary analyses described in the broader framing.

## 7. Where Outputs Are Written

Layer-wise outputs are written by `log_inforidge_results(...)` in `utils/inforidge_experiment.py`. The CSV contains `layer_type`, `layer_index`, `I_Z_Y`, and `I_dZ_Y`. The same values are also logged to Weights & Biases under `inforidge/...` keys.

ACT-step outputs are written by `log_inforidge_act_mi(...)` in the same module, with columns `act_step`, `I_Z_Y`, and `I_dZ_Y`.

## 8. What the Tests Actually Validate

`tests/test_inforidge.py` validates that the matrix MI estimator behaves sensibly (aligned pairs produce higher MI than shuffled pairs), that layer-wise MI responds correctly when synthetic signal is injected, and that masking/sampling behaviors match expectations.

It also validates that `ACTLossHead` correctly exposes InfoRidge payload tensors. This is important because InfoRidge depends on those payloads to compute metrics.

What is not yet covered by direct unit tests is the end-to-end runtime/logging functions in `utils/inforidge_experiment.py` themselves. Those functions are currently validated mainly through integration runs.

## 9. Short Code Anchor List

If you want a compact reference while reading:

- Training loop calls: `pretrain.py` around lines 459 to 469.
- Experiment runtime and logging: `utils/inforidge_experiment.py`.
- MI math and layer-wise metric logic: `utils/inforidge.py`.
- Payload exposure from loss wrapper: `models/losses.py` in `ACTLossHead.forward(...)`.
- Model production of layer states and ACT states: `models/hrm/hrm_act_v1.py`.
- Tests: `tests/test_inforidge.py`.

