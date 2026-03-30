# Maze HRM Comparison: Old Run vs Corrected Run

## Inputs compared

- Old training run: `analysis/hrm-training-step7810/eval_metrics.csv`
- Old InfoRidge (small/full test variants): `analysis/inforidge-step7810/` and `analysis/inforidge-step7810-fulltest/`
- Corrected training run: `analysis/hrm-training-4090-6gpu-corrected/eval_metrics.csv`
- Corrected InfoRidge from the best checkpoint weights (`step_20832.pt`, loaded as model-only, so filenames use `step_0`): `analysis/inforidge-4090-6gpu-corrected-step20832/`

## Plots

- Eval comparison: `analysis/compare-maze-old-vs-corrected/eval_comparison.png`
- Layerwise InfoRidge comparison: `analysis/compare-maze-old-vs-corrected/layerwise_inforidge_comparison.png`
- ACT-step InfoRidge comparison: `analysis/compare-maze-old-vs-corrected/act_inforidge_comparison.png`

## Main outcome

The corrected run is a large improvement over the old run. The old run peaked at exact accuracy `0.068` on step `7810`. The corrected run peaked at exact accuracy `0.726` on step `20832`.

This means the earlier weak maze result was not evidence that HRM fundamentally fails on maze under the corrected setup. The corrected run clearly solves the task much better.

## Evaluation comparison

Old run best checkpoint:

- Step: `7810`
- Accuracy: `0.9628268`
- Exact accuracy: `0.068`
- LM loss: `0.10443103`
- Steps: `16.0`

Corrected run best checkpoint:

- Step: `20832`
- Accuracy: `0.9912611`
- Exact accuracy: `0.726`
- LM loss: `0.04901285`
- Steps: `16.0`

Interpretation:

- Task performance improved dramatically.
- The model still uses the full ACT budget (`16.0`) throughout evaluation.
- The final corrected checkpoint (`34720`) collapses back to exact accuracy `0.065`, so checkpoint selection / early stopping matters.

## Layerwise InfoRidge comparison

Old small-test predictive MI:

- `H`: `[0.5522, 0.5032, 0.5487, 0.5448]`
- `L`: `[0.5750, 0.5732, 0.5687, 0.5374]`

Old full-test predictive MI:

- `H`: `[0.5387, 0.4975, 0.5435, 0.5327]`
- `L`: `[0.5455, 0.5494, 0.5431, 0.5067]`

Corrected run predictive MI:

- `H`: `[0.5597, 0.5624, 0.4952, 0.4321]`
- `L`: `[0.5147, 0.5300, 0.5514, 0.5456]`

Ridge location shift:

- Old small-test ridge: `H0`, `L0`
- Old full-test ridge: `H2`, `L1`
- Corrected ridge: `H1`, `L2`

Interpretation:

- The corrected successful run is more front-loaded in `H`: the strongest predictive MI is at `H1`, then later `H` layers fall off.
- The corrected successful run is more back-loaded in `L`: the strongest predictive MI is at `L2`, with `L3` staying high.
- Relative to the weak run, information appears to move away from shallow `L0/L1` dominance and toward deeper low-level processing.

## ACT-step InfoRidge comparison

Old small-test ACT predictive MI:

- First: `0.00845`
- Max: `0.00899`
- Max incremental MI: `0.02028`

Old full-test ACT predictive MI:

- First: `0.01211`
- Max: `0.01311`
- Max incremental MI: `0.02001`

Corrected ACT predictive MI:

- First: `0.00583`
- Max: `0.00703`
- Max incremental MI: `0.02694`

Interpretation:

- Absolute ACT predictive MI is lower in the corrected successful run.
- ACT incremental MI is higher in the corrected successful run.
- So the corrected run does not win by having a more information-rich pooled ACT state at each step. It wins by making larger representation changes across ACT steps.

This is important: absolute `I(Z;Y)` at the ACT-step level is not monotonic with task success in these saved runs. The more useful signal here is the stronger per-step update magnitude (`I(dZ;Y)`).

## Decision

Move forward with HRM on maze under the corrected training setup.

Do not use the old weak maze run as the main reference anymore. It is now dominated by the corrected run.

However, do not interpret the corrected run as solving everything:

- Adaptive halting still did not emerge. Evaluation stayed at `16.0` steps.
- Late training degrades sharply, so early stopping / best-checkpoint selection is required.
- InfoRidge suggests the successful run changes *where* information is concentrated rather than simply increasing all InfoRidge metrics.

## Recommended next steps

1. Treat `step_20832.pt` as the main maze checkpoint for follow-up analysis.
2. Use best-checkpoint selection by eval exact accuracy, not final checkpoint.
3. Keep the corrected optimization setup for future maze experiments.
4. If the next research question is about ACT itself, focus on halting and ACT-step dynamics, because task performance improved while halting behavior did not.
