# Tonight Experiments

Goal: use the best trained maze checkpoint and perturb internal layers at inference time. No retraining, no optimizer updates.

## Priority 1: Select the Best Non-Crashed W&B Run

Run this where `wandb` is installed and `WANDB_API_KEY` is available:

```bash
python scripts/select_best_wandb_checkpoint.py \
  --entity <wandb-entity> \
  --project maze-hrm \
  --metric all/exact_accuracy \
  --output-json /workspace/HRM/checkpoints/best_wandb_checkpoint.json \
  --print-layer-sweep-command
```

This filters to `finished` runs by default, scans each run's metric history, picks the best metric value and step, and constructs the matching checkpoint path from the run's `checkpoint_path` config.

If the W&B run config does not include `checkpoint_path`, pass:

```bash
--checkpoint-root /workspace/HRM/checkpoints
```

## Priority 2: Layer Intervention Sweep

Run this where the trained checkpoint exists:

```bash
CHECKPOINT_PATH=/workspace/HRM/checkpoints/maze-corrected-rerun/step_<best>.pt \
OUTPUT_ROOT=/workspace/HRM/checkpoints/layer-intervention-sweep-best \
bash scripts/run_layer_intervention_sweep.sh
```

By default this evaluates:

- levels: `H`, `L`
- layers: `0 1 2 3`
- interventions: `bypass`, `zero`, `shuffle_batch`, `mean`, `noise`

Each case uses the same checkpoint and writes its own `eval_metrics.csv`, `inforidge_step_*.csv`, and `inforidge_act_mi_step_*.csv`.

Interpretation:

- Big eval drop from `bypass` means that layer’s transformation is functionally necessary.
- Big drop from `shuffle_batch` means example-specific information in that layer matters.
- Big drop from `mean` means the layer carries per-example signal, not just global bias.
- `zero` is a hard ablation and mostly tells you gross dependency.
- `noise` tests robustness of the representation.

## Smaller First Pass

For a quick first pass:

```bash
CHECKPOINT_PATH=/workspace/HRM/checkpoints/maze-corrected-rerun/step_<best>.pt \
INTERVENTION_MODES="bypass shuffle_batch" \
LEVELS="H L" \
LAYERS="0 1 2 3" \
MAX_BATCHES=4 \
MAX_TOKENS=512 \
bash scripts/run_layer_intervention_sweep.sh
```

## Priority 3: Compare Outputs

Generate a ranked damage table:

```bash
python scripts/summarize_layer_interventions.py \
  /workspace/HRM/checkpoints/layer-intervention-sweep-best \
  --output-csv /workspace/HRM/checkpoints/layer-intervention-sweep-best/summary.csv
```

The important comparison is baseline vs each intervention:

- `eval/exact_accuracy`
- `eval/accuracy`
- `eval/steps`
- InfoRidge `I_Z_Y`
- InfoRidge `I_dZ_Y`
- ACT MI saturation pattern

The strongest “play with layers” result is a table like:

```text
case              exact_acc_delta   token_acc_delta   interpretation
H2-bypass         large negative    large negative    H layer 2 is causal
L3-shuffle_batch  large negative    small negative    L layer 3 has example-specific path detail
H0-mean           near zero         near zero         H layer 0 less critical at final checkpoint
```

## Priority 4: Success/Failure Examples

For the baseline and the most damaging interventions, save logits and inspect examples:

```bash
WANDB_MODE=offline \
DISABLE_COMPILE=1 \
python pretrain.py \
  data_path=data/maze-30x30-hard-1k \
  checkpoint_path=/workspace/HRM/checkpoints/layer-play-preds/H2-bypass \
  load_checkpoint_path=/workspace/HRM/checkpoints/maze-corrected-rerun/step_<best>.pt \
  analysis_only=true \
  global_batch_size=16 \
  eval_save_outputs='[inputs,labels,logits]' \
  arch.layer_intervention_level=H \
  arch.layer_intervention_layers='[2]' \
  arch.layer_intervention_mode=bypass
```

Then:

```bash
python scripts/analyze_maze_predictions.py \
  --pred-file /workspace/HRM/checkpoints/layer-play-preds/H2-bypass/step_<best>_all_preds.0 \
  --summary-json /workspace/HRM/checkpoints/layer-play-preds/H2-bypass/summary.json \
  --examples-csv /workspace/HRM/checkpoints/layer-play-preds/H2-bypass/examples.csv
```
