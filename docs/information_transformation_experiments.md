# Information Transformation in HRM

## Scientific Question

> How is task-relevant information transformed across HRM's recurrent H and L
> states, and which information transformations are causally necessary for
> solving a maze?

The project remains information-theoretic. Causal intervention is used to test
whether a measured information transformation is functionally required; it is
not the premise of the project.

## Previous Evidence

The analyzed HRM checkpoint uses four H blocks, four L blocks, and 16 ACT steps.
Its unmodified maze evaluation reached:

- exact accuracy: `0.736`;
- cell accuracy: `0.9915`.

The previous bypass experiment selected one block and returned its input instead
of its learned output every time that shared block was called. Under the saved
configuration, an H block was bypassed twice per ACT step and an L block four
times per ACT step. The intervention was therefore repeated approximately 32
times for an H block and 64 times for an L block during one maze.

| Bypassed block | Exact accuracy | Mean change in current `I(Z;Y)` summary |
|---|---:|---:|
| H0 | 0.155 | -0.0072 |
| H1 | 0.040 | -0.0072 |
| H2 | 0.370 | +0.0001 |
| H3 | 0.000 | -0.0151 |
| L0 | 0.000 | -0.0661 |
| L1 | 0.000 | -0.0087 |
| L2 | 0.052 | +0.0007 |
| L3 | 0.000 | -0.0023 |

The behavioral result is clear: repeatedly removing most learned block
transformations severely damages solving. The information result is more limited:
under the current estimator, the average change in marginal target dependence was
sometimes small relative to the behavioral collapse.

### What InfoRidge measured

The layerwise analysis estimated:

\[
I(Z;Y)
\]

between a hidden representation and a target-token embedding. It also estimated:

\[
I(Z_{after}-Z_{before};Y).
\]

The second quantity measures dependence between the update vector and the target.
It is not conditional information gain and must not be described as information
newly created by the block.

The saved layerwise runs also have important scope limits:

- they record block states from the final ACT step rather than every ACT step;
- they use one final valid target position per maze in the layerwise calculation;
- the reported intervention summary averages changes over measured H/L states;
- the estimator uses finite samples, a fixed kernel choice, and a fixed bandwidth.

Therefore the previous experiment does not establish that total solution
information is preserved after bypass. It establishes a sharper and more useful
tension:

> Severe behavioral damage was not consistently accompanied by a comparably
> large change in the particular marginal-dependence measurement we used.

We are not using the batch-shuffle or noise experiments as established evidence
for the present argument.

## The Information-Theoretic Ambiguity

A block can be functionally important without producing a large increase in
marginal `I(Z;Y)`.

**Reorganization.** The block may rotate, denoise, separate, or bind information
that was already present. The total dependence with the target can remain similar
while the information becomes easier for the next computation to use.

**Routing and compatibility.** H and L communicate through learned state
transformations. A state may still correlate with the target but be expressed in
a coordinate system or spatial arrangement that the next module or output head
cannot use correctly.

**Combination.** Useful information may be distributed across H, L, the input,
and spatial positions. A transition can combine variables whose joint value is
useful even when a marginal measurement changes little.

**Estimator limitation.** The current target choice, final-step sampling, kernel
bandwidth, aggregation, or sample size may make InfoRidge insensitive to the
relevant transformation.

These explanations are alternatives to test. None is currently established.

Here, "transformation" does not mean that a deterministic block creates Shannon
information from nothing. For a block output that is only a deterministic
function of its complete input, the data-processing inequality prevents true
mutual information with `Y` from increasing. The block can preserve, discard, or
re-encode information. Across recurrent updates, L can also incorporate
information available in H, and H can incorporate information available in L.
The experiments must therefore separate re-encoding within a block from transfer
between recurrent states.

## Measurements To Distinguish

For each block occurrence at ACT step `t`, collect the state immediately before
and after the block:

\[
Z_{before}^{t,l}, \qquad Z_{after}^{t,l}.
\]

We then distinguish four measurements.

### 1. Information present

\[
I(Z_{before};Y), \qquad I(Z_{after};Y)
\]

This asks how much marginal target dependence is detectable before and after the
transition.

### 2. Marginal information change

\[
\Delta I_{marginal}=I(Z_{after};Y)-I(Z_{before};Y)
\]

This is a difference between two marginal estimates. It is not conditional
information gain.

### 3. Update-target dependence

\[
I(Z_{after}-Z_{before};Y)
\]

This asks whether the direction written by the block is target-related. It does
not show that the update adds information beyond the previous state.

### 4. Predictive accessibility

Train held-out probes and compare their target cross-entropy:

\[
A_{t,l}=CE(Y\mid Z_{before})-CE(Y\mid Z_{after}).
\]

Positive `A` means that the target became easier for the chosen probe to recover.
This operationalizes usable or accessible information without claiming that
Shannon information was created.

### 5. Cross-module complementarity

At an H/L communication boundary, compare a probe using one recurrent state with
a probe using both:

\[
U_{H\mid L}=CE(Y\mid L)-CE(Y\mid L,H)
\]

and symmetrically `U_{L|H}`. These quantities estimate whether H contains target
signal not accessible from L alone, or vice versa. Evidence of transfer requires
more than a positive probe score: the signal should become accessible in the
receiving state after its update and disappear when the relevant communication
is causally disrupted. These probe differences should not be called conditional
mutual information until they pass synthetic and capacity controls.


## Analysis Commitments

These decisions are fixed before examining the new results.

**Hypotheses.** Before examining the new outputs, we test:

- `H1`: all-step bypass produces a positive mean paired `Delta NLL` for at least
  one block after multiple-testing correction;
- `H2`: within a block, paired bypass effects vary across mazes and are associated
  with clean-model difficulty or structural maze properties;
- `H3`: at some transitions, predictive accessibility changes more strongly than
  marginal `I(Z;Y)`, consistent with re-encoding rather than simple information
  accumulation;
- `H4`: the revised InfoRidge estimator separates aligned synthetic data from
  shuffled targets and preserves its qualitative conclusions across reasonable
  bandwidth and subsample choices.

The joint analysis then compares information loss, reorganization, routing
failure, off-trajectory dynamics, and estimator limitation as competing
explanations of the observed maze-level effects.

**Primary behavioral outcome.** For maze `x` and bypassed block `l`:

\[
\Delta NLL_{x,l}=NLL_{bypass}(x,l)-NLL_{clean}(x).
\]

Target margin, cell accuracy, and exact accuracy are secondary outcomes. All
clean-bypass comparisons are paired on the same maze.

**Statistical unit.** The maze is the independent unit. Confidence intervals use
bootstrap resampling by maze. Tests across many blocks and ACT locations use
Benjamini-Hochberg false-discovery-rate correction. Effect sizes and confidence
intervals are reported alongside corrected p-values.

**Discovery and confirmation.** The current checkpoint is the discovery study.
Any important mechanism must be reproduced on additional checkpoints and at least
three independently trained seeds before it becomes a strong general claim about
HRM rather than a result about one model.

## Experiment 1: Clean Information-Flow Baseline

Record `Z_before` and `Z_after` for every H/L block at every ACT step in the
unmodified model. For the same maze and ACT step, record target NLL, target margin,
cell accuracy, exact accuracy, and H/L update magnitude. This corrects the current
final-step limitation and shows whether representations and predictions improve,
stabilize, oscillate, or regress during recurrent computation.

The activation index retains the internal recurrence occurrence explicitly. With
the current `H_cycles=2` and `L_cycles=2` configuration, this means two H-module
occurrences and four L-module occurrences per ACT step; they are not averaged or
silently reduced to the final occurrence.

**Output:** one indexed activation dataset, per-step behavioral curves, and
block-by-ACT maps of clean information measurements.

## Experiment 2: Example-Level Block Bypass

Repeat the established all-step bypass for each of the eight blocks and save
per-maze outputs rather than only aggregate accuracy. Each maze is classified as:

- unaffected;
- still correct but with worse NLL or margin;
- partially incorrect;
- complete failure.

For every maze-block pair, report paired changes in NLL, margin, cell accuracy,
exact accuracy, downstream states, and trajectory divergence from the clean run.
This identifies whether dependence on a transformation is universal or specific
to particular examples. It does not yet identify why those examples differ.

**Output:** a maze-by-block effect table and distributions of paired behavioral
and trajectory effects.

## Experiment 3: Information-Measure Validation

Recalculate InfoRidge on the clean and bypassed states using a whole-maze target:
the flattened one-hot solution grid, with one statistical sample per maze and a
fixed normalization for the whole spatial hidden state. Compute `I(Z;Y)`, marginal
before-after differences, and update-target dependence at matched block
occurrences across ACT steps.

Validate the estimator with shuffled-target null distributions, synthetic
representations containing known target signal, repeated subsamples, and a
documented median-distance bandwidth rule. Report sensitivity at nearby bandwidths.
The estimator is considered usable only if aligned controls separate from shuffled
controls and the qualitative result survives reasonable bandwidth and subsample
choices.

**Output:** validated information maps, null distributions, estimator variability,
and bandwidth-sensitivity plots. This tests measurement reliability, not causal
use.

## Experiment 4: Predictive-Accessibility Probes

Train regularized linear probes before and after every block occurrence to predict
the solution label at each spatial position. Split probe training, validation, and
testing by maze so cells from the same maze cannot leak across partitions. Compare
held-out cross-entropy before and after the block:

\[
A_{t,l}=CE(Y\mid Z_{before})-CE(Y\mid Z_{after}).
\]

At H/L boundaries, compare probes using one recurrent state against probes using
both states to estimate complementarity. Use matched probe capacity and include
majority-class, input-only, and shuffled-label controls. A positive accessibility
gain means the target became easier for this probe to recover; it is not proof
that the model uses the decoded signal.

**Output:** accessibility gain and H/L complementarity by block and ACT step,
evaluated against the corrected InfoRidge results.

## Experiment 5: Joint Maze-Level Analysis

Build one aligned record for every maze and bypassed block containing clean and
bypass behavior, information measurements, accessibility, and trajectory
divergence. Add structural maze features such as shortest-path length, obstacle
density, reachable area, branch count, and dead-end count. Compare affected and
unaffected mazes using preregistered outcomes rather than selecting only visually
interesting examples.

The analysis explicitly compares alternative explanations:

- **information loss:** `I(Z;Y)` and accessibility fall before behavior degrades;
- **reorganization failure:** marginal `I(Z;Y)` remains stable, accessibility
  falls, and NLL rises;
- **routing failure:** information remains in the sending state but fails to
  become accessible in the receiving H/L state;
- **off-trajectory dynamics:** state divergence is large and persistent despite
  small changes in the measured target information;
- **estimator limitation:** behavior and accessibility change, but the validated
  information estimator remains insensitive.

Use multivariable models and stratified comparisons to test whether bypass effects
are associated with maze structure while controlling for clean-model difficulty.
The goal is to discriminate explanations, not to convert correlations into a
claim that a block implements a named algorithm.

**Output:** paired maze-level datasets, corrected effect estimates, affected-maze
profiles, and an evidence table comparing the alternative explanations.

## Focused Confirmation

Timed bypass or activation patching is not part of the initial five experiments.
It is selected after the joint analysis identifies the strongest interpretation.
A focused timed bypass tests when the transformation is necessary; activation
patching tests whether restoring the relevant clean state recovers information and
behavior. The chosen causal test and its expected result are specified before it
is run.

## Joint Interpretation

| Observation | Supported interpretation |
|---|---|
| `I(Z;Y)` falls, accessibility falls, and bypass damages behavior | The transition maintains or routes target-related content required downstream |
| `I(Z;Y)` is stable, accessibility improves, and bypass damages behavior | The transition reorganizes target information into a more usable form |
| `I(Z;Y)` is high but bypass has little effect | The information may be redundant or unused at that location |
| Information and accessibility appear stable but bypass damages behavior | The target or estimator misses routing, geometry, synergy, or another task variable |
| A focused timed bypass localizes the effect | A temporally specific transformation is necessary |
| Clean-state patching restores information and behavior | The patched state is causally sufficient under the chosen corruption |

## Execution Order

1. Freeze the hypotheses, outcomes, maze cohort, and analysis rules.
2. Build the clean information-flow baseline.
3. Run paired example-level bypasses for all blocks.
4. Validate the whole-maze information estimator.
5. Train predictive-accessibility probes.
6. Perform the joint maze-level analysis.
7. Select one focused timed-bypass or patching confirmation.
8. Replicate the strongest finding across checkpoints and at least three seeds.

## Meeting Formulation

> We study HRM reasoning as recurrent information transformation. Previous
> repeated bypasses caused severe behavioral failure without consistently large
> changes in our marginal target-dependence estimate. This week we will determine
> whether that mismatch reflects information reorganization, distributed or
> synergistic information, routing failure, off-trajectory dynamics, or limitations
> of the current estimator. We will map information before and after each
> transition, measure whether it becomes more predictively accessible, and align
> those measurements with paired example-level causal effects.
