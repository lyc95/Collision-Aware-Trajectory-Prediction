# Observation: The Trajectory-Reversal Artifact in Collision-Aware Training

## Summary

When the collision-aware loss is too aggressive — high λ, large d_min, or many gradient
steps per specialized batch — the model achieves low collision rate by predicting that
agents *reverse direction* or *stop moving*, rather than by gently steering them apart.
This produces unrealistic, socially-implausible trajectories that no real pedestrian
would walk. **The standard metrics (ADE, FDE, ColRate) fail to capture this artifact
on their own**, and visual inspection on synthetic head-on scenarios is required to
detect it. This is a fundamental trade-off in collision-aware trajectory prediction
that we believe has been underreported in prior work.

## Phenomenon

In mock head-on collision scenarios (two agents walking toward each other along a
near-collinear axis), the baseline Social-STGCNN predicts that both agents continue
straight — producing collisions. A well-trained collision-aware model should predict
a small lateral swerve so agents pass each other. We observed three distinct failure
modes in the *more aggressive* specialized-trained models:

1. **Trajectory reversal** — the predicted mean trajectory turns around mid-prediction
   so the agent ends up behind its starting position (negative forward displacement).
2. **Trajectory stalling** — the predicted mean barely moves; agents are predicted
   to be near-stationary across the full prediction horizon.
3. **Excessive lateral deviation** — the predicted mean swerves so far laterally
   (e.g., 1 m or more) that no real pedestrian would walk such a path to avoid a
   collision that does not require it.

All three modes succeed at one objective — reducing the collision rate — at the cost
of trajectory realism.

## Empirical Evidence

We evaluated multiple specialized-trained models on a synthetic 2-agent head-on
scenario (both agents walking at 0.15 m/s toward the origin, 18 cm lateral offset,
collision at prediction frame 8). For each model we measured:

- `ColRate (mock)` — fraction of K=20 stochastic samples with any pair within 0.2 m
- `Mean Δx` — predicted forward displacement of the mean trajectory
  (positive = moves toward original walking direction; negative = reversed)
- `Lateral spread` — max lateral deviation of the mean trajectory from observed line

Approximate measurements from our visualization runs:

| Model (specialized, zara2)              | Test ADE | Test ColRate | Mock collisions | Mean Δx (head-on) | Realism      |
|------------------------------------------|----------|--------------|-----------------|--------------------|--------------|
| Baseline (no collision loss)             | 0.30     | 0.43         | 6/20            | +1.4 m (forward)   | Realistic    |
| HINGE  d=2.0 λ=10  [val_best]            | 0.44     | 0.37         | **0/20**        | **+1.1 m (forward swerve)** | **Realistic — gentle swerve** |
| HINGE  d=2.0 λ=10  [final ep]            | 0.48     | 0.36         | 0/20            | +0.9 m (forward swerve)     | Realistic    |
| GAUSS  d=1.0 λ=10  [final ep]            | 0.57     | 0.38         | 3/20            | +0.7 m (curved)             | Borderline   |
| ECP    d=2.0 λ=10  [val_best]            | 0.71     | 0.29         | 0/20            | **−0.3 m (reversed)**       | **Unrealistic — turns around** |
| ECP    d=2.0 λ=10  [final ep]            | 0.94     | 0.24         | 0/20            | **−0.6 m (reversed)**       | **Unrealistic** |
| EXP    d=2.0 λ=10  [final ep]            | 1.09     | 0.23         | 0/20            | **near zero (stalled)**     | **Unrealistic** |
| GAUSS  d=2.0 λ=10  [final ep]            | 2.13     | 0.14         | 0/20            | **near zero (collapsed)**   | **Catastrophic** |

**Key finding:** the configurations that achieve the best raw ColRate
(ECP/GAUSS d=2.0 λ=10) also produce the most physically-implausible predictions.
The HINGE d=2.0 λ=10 [val_best] configuration is the Pareto-optimal point for
*realistic* avoidance — collision rate drops from 6/20 to 0/20 on the mock scene
while the mean trajectory still moves forward at roughly its observed speed.

## Mechanism: Why This Happens

The collision penalty is a function of pairwise predicted distance only. Its gradient
at every point pushes any direction that *increases* pairwise distance. In a
symmetric head-on scenario:

```
                              ← agent B
   agent A  →
              ↑↓
        Three directions all increase pairwise distance:
        (a) lateral (swerve up/down)
        (b) reversal (walk backward)
        (c) stalling (don't move)
```

The optimization is indifferent between these three directions until other terms in
the loss break the symmetry. With a strong NLL loss and a *gentle* collision penalty
(low λ, small d_min), option (a) — lateral swerve — wins because it has the smallest
NLL cost: real pedestrians do swerve sideways. The lateral predictions retain forward
motion and look natural.

With an *aggressive* collision penalty (high λ, large d_min, many specialized steps),
the NLL gradient is overwhelmed and any pairwise-distance-increasing direction
becomes acceptable. Reversal (b) and stalling (c) are perfectly valid solutions to
the new objective. They are the fastest way to *guarantee* that no pair of predicted
positions ever falls inside the d_min radius — including pairs that were never in
collision in the first place.

The specialized training protocol amplifies this effect because:

1. Specialized batches contain *only* near-collision scenes.
2. Within those scenes the collision gradient dominates every step.
3. There is no "easy" non-collision scene to balance the gradient toward forward motion.

This is consistent with our metadata observations on round-2 specialized training:
gauss λ=2 with 10 gradient steps per batch achieved the lowest ColRate of the entire
study (0.138) — but at ADE 2.13, ~7× the baseline. That model was *predicting nothing*.

## Why Standard Metrics Miss It

- **ColRate rewards the artifact**: a frozen model has zero collisions.
- **ADE captures it imperfectly**: ADE 0.94 (ECP d=2.0 λ=10 final) does not look
  catastrophic in a table, yet visualization reveals reversal. ADE goes up because
  the predictions don't match ground-truth forward motion, but the metric does not
  distinguish "reversed by 30 cm" from "swerved laterally by 30 cm".
- **FDE** has the same blindness as ADE.
- **Mock-scene visualization is the only reliable detector** with current metrics.

A simple supplementary metric we recommend for future work:

```
Forward-Progress Ratio  =  ⟨ predicted Δposition · observed velocity ⟩ / |observed velocity|²
```

A value near 1.0 means predictions move forward at the observed speed; 0 means stalled;
< 0 means reversed. Any value below ~0.3 should be treated as a reversal warning
and the configuration discarded for deployment, regardless of its ColRate.

## The Trade-off We Observed

Our experiments delineate four regimes along the collision-aware tuning axis,
each with a characteristic outcome:

| Regime           | Configuration example         | Outcome                                                           |
|------------------|-------------------------------|-------------------------------------------------------------------|
| **Under-tuned**  | d_min=0.5, λ=1, specialized   | ADE preserved (0.31) but minimal mock-scene collision reduction. |
| **Sweet spot**   | HINGE d=2.0 λ=10 [val_best]   | Realistic lateral swerve, mock-scene collisions 6→0, ADE +0.13. |
| **Over-correction** | ECP d=2.0 λ=10 [val_best]  | Reversal artifact appears, ADE +0.41, ColRate looks excellent. |
| **Collapse**     | GAUSS d=2.0 λ=10 [final ep]   | Near-static predictions, ADE +1.83, "best" ColRate.            |

The boundary between the *sweet spot* and *over-correction* regimes is narrow and
depends on the loss type, d_min, λ, and checkpoint selection (val_best vs final ep).
HINGE's bounded, constant gradient below d_min appears to extend the sweet spot
relative to the smoother but unbounded GAUSS/ECP penalties.

## Practical Recommendations

1. **Always validate collision-aware models on synthetic head-on scenarios**, not
   only on the test split. The test split lacks the symmetric pure-collision
   geometry that exposes the reversal artifact.
2. **Prefer val_best checkpoints over final-epoch** for collision-aware training.
   The final-epoch model has continued optimizing the collision term past the
   point where NLL has stabilized, deepening the artifact.
3. **Cap λ ≤ 10 for d_min ≤ 1.0**, and λ ≤ 5 for d_min ≥ 2.0.
4. **Report a forward-progress metric** alongside ADE/FDE/ColRate when publishing
   collision-aware results.

## Visual Evidence

`mock_collision_compare.png` (and the per-geometry variants `mock_collision_compare_head_on.png`,
`mock_collision_compare_t_cross.png`, `mock_collision_compare_lane_merge.png`) show
the difference between baseline, sweet-spot, and over-correction models on three
collision geometries (head-on, T-intersection, lane-merge) at three times-to-collision
(4, 8, 12 prediction frames). The reversal artifact is visible in the previously-rendered
ECP d=2.0 λ=10 panels, while HINGE d=2.0 λ=10 [val_best] preserves forward motion
even with collision rate dropping to zero.
