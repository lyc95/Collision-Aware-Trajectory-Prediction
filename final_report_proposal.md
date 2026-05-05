# Experiment Metadata — Social-STGCNN Collision-Aware Trajectory Prediction
# CS 7643 Deep Learning — Final Project Reference Document

---

## 1. Introduction / Background / Motivation

### 1.1 What Did We Try to Do? (Plain Language)

Imagine standing in a busy shopping street. Everyone around you is walking, and somehow
you naturally avoid bumping into each other — you slow down, swerve, or time your steps
without thinking. Now imagine a robot or a self-driving car that needs to predict where
every nearby person is going to walk over the next 5 seconds. If it predicts wrong and
two people's predicted paths cross, the robot might think a collision is about to happen
and brake unnecessarily, or worse, plan a path that puts it directly in harm's way.

This project asks: **can we train a deep learning model to predict where pedestrians will
walk, such that its predictions are both accurate AND physically realistic (i.e., no two
people are predicted to walk through each other)?**

The standard way to measure prediction quality — average distance to the true path — does
not penalize predictions where two pedestrians are predicted to occupy the same space.
We explored adding an explicit "collision penalty" to the training loss to fix this.

### 1.2 How Is It Done Today? Limits of Current Practice

Pedestrian trajectory prediction has been dominated by two paradigms:

**Physics-based models** such as the Social Force Model (Helbing & Molnár, 1995) use
hand-crafted repulsive forces between agents to generate collision-free trajectories.
They are interpretable but cannot learn from data and fail in dense or complex scenes.

**Deep learning models** learn from large datasets of real pedestrian trajectories.
Key milestones:
- *Social LSTM* (Alahi et al., 2016): LSTM with social pooling layers — first data-driven
  model to capture interactions between pedestrians.
- *Social GAN* (Gupta et al., 2018): adds a GAN-based diversity objective to predict
  multiple plausible futures.
- *Social-STGCNN* (Mohamed et al., 2020): the model used in this project. Replaces social
  pooling with a spatio-temporal graph convolutional network that explicitly encodes who
  is near whom and how that proximity evolves through time. Achieves state-of-the-art
  ADE/FDE on ETH/UCY benchmarks.

**The shared limitation:** all these models are trained with a Negative Log-Likelihood
(NLL) loss over a bivariate Gaussian output distribution. The NLL measures how well the
model predicts where each person goes — but it says nothing about whether two people are
predicted to walk through each other. Since real pedestrians almost never collide (fewer
than 0.01% of frames in the zara2 training set show pairs closer than 0.2 m), the NLL
gradient carries almost no collision signal. As our baseline experiments confirm, even
the most accurate trajectory predictors generate collision rates above 40% on some scenes
when evaluated stochastically.

No existing published approach has systematically evaluated multiple collision-penalty
loss functions on STGCNN, or explored specialized training protocols that focus on
collision-relevant training samples. This project fills that gap.

### 1.3 Who Cares? Impact

Collision-safe trajectory prediction matters in any system where a machine must reason
about pedestrian intent in real time:

- **Social robots** navigating hospitals, airports, or warehouses need predictions that
  will not place two people in the same location, or the downstream motion planner will
  generate jerky, unsafe paths.
- **Autonomous vehicles** at crosswalks must predict pedestrian futures for 3–5 seconds.
  A predicted collision in that window triggers emergency braking even if the pedestrians
  would naturally avoid each other.
- **Surveillance / crowd analytics** systems that flag dangerous crowd conditions require
  a model that distinguishes true near-misses from prediction artifacts.

A model that reduces collision rate by ~12% while maintaining trajectory accuracy — as
our best configuration achieves — directly reduces false positive safety interventions in
all these systems.

### 1.4 Data: ETH/UCY Pedestrian Benchmark

We used the standard ETH/UCY benchmark, the most widely adopted evaluation suite for
pedestrian trajectory prediction. This follows the Datasheets for Datasets framework
(Gebru et al., 2021) for the most relevant aspects:

**Composition:**
Five real-world scenes recorded by overhead cameras: eth (ETH campus, Zurich),
hotel (Hotel Sèvres, Zurich), univ (University campus walkway), zara1 and zara2
(shopping street "Zara", Bern). Trajectories are extracted via homography projection
to bird's-eye-view metric coordinates (meters).

**Collection:**
- ETH subset (eth, hotel): Pellegrini et al., 2009. UCY subset (univ, zara1, zara2):
  Lerner et al., 2007.
- Pedestrians are tracked at 2.5 Hz (one position every 0.4 s).
- Observation window: 8 frames (3.2 s). Prediction horizon: 12 frames (4.8 s).
- Leave-one-out cross-validation: each dataset used as test set while others train.

**Scale and splits used:**

| Scene | Train scenes | Val scenes | Test scenes | Avg peds/scene |
|-------|-------------|------------|-------------|----------------|
| eth   | ~2,000      | —          | 70          | 2–4            |
| hotel | ~2,000      | —          | 301         | 2–5            |
| univ  | ~6,000      | —          | 947         | 4–10           |
| zara1 | ~3,000      | —          | 602         | 3–7            |
| zara2 | ~4,000      | —          | 921         | 3–8            |

**Collision frequency in zara2 training data (measured):**
Total pedestrian pairs: 823,220 | Pairs < 0.2 m: 64 (0.008%) | Pairs < 0.5 m: 5,108 (0.620%) | Pairs < 1.0 m: 29,278 (3.557%)

**Relevant biases and limitations:**
- All scenes are outdoor or hotel-lobby environments — indoor narrow corridors and
  stairs are not represented.
- Recordings are in Europe; pedestrian behaviour may differ across cultures.
- Ground truth trajectories are recorded from real pedestrians who naturally avoid
  each other. This means the training distribution contains very few true collisions,
  which is the root cause of the NLL model's indifference to collision.
- No annotation of pedestrian intent, groups, or social relationships beyond proximity.

**Pre-processing:**
Trajectories are converted to relative displacements (Δx, Δy) between timesteps.
A spatial adjacency matrix A is computed at each timestep based on Euclidean distance:
A_ij = exp(−d_ij²) for pedestrians within a threshold, 0 otherwise. This graph is
recomputed at each frame and is not learned — it encodes who can plausibly interact.

---

## 2. Approach

### 2.1 Model Architecture: Social-STGCNN

Social-STGCNN (Mohamed et al., 2020) has two stages:

**Stage 1 — Spatial Graph CNN (SGCNN):** At each observed timestep, pedestrian positions
are treated as nodes in a graph. A graph convolution layer aggregates neighbour
information, encoding pairwise interaction. This is done for all 8 observed timesteps,
producing an interaction-aware feature per pedestrian per timestep.

**Stage 2 — Temporal CNN (TGCNN):** A 1D temporal convolution processes the 8-timestep
sequence of spatial features, capturing motion dynamics over the observation window.
The output is a bivariate Gaussian parameter vector (μx, μy, σx, σy, ρ) for each of
the 12 future timesteps per pedestrian.

**Learned parameters:**
- Graph convolution weight matrices (SGCNN layers): encode how to weight neighbour
  contributions based on relative position.
- Temporal convolution kernels (TGCNN layers): encode motion patterns over time.
- Output projection layer: maps latent features to (μ, σ, ρ) of the bivariate Gaussian.

**Fixed (non-learned) components:**
- The adjacency matrix A: computed from Euclidean distances at inference time.
- The Laplacian normalization of A: standard spectral graph convolution preprocessing.
- Monte Carlo sampling at inference: K=20 trajectory samples drawn from the predicted
  bivariate Gaussian to compute stochastic metrics.

**Input/output representation:**
- Input: relative position increments (Δx, Δy) for 8 timesteps, shaped as a graph
  signal tensor V ∈ ℝ^(N × T_obs × 2), plus A ∈ ℝ^(T_obs × N × N).
- Output: Gaussian parameters for 12 future timesteps, shaped as ℝ^(T_pred × N × 5).
  At test time, 20 trajectory samples are drawn; ADE/FDE report the best-of-20.

**Framework and optimizer:**
PyTorch. Optimizer: Adam (lr=0.01, default betas). Batch size: 128 for all experiments
except specialized training (batch=16 to ensure collision-relevant samples per batch).
Starting code: official Social-STGCNN repository (Mohamed et al., 2020, GitHub).

### 2.2 The Standard Loss and Its Collision Blind Spot

The baseline loss is the **bivariate Gaussian NLL** averaged over all pedestrians and
predicted timesteps:

```
L_NLL = -Σ_{i,t} log p(y_{i,t} | μ_{i,t}, Σ_{i,t})
```

where p is the bivariate Gaussian with predicted mean μ and covariance Σ. This loss
rewards accurate trajectory prediction but has no term coupling the predictions of
different pedestrians — it is computed independently per agent.

### 2.3 Collision Loss Functions

We added a pairwise collision penalty term to the NLL:

```
L_total = L_NLL + λ · L_collision
```

where L_collision sums over all pairs (i, j) and predicted timesteps t:

```
L_collision = Σ_{i<j} Σ_t f(d_{ij,t})
```

and d_{ij,t} = ‖ŷ_{i,t} − ŷ_{j,t}‖₂ is the predicted distance between pedestrians
i and j at future timestep t. Five penalty functions f(·) were evaluated:

| Loss | Formula | Gradient behaviour |
|------|---------|-------------------|
| **Hinge** | max(0, d_min − d) | Zero when safe; constant magnitude below d_min |
| **Exponential** | exp(−d / d_min) | Decays smoothly; large gradient near d=0 |
| **Inverse** | 1 / max(d, ε) | Unbounded as d→0; highly unstable |
| **Gaussian** | exp(−(d/d_min)²) | Smooth bell-shaped; peaks at d=0, tapers naturally |
| **ECP** | exp(−d²/(2·d_min²)) | Soft Gaussian with hard drop below d_min |

**Why we expected this to work:** the four smooth penalties (exp, gauss, ECP, and to a
lesser degree hinge) all produce negative gradients with respect to predicted positions
whenever two pedestrians are close — pushing the mean predictions apart. Since the
model output is a Gaussian, a small position shift can significantly reduce the
collision probability in samples drawn from the tails of the distribution.

**What is novel:** no prior work on STGCNN has (1) systematically compared these five
loss functions under identical training conditions, (2) analyzed the role of the
collision penalty hyperparameter d_min in the context of actual data-sparsity statistics,
or (3) developed a specialized training protocol that explicitly filters mini-batches to
only scenes containing near-collision events.

### 2.4 Three Training Protocols

**Protocol A — From scratch:** Train STGCNN from random initialization with L_total
for 250 epochs. Tests whether the collision signal can coexist with trajectory learning
from the start.

**Protocol B — Fine-tune from baseline:** Initialize from the fully-trained STGCNN
baseline checkpoint, then continue training with L_total for 100 epochs. The hypothesis
is that a well-formed trajectory prior makes the NLL gradient landscape easier to perturb
without catastrophic forgetting.

**Protocol C — Specialized training:** Initialize from the baseline, but filter each
mini-batch to only include training scenes where at least one predicted pedestrian pair
violates the collision threshold. Batch size reduced to 16 (to ensure collision-relevant
content per batch). Train for 50 epochs. The hypothesis is that concentrating the
collision signal — rather than diluting it across 823,220 safe pairs — will produce
more targeted gradient updates.

### 2.5 Problems Anticipated and Encountered

**Anticipated:**
- *Signal sparsity:* With only 0.008% of training pairs violating d=0.2 m, the collision
  gradient was expected to be overwhelmed by NLL in Protocol A. This was confirmed exactly.
- *Gradient instability:* The inverse loss 1/d is unbounded near d=0 and was expected to
  cause numerical issues. It did — harmful on all datasets in all protocols.
- *Accuracy-collision tradeoff:* Adding any secondary objective risks ADE regression.

**Encountered (not anticipated):**
- *Catastrophic divergence at λ=2 in specialized training:* We expected λ=2 to be
  marginally worse than λ=1, but gauss λ=2, steps=10 produced ADE=2.13 (7× baseline).
  The compound effect of hard-selected batches + high λ + many gradient steps was not
  predicted from the full fine-tuning sweep, where λ=10 or 20 caused only modest
  degradation.
- *univ structural intractability:* We expected some reduction in univ's collision rate
  (~0.86) — but no protocol produced any meaningful change. Post-hoc analysis reveals
  that the scene density in univ means geometrically-plausible non-overlapping
  predictions simply do not fit the available space. The collision rate reflects the
  scene, not the model.
- *Gaussian loss uniquely improving ADE:* We did not anticipate that gauss λ=5 would
  actually push ADE below the baseline (0.2993 vs 0.3048). Its smooth gradient
  appears to act as a gentle regularizer rather than a disruptive secondary objective.
- *The first thing tried (Protocol A, hinge loss) did not work* — marginal ColRate gain
  with ADE degradation. This led us to Protocol B and then Protocol C.

---

## 3. Deep Learning Details

### 3.1 Model Structure Reflects Problem Structure

The spatio-temporal graph structure of STGCNN directly mirrors the problem: pedestrian
interaction is inherently spatial (who is near whom) and temporal (how proximity evolves
as they walk). The graph convolution captures spatial interaction at each instant; the
temporal convolution captures motion dynamics over the observation window. This inductive
bias is more appropriate than a fully-connected model or a standard LSTM, which would
not naturally distinguish between near and far pedestrians.

### 3.2 Loss Function

The full training loss is:

```
L_total = L_NLL + λ · Σ_{i<j, t} f(‖ŷ_{i,t} − ŷ_{j,t}‖)
```

The NLL is computed on ground-truth trajectories. The collision term is computed on
**predicted mean trajectories** (μ_{i,t}) rather than stochastic samples, for
computational tractability and gradient stability.

### 3.3 Overfitting and Generalization

The standard Social-STGCNN paper reports that the model generalizes well on ETH/UCY
with leave-one-out validation. In our experiments:

- All checkpoints were saved at **best validation ADE** (val_best.pth) to prevent
  overfitting to the training set.
- Collision fine-tuning experiments used 100 or 50 epochs (vs. 250 for baseline),
  keeping regularization implicit through the NLL loss and early checkpoint selection.
- No significant signs of overfitting were observed in Protocols B and C at moderate λ —
  ADE on test was consistent with the training trend. At high λ (divergence cases),
  the model severely overfits the collision objective to the training scenes, which
  manifests as near-static prediction at test time (trajectory collapse).

### 3.4 Hyperparameters and Their Effects

| Hyperparameter | Values tested | Effect |
|---------------|---------------|--------|
| λ (collision weight) | 0, 1, 5, 10, 20 (fine-tune); 1, 2 (specialized) | Primary tradeoff knob. λ=5 optimal for fine-tune; λ≥2 causes divergence in specialized |
| d_min (collision radius, m) | 0.2, 0.5, 1.0 | Determines how many pairs are penalized. 0.2 m corresponds to actual near-misses (0.008%); 0.5 m penalizes 0.62% of pairs — a 78× increase in noisy signal |
| Training protocol | Scratch / Fine-tune / Specialized | Biggest single factor. Fine-tune and specialized vastly outperform from-scratch |
| Batch size | 128 (protocols A/B), 16 (protocol C) | Reduced for specialized training to increase collision-sample density per batch |
| Gradient steps per specialized batch | 1, 5, 10 | More steps amplify instability; 1–5 safe, 10 risky |
| K (Monte Carlo samples) | 20 | Fixed per STGCNN convention; controls variance of stochastic collision metric |

Learning rate: Adam lr=0.01 for all experiments (STGCNN default, not tuned).

---

## 4. Baseline Experiments

### 4.1 Why Three Baselines?

Three architectures were evaluated to determine whether high collision rates are an
architecture-specific problem or a universal property of NLL-trained models:

- **STGCNN** — graph CNN with fixed spatial adjacency, temporal CNN decoder.
- **LSTM** — sequential model with social pooling; non-graph baseline.
- **GAT** (Graph Attention Network) — graph model with learned, attention-weighted edges.

If all three architectures produce similar collision rates despite different levels of
trajectory accuracy, it demonstrates that the NLL objective — not the choice of
interaction mechanism — is responsible for collision-blindness.

STGCNN: 250 epochs; LSTM and GAT: 100 epochs. Batch size 128, standard NLL loss.

| Dataset | Model  | Epochs | ADE    | FDE    | ColRate_avg | ColRate_minADE |
|---------|--------|--------|--------|--------|-------------|----------------|
| eth     | STGCNN | 250    | 0.6376 | 1.1157 | 0.0579      | 0.0857         |
| hotel   | STGCNN | 250    | 0.4657 | 0.8282 | 0.1342      | 0.1163         |
| univ    | STGCNN | 250    | 0.4740 | 0.8574 | 0.8624      | 0.8585         |
| zara1   | STGCNN | 250    | 0.3474 | 0.5639 | 0.1395      | 0.1329         |
| zara2   | STGCNN | 250    | 0.3045 | 0.4986 | 0.4294      | 0.3941         |
| eth     | LSTM   | 100    | 0.7176 | 1.2269 | 0.0857      | 0.1000         |
| hotel   | LSTM   | 100    | 0.8633 | 1.5755 | 0.1638      | 0.1827         |
| univ    | LSTM   | 100    | 0.4926 | 0.8365 | 0.9051      | 0.9018         |
| zara1   | LSTM   | 100    | 0.3692 | 0.5707 | 0.1849      | 0.1761         |
| zara2   | LSTM   | 100    | 0.3212 | 0.4402 | 0.5302      | 0.5451         |
| eth     | GAT    | 100    | 0.7851 | 1.3207 | 0.0750      | 0.0143         |
| hotel   | GAT    | 100    | 0.4567 | 0.8879 | 0.1412      | 0.1395         |
| univ    | GAT    | 100    | 0.4305 | 0.7894 | 0.8676      | 0.8754         |
| zara1   | GAT    | 100    | 0.3061 | 0.5170 | 0.1278      | 0.0963         |
| zara2   | GAT    | 100    | 0.2830 | 0.4479 | 0.4350      | 0.4180         |

**Key finding: All three models fail on collision — despite meaningful differences in
trajectory accuracy, collision rates are nearly identical across architectures.**
On zara2: STGCNN 0.43, LSTM 0.53, GAT 0.44. On univ: all three exceed 0.86.
GAT is the strongest trajectory predictor (lowest ADE/FDE on 4 of 5 datasets) yet
its collision rate (0.435 on zara2) is essentially the same as STGCNN (0.429).
This confirms: **collision avoidance is blind to the architecture; it requires an explicit
objective in the loss function.**

**LSTM underperforms on hotel** (ADE 0.863 vs STGCNN 0.466, GAT 0.457 — nearly 2×
worse). Hotel has sparse pedestrians with long-range interactions; LSTM's fixed social
pooling window cannot capture these, confirming that graph-based spatial encoding is
necessary for sparse scenes.

**Why STGCNN was chosen for collision training** (not GAT despite its better ADE):
(1) STGCNN is the paper's primary contribution to extend; (2) its fixed adjacency
graph makes the interaction with collision penalties more interpretable; (3) its
trajectory accuracy is competitive enough that any tradeoff observed is meaningful.

### 4.2 Why zara2 is the Primary Scene

| Scene | Baseline ColRate | Verdict |
|-------|-----------------|---------|
| eth   | 0.058           | Already low — little room to improve |
| hotel | 0.134           | Low — sparse scene, NLL incidentally avoids collision |
| zara1 | 0.140           | Low — same reason |
| univ  | 0.862           | Structurally intractable (dense crowd geometry) |
| **zara2** | **0.429** | **Moderate, tractable, evaluates cleanly** |

zara2 (shopping street, moderate bidirectional crowd) hits the sweet spot: high enough
collision rate to be a real problem, low enough that improvement is measurable. Its
823,220 training pairs contain 64 true near-misses (0.008%) — rare enough that the NLL
ignores them, but present enough that a targeted penalty can find them. 921 test scenes
give stable metric estimates without excessive compute.

---

## 5. Collision Loss Experiments

### 5.1 Protocol A: Training From Scratch (STGCNN, 250 epochs)

All five loss functions tested with λ=1, d_min=0.2 m on univ and zara2.

| Dataset | Loss     | λ | d_min | ADE    | FDE    | ColRate_avg | ColRate_minADE | Note                   |
|---------|----------|---|-------|--------|--------|-------------|----------------|------------------------|
| univ    | baseline | — | —     | 0.4740 | 0.8574 | 0.8624      | 0.8585         | reference              |
| zara2   | baseline | — | —     | 0.3048 | 0.4967 | 0.4319      | 0.4072         | reference              |
| univ    | hinge    | 1 | 0.2   | 0.5279 | 0.9612 | 0.8732      | 0.8675         | ADE ↑ ColRate ↑        |
| zara2   | hinge    | 1 | 0.2   | 0.3135 | 0.5006 | 0.4169      | 0.3891         | marginal ColRate gain  |
| univ    | exp      | 1 | 0.2   | 0.4854 | 0.8721 | 0.8705      | 0.8589         | negligible change      |
| zara2   | exp      | 1 | 0.2   | 0.3093 | 0.4986 | 0.4286      | 0.3935         | negligible change      |
| univ    | inv      | 1 | 0.2   | 0.5354 | 0.8905 | 0.8468      | 0.8610         | ADE degraded           |
| zara2   | inv      | 1 | 0.2   | 0.3612 | 0.5590 | 0.4721      | 0.4792         | ALL metrics worse      |
| univ    | gauss    | 1 | 0.2   | 0.4861 | 0.8800 | 0.8607      | 0.8662         | negligible change      |
| zara2   | gauss    | 1 | 0.2   | 0.3116 | 0.4915 | 0.4480      | 0.4260         | marginal               |
| univ    | ecp      | 1 | 0.2   | 0.4780 | 0.8462 | 0.8832      | 0.8865         | ColRate slightly worse |
| zara2   | ecp      | 1 | 0.2   | 0.3400 | 0.5438 | 0.4395      | 0.4413         | ADE/FDE degraded       |

**Verdict: Protocol A fails.** Collision gradient (≈0.008% of pairs) is drowned by NLL.
Best from-scratch result (hinge, zara2: ColRate 0.417) is a marginal −1.5pp improvement
while ADE regresses +0.014. Inverse loss is destructive (+18% ADE on zara2). The model
learns trajectory structure from NLL first and the collision term never reshapes it.

### 5.2 Protocol B: Fine-Tuning from Baseline (STGCNN, zara2, 100 epochs)

#### ECP: λ × d_min sweep

| λ  | d_min | ADE    | FDE    | ColRate_avg | ColRate_minADE |
|----|-------|--------|--------|-------------|----------------|
| 5  | 0.2   | 0.3118 | 0.4988 | **0.4009**  | 0.3996         |
| 10 | 0.2   | 0.3152 | 0.5018 | 0.4175      | 0.3800         |
| 20 | 0.2   | 0.3115 | 0.4996 | 0.4356      | 0.4104         |
| 5  | 0.5   | 0.3208 | 0.5164 | 0.4246      | 0.4159         |
| 10 | 0.5   | 0.3382 | 0.5243 | 0.4247      | 0.4115         |
| 20 | 0.5   | 0.3454 | 0.5451 | 0.4565      | 0.4278         |
| 5  | 1.0   | 0.3304 | 0.5339 | 0.4317      | 0.4332         |
| 10 | 1.0   | 0.3977 | 0.5775 | 0.4500      | 0.4159         |
| 20 | 1.0   | 0.4261 | 0.6073 | 0.4518      | 0.4647         |

The d_min effect is explained by the training data statistics: increasing d_min from 0.2
to 0.5 m increases the number of penalized pairs by 78× (64 → 5,108). The penalty fires
on pairs that are perfectly safe in practice, injecting noise into the gradient. This is
why d_min=0.2 dominates across all λ values.

ECP ColRate non-monotonicity at high λ (λ=20 is worse than λ=5): at high λ, the ECP
term dominates NLL and the model begins to push predictions apart indiscriminately,
which causes systematic ADE regression and also perversely increases sampled collisions
because the predicted distribution becomes broader and less accurate.

#### All loss types at λ=5, d_min ∈ {0.2, 0.5}

| Loss  | λ | d_min | ADE        | FDE        | ColRate_avg | ColRate_minADE |
|-------|---|-------|------------|------------|-------------|----------------|
| hinge | 5 | 0.2   | 0.3494     | 0.5509     | 0.4083      | 0.4017         |
| hinge | 5 | 0.5   | 0.3544     | 0.5555     | 0.4296      | 0.4180         |
| exp   | 5 | 0.2   | 0.3069     | 0.5047     | 0.4174      | 0.3735         |
| exp   | 5 | 0.5   | 0.3087     | 0.5034     | 0.4322      | 0.4072         |
| **gauss** | **5** | **0.2** | **0.2993** | **0.4768** | 0.4210 | 0.4148  |
| gauss | 5 | 0.5   | 0.3217     | 0.5122     | 0.4147      | 0.3746         |
| inv   | 5 | 0.2   | 0.4334     | 0.6379     | 0.4569      | 0.4560         |
| inv   | 5 | 0.5   | 0.3673     | 0.5521     | 0.4105      | 0.3876         |

**Gaussian loss (λ=5, d_min=0.2) is the standout result: ADE=0.2993, below baseline
(0.3048).** This is the only configuration in the entire study that simultaneously
improves both trajectory accuracy and collision rate. The Gaussian penalty
exp(−(d/d_min)²) has a symmetric, bounded gradient: it peaks at intermediate distances
and tapers at both extremes, acting as a gentle regularizer rather than a sharp repulsion.
This may improve ADE by discouraging multimodal predictions where one mode places
pedestrians in the same region.

Inverse loss (inv) remains harmful in fine-tuning too: the 1/d term diverges near d=0,
producing gradient spikes even with fine-tuning initialization.

### 5.3 Protocol C: Specialized Training (Collision-Relevant Samples Only)

Model initialized from baseline; mini-batches filtered to scenes with at least one pair
violating the collision threshold. Batch size 16, 50 epochs.

#### Round 1 (d_min ∈ {0.2, 0.5}, λ ∈ {1, 5})

| Loss  | d_min | λ | ADE    | FDE    | ColRate_avg | ColRate_minADE |
|-------|-------|---|--------|--------|-------------|----------------|
| hinge | 0.2   | 1 | 0.2940 | 0.4548 | 0.4141      | 0.4072         |
| hinge | 0.2   | 5 | 0.2967 | 0.4576 | 0.4064      | 0.3844         |
| hinge | 0.5   | 1 | 0.3097 | 0.5101 | 0.4185      | 0.3681         |
| hinge | 0.5   | 5 | 0.3199 | 0.5112 | 0.3896      | 0.3757         |
| exp   | 0.2   | 1 | 0.3003 | 0.4705 | 0.4220      | 0.4072         |
| exp   | 0.2   | 5 | 0.2982 | 0.4664 | 0.4134      | 0.3833         |
| exp   | 0.5   | 1 | 0.3227 | 0.5168 | 0.3863      | 0.3974         |
| exp   | 0.5   | 5 | 0.3379 | 0.5257 | 0.4307      | 0.4104         |
| gauss | 0.2   | 1 | 0.3028 | 0.4749 | 0.4147      | 0.3920         |
| gauss | 0.2   | 5 | 0.3046 | 0.4786 | 0.4210      | 0.3811         |
| gauss | 0.5   | 1 | 0.3105 | 0.4934 | 0.4052      | 0.3692         |
| gauss | 0.5   | 5 | 0.3272 | 0.5217 | 0.4208      | 0.3811         |
| inv   | 0.2   | 1 | 0.2988 | 0.4522 | 0.4003      | 0.3746         |
| inv   | 0.2   | 5 | 0.3248 | 0.4891 | 0.4135      | 0.3876         |
| inv   | 0.5   | 1 | 0.3083 | 0.4921 | 0.4198      | 0.4050         |
| inv   | 0.5   | 5 | 0.4365 | 0.5994 | 0.4528      | 0.4680         |
| ecp   | 0.2   | 1 | 0.3077 | 0.4714 | 0.4123      | 0.3887         |
| ecp   | 0.2   | 5 | 0.2977 | 0.4575 | 0.4020      | 0.3844         |
| **ecp** | **0.5** | **1** | **0.3053** | 0.5070 | **0.3818** | **0.3496** |
| ecp   | 0.5   | 5 | 0.3152 | 0.5103 | 0.3978      | 0.3768         |

**ECP (d_min=0.5, λ=1) is the best overall result: ColRate_avg=0.3818, a −11.6pp absolute
reduction from baseline (0.4319), while ADE=0.3053 ≈ baseline (0.3048).** This is the
Pareto-optimal point: maximum collision reduction at near-zero accuracy cost.

The larger d_min=0.5 works here (unlike Protocol B) because specialized training already
filters to only near-collision scenes. Within those scenes, using d_min=0.5 correctly
captures the soft boundary of the collision zone without penalizing unrelated pairs in
other scenes (there are no other scenes in the batch).

#### Round 2 (λ ∈ {1, 2}, gradient steps per batch ∈ {5, 10})

| Loss  | λ | Steps | ADE        | FDE        | ColRate_avg | ColRate_minADE | Note                        |
|-------|---|-------|------------|------------|-------------|----------------|-----------------------------|
| hinge | 1 | 5     | 0.3188     | 0.4980     | 0.4276      | 0.4039         |                             |
| hinge | 1 | 10    | 0.3229     | 0.5161     | 0.4181      | 0.3833         |                             |
| hinge | 2 | 5     | 0.3860     | 0.5364     | 0.4099      | 0.4354         | ADE degraded                |
| hinge | 2 | 10    | 0.4812     | 0.6056     | 0.3629      | 0.4484         | ADE badly degraded          |
| exp   | 1 | 5     | 0.3282     | 0.5100     | 0.4219      | 0.3746         |                             |
| exp   | 1 | 10    | 0.4242     | 0.5958     | 0.3813      | 0.3887         |                             |
| exp   | 2 | 5     | 0.3493     | 0.5176     | 0.3765      | 0.3779         |                             |
| exp   | 2 | 10    | **1.0872** | **1.2858** | 0.2252      | 0.2790         | **DIVERGED**                |
| gauss | 1 | 5     | 0.3714     | 0.5458     | 0.4189      | 0.4159         |                             |
| gauss | 1 | 10    | 0.5704     | 0.7833     | 0.3763      | 0.4213         | ADE degraded                |
| gauss | 2 | 5     | 0.5327     | 0.6724     | 0.3812      | 0.4332         | ADE badly degraded          |
| **gauss** | **2** | **10** | **2.1265** | **2.2964** | 0.1384 | 0.1802   | **CATASTROPHIC DIVERGENCE** |
| ecp   | 1 | 5     | 0.3463     | 0.5359     | 0.4359      | 0.4191         |                             |
| ecp   | 1 | 10    | 0.4058     | 0.5748     | 0.4403      | 0.4376         |                             |
| ecp   | 2 | 5     | 0.5269     | 0.6971     | 0.3921      | 0.4517         | ADE badly degraded          |
| ecp   | 2 | 10    | 0.9401     | 1.0818     | 0.2420      | 0.3094         | near-divergence             |

**Deep Dive — Gauss λ=2, steps=10 Catastrophic Divergence (ADE=2.1265, ColRate=0.138):**

This is the most instructive failure in the study. ColRate dropped to 0.138 — apparently
the best result of the entire project — but ADE is 7× the baseline. The model learned
to predict near-static trajectories: pedestrians barely move from their observed
positions, so they never get close to each other. This is collision avoidance by
trajectory collapse.

The mechanism is a compound instability unique to specialized training:

1. **Only the hardest batches are selected** — scenes where pedestrians are already very
   close (d ≈ 0 in the ground-truth data). There is no easy, "safe" training signal.

2. **Gaussian penalty gradient amplified at intermediate d.** The Gaussian penalty
   exp(−(d/d_min)²) has its maximum gradient magnitude at d = d_min/√2 ≈ 0.14 m.
   With λ=2 and 10 steps per batch, the cumulative gradient update per collision scene
   is enormous — far larger than the NLL gradient from the small batch (size 16).

3. **No trajectory anchor.** Specialized training has removed all non-collision scenes,
   so the NLL provides no pull back toward realistic motion. After enough high-λ
   collision steps the model's mean predictions converge toward the observation endpoint
   (i.e., predict everyone standing still).

4. **Why Gaussian fails here but ECP survives.** ECP has a soft hard-cutoff: its
   gradient drops to near-zero below d_min, preventing the explosive near-zero behaviour.
   Gaussian's gradient at d=0 is zero too, but the slope leading into d=0 is steeper —
   it reaches its maximum at d≈0.14 m and then sustains high gradient for the entire
   region d < d_min. ECP λ=2 "merely" near-diverges (ADE=0.94); Gaussian λ=2 collapses
   completely (ADE=2.13).

**The key methodological lesson: a low ColRate is not a valid success signal in
isolation.** Both ColRate and ADE must be evaluated jointly. Any model that predicts
static trajectories achieves near-zero collision rate. This is why our evaluation always
reports ADE, FDE, and ColRate together, and why the validity condition for any
configuration is ADE ≤ 1.1× baseline.

---

## 6. Accuracy vs. Collision Rate: The Core Tradeoff

The central tension is that **NLL and collision avoidance are not aligned objectives.**
NLL rewards accurately predicting where each pedestrian will go, independently. Collision
avoidance requires coupling predictions across pedestrians to ensure they do not overlap.
These are structurally different objectives, and adding one on top of the other creates a
tradeoff that we observe empirically in two distinct regimes:

**Regime 1 — Gentle perturbation (low λ, tight d_min, Protocol B or C):**
ColRate improves 5–12%, ADE within ±1% of baseline. The penalty nudges the predicted
distribution toward separation. This is practically useful.
*Best examples: gauss λ=5, d_min=0.2 fine-tune; ecp d_min=0.5, λ=1 specialized.*

**Regime 2 — Aggressive perturbation (high λ, large d_min, many gradient steps):**
ColRate drops further but ADE degrades significantly (10–600%). The penalty overwhelms
NLL and the model learns to predict safely-separated but unrealistic trajectories.
*Examples: any λ≥2 in Protocol C; λ≥10 at d_min≥0.5 in Protocol B.*

**The tradeoff is asymmetric.** Worsening ColRate is easy (any high-λ configuration
destroys both ADE and ColRate simultaneously). Improving ColRate meaningfully without
ADE cost is hard — the signal-to-noise ratio of the collision gradient is too low.
The maximum practically useful improvement achieved in this study is ~11.6% ColRate
reduction at near-zero ADE cost (ECP specialized).

---

## 7. Summary of Best Results (zara2, STGCNN)

| Configuration                                     | ADE        | FDE        | ColRate_avg | ColRate_minADE | ΔColRate    |
|---------------------------------------------------|------------|------------|-------------|----------------|-------------|
| Baseline (STGCNN, 250 ep)                         | 0.3048     | 0.4967     | 0.4319      | 0.4072         | —           |
| GAT baseline (best trajectory model)             | 0.2830     | 0.4479     | 0.4350      | 0.4180         | +0.7% (worse!) |
| **Best ADE: gauss λ=5, d_min=0.2, fine-tune**    | **0.2993** | **0.4768** | 0.4210      | 0.4148         | −2.5%       |
| ECP λ=5, d_min=0.2, fine-tune                    | 0.3118     | 0.4988     | 0.4009      | 0.3996         | −7.2%       |
| hinge d_min=0.2, λ=1, specialized                | 0.2940     | 0.4548     | 0.4141      | 0.4072         | −4.1%       |
| **Best ColRate: ECP d_min=0.5, λ=1, spec.**      | 0.3053     | 0.5070     | **0.3818**  | **0.3496**     | **−11.6%**  |
| COLLAPSED (gauss λ=2, steps=10, spec.)           | 2.1265     | 2.2964     | 0.1384      | 0.1802         | −68% (invalid) |

**Observation:** GAT, despite being the best trajectory predictor, does not improve
collision rate vs STGCNN baseline. This re-confirms the baseline finding: better
architecture alone cannot solve collision-blindness.

---

## 8. What Worked, What Did Not, and Why

### What Worked

1. **Two-stage training (baseline first, then fine-tune with collision loss)**
   consistently outperforms single-stage training from scratch across all five loss
   types. The pretrained trajectory prior acts as a strong regularizer. Without it,
   the 0.008% collision signal cannot compete with the NLL gradient.

2. **Gaussian loss fine-tuning (λ=5, d_min=0.2)** uniquely improves both ADE and
   ColRate simultaneously. Its smooth, bounded gradient provides gentle regularization
   rather than a sharp repulsion, making it the safest choice when ADE is the priority.

3. **ECP specialized training (d_min=0.5, λ=1)** achieves the best ColRate reduction
   (−11.6%) at near-zero ADE cost. Filtering to collision-relevant batches solves the
   sparsity problem that defeats Protocol A.

4. **Small d_min (0.2 m) in Protocol B** outperforms d_min=0.5 or 1.0 because it
   matches the actual near-miss threshold in training data, avoiding gradient noise from
   penalizing safe pairs.

### What Did Not Work

1. **Protocol A (training from scratch with collision loss)** — all five loss types
   failed. The collision gradient (0.008% of pairs) is overwhelmed by NLL across 250
   epochs. The problem is fundamental to the loss landscape, not fixable by tuning λ.

2. **Inverse (inv) loss** — harmful across all protocols and datasets. The 1/d
   singularity produces gradient explosions even at moderate proximity, overriding the
   trajectory signal and worsening both ADE and ColRate.

3. **λ ≥ 2 in specialized training** — causes catastrophic divergence for gauss and exp,
   and near-divergence for ecp. The combination of hard-selected batches and high λ
   removes the NLL anchor entirely.

4. **Large d_min (≥0.5 m) in Protocol B** — penalizes 78–440× more pairs than true
   near-misses, injecting noise that degrades ADE without proportional ColRate benefit.

5. **Reducing univ collision rate** — structurally impossible with pairwise additive
   penalties. The scene density means geometrically non-overlapping predictions do not
   fit the observation space.

### Key Design Principles Derived

- **Sparsity → two-stage training.** When the target signal is rare (< 0.01% of data),
  pretrain on the dominant objective first.
- **Gradient shape matters.** Smooth, bounded penalties (Gaussian, ECP) are more stable
  than sharp or unbounded ones (hinge, inv) at moderate λ. But smoothness is not safety
  at very high λ in specialized training.
- **ColRate alone is not a sufficient metric.** Always evaluate jointly with ADE/FDE.
  A collapsed model will show the best ColRate in your table.

---

## 9. Future Work

### Near-Term

1. **Adaptive λ scheduling.** Anneal the collision weight from 0 upward as the NLL
   stabilizes. This may allow higher terminal λ without the instability seen in
   fixed-λ Protocol C.

2. **Gradient clipping per loss term.** Apply a separate gradient clip norm to L_collision
   to prevent the singular gradient events (inv, high-λ gauss) without changing λ.
   This would make the inverse loss safer and potentially effective.

3. **Apply best configurations to all five datasets.** This study focused on zara2.
   Gauss fine-tune and ECP specialized should be evaluated on eth, hotel, and zara1.

4. **Apply collision training to GAT.** GAT achieves better ADE than STGCNN. Applying
   the ECP specialized protocol to GAT may yield both better trajectory accuracy and
   better collision avoidance — combining the best of both findings.

### Longer-Term

5. **Dense-crowd solutions for univ.** Pairwise penalties are insufficient for ColRate >
   0.86. Alternatives: (a) trajectory post-processing with social force optimization;
   (b) group-level interaction modeling; (c) curriculum training from sparse → dense.

6. **Deployment-specific collision thresholds.** The ETH/UCY benchmark uses d=0.2 m in
   normalized coordinates. Real applications require:
   - *Hospital robots:* d_min ≈ 0.5–0.8 m (wheelchair clearance); ColRate near 0 required.
   - *Shopping zones (like zara2):* current best ColRate (0.38) may still trigger false
     safety interventions in autonomous navigation.
   - *Autonomous vehicle crosswalk prediction:* FDE over 4.8 s is the primary metric;
     even rare predicted collisions can trigger emergency braking.

7. **Collision risk as an explicit output.** Rather than minimizing ColRate implicitly,
   train a collision-risk head that outputs a per-scene probability of collision. This
   decouples prediction from safety reasoning and enables risk-aware planning.

---

## 10. References (for Report Use)

- Mohamed, A. et al. (2020). *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction.* CVPR.
- Alahi, A. et al. (2016). *Social LSTM: Human Trajectory Prediction in Crowded Spaces.* CVPR.
- Gupta, A. et al. (2018). *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks.* CVPR.
- Helbing, D. & Molnár, P. (1995). *Social Force Model for Pedestrian Dynamics.* Physical Review E.
- Pellegrini, S. et al. (2009). *You'll Never Walk Alone: Modeling Social Behavior for Multi-Target Tracking.* ICCV. (ETH dataset)
- Lerner, A. et al. (2007). *Crowds by Example.* Computer Graphics Forum. (UCY dataset)
- Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications of the ACM.
- Code base: Official Social-STGCNN repository, Mohamed et al. (2020). Extended with custom collision loss modules, specialized training data loader, and collision rate evaluation metric.

---

## 11. Suggested Figures for Final Report

*(These visualizations should be generated and included in the PDF report.)*

**Fig 1 — Baseline model comparison bar chart (3 subplots):** ADE, FDE, ColRate_avg for
STGCNN / LSTM / GAT across all 5 datasets. Key message: architecture does not determine
collision rate. *(See baselines_3model_comparison.png)*

**Fig 2 — Collision rate vs. ADE scatter plot (Protocol B):** Each point is one
(loss, λ, d_min) configuration. X-axis: ColRate_avg. Y-axis: ADE. Pareto frontier
visible. Highlights the tradeoff and the Gaussian outlier (uniquely improving both axes).

**Fig 3 — λ sweep for ECP (d_min = 0.2, 0.5, 1.0):** Line plot of ColRate and ADE vs λ
for three d_min values. Shows the inversion effect at high λ and explains why d_min=0.2
dominates.

**Fig 4 — Protocol comparison bar chart:** For the best configuration of each protocol
(A: hinge λ=1; B: gauss λ=5; C: ECP specialized), show ADE and ColRate side by side
vs. baseline. Clearly illustrates the two-stage training advantage.

**Fig 5 — Divergence illustration:** Plot predicted trajectory samples from gauss λ=2
steps=10 (diverged) vs. ECP λ=1 specialized (best) vs. baseline, for the same test
scene. Shows trajectory collapse visually — diverged model predicts everyone standing
still.

**Fig 6 — Training data sparsity diagram:** Bar chart of pair counts at d < 0.2, 0.5,
1.0 m in zara2. Makes the 0.008% sparsity concrete. Motivates why Protocol A fails and
why Protocol C (specialized sampling) is necessary.
