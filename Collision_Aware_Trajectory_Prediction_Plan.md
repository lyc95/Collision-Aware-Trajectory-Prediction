# Collision-Aware Trajectory Prediction
## Revised Project Plan — Deep Learning Group Assignment

---

## 1. Project title

**Collision-Aware Trajectory Prediction: Quantifying and Mitigating Collision Failures in Graph-Based Social Models**

---

## 2. Motivation

Current SOTA trajectory prediction models (Social-STGCNN, Social-GAN, Social-BiGAT, Trajectron++) are evaluated almost exclusively on ADE/FDE metrics — measuring geometric closeness to ground truth. For safety-critical applications (autonomous vehicles, social robots, crowd simulation), a model that achieves low ADE but produces collision-containing predictions is dangerous.

Social-STGCNN (CVPR 2020) acknowledges this in its own qualitative analysis (Figure 4), where sampled trajectories visibly intersect. Prior collision-aware work exists (e.g., Trajectron++ uses dynamics constraints, Social-GAN mentions social acceptability informally), but **no systematic study compares collision-avoidance loss formulations** under a common base model and evaluation protocol.

**Our contribution:** A controlled comparison of 5 collision-aware loss formulations on top of Social-STGCNN, producing a Pareto frontier of the ADE/FDE ↔ collision-rate trade-off.

---

## 3. Architecture overview

We build on Social-STGCNN (7.6K parameters, 2ms inference, open-source PyTorch code) and add a collision-aware training objective. The architecture is unchanged — only the loss function is modified.

### 3.1 Base model: Social-STGCNN (unchanged)

```
Input: Observed trajectories (2, T_obs=8, N pedestrians)
                    ↓
    Spatio-temporal graph construction
    - Vertices V: pedestrian (x,y) positions
    - Edges A: kernel function 1/‖v_i - v_j‖₂
    - Adjacency normalized: Λ^(-1/2) Â Λ^(-1/2)
                    ↓
    ST-GCNN (1 layer)
    - Graph convolution: σ(Λ^(-1/2) Â Λ^(-1/2) V^(l) W^(l))
    - Output: embedding V̄ of shape (P̂, T_obs, N)
                    ↓
    TXP-CNN (5 layers)
    - Layer 1: Conv1D(8→12), no residual
    - Layers 2-5: Conv1D(12→12) + residual
                    ↓
    Output: Bivariate Gaussian parameters
    (μx, μy, σx, σy, ρ) per pedestrian per future frame
    Shape: (5, T_pred=12, N)
```

### 3.2 Our contribution: collision-aware loss

```
Original training:
    L = L_NLL

Our training:
    L_total = L_NLL + λ · L_collision
```

We systematically compare **five** formulations (Section 5). One of them — Expected Collision Probability (ECP) — operates on the predicted *distribution* rather than just the mean, which is critical because the evaluation samples from that distribution.

### 3.3 Collision rate metric (for evaluation)

For each test scene:
1. Generate 20 sample trajectories from the predicted Gaussian per pedestrian
2. For each sample index k ∈ [1..20]: check all pairs (i, j) at all future time steps t
3. A sample has a collision iff any pair satisfies ‖p̂_t^i - p̂_t^j‖ < d_col
4. **We report three collision metrics** (Section 6.3) since each captures a different deployment regime.

**Crucial distinction — two thresholds:**
- `d_col` = collision threshold used in the **metric** (physical). Fixed to **0.2m** (≈ body radius overlap, unambiguous collision).
- `d_min` = safety threshold used in the **loss** (conservative, push-apart radius). Swept in Phase 3.

These serve different purposes and must not share a symbol in the report.

---

## 4. Baselines

| Model | Source | Role |
|---|---|---|
| Social-STGCNN (vanilla) | Official GitHub | Primary baseline; model we modify |
| Social-STGCNN + **post-hoc filtering** | Our implementation | **Strong baseline.** Sample 100 trajectories, reject any with collisions, report best-of-remaining ADE |
| Social-GAN | Official GitHub | Secondary baseline (collision-rate comparison only — not re-trained) |
| Constant-velocity | ~10 lines of code | Trivial reference |
| Social-BiGAT | No official code | Cite paper numbers only |

The post-hoc filter is the baseline we *must* beat. If collision-aware training cannot outperform simple rejection sampling, the training cost isn't justified.

---

## 5. Five collision loss formulations

All losses operate on predicted positions **at the future time steps** (T_pred = 12 frames). For a scene with N pedestrians, let μ̂_t ∈ ℝ^{N×2} be the predicted mean positions at time t.

### 5.1 Hinge loss (hard margin)

```
L_hinge = Σ_{t, i<j} max(0, d_min - ‖μ̂_t^i - μ̂_t^j‖)²
```

- Only activates when distance < d_min → zero gradient when agents are far apart
- Sharp boundary; simple and interpretable

### 5.2 Exponential decay loss

```
L_exp = Σ_{t, i<j} exp(-α · ‖μ̂_t^i - μ̂_t^j‖)
```

- Penalty at all distances; exponentially stronger when close
- Smooth gradients everywhere
- α tuned so L_exp ≈ 0.1 at d = d_min (matches hinge activation)

### 5.3 Inverse distance loss

```
L_inv = Σ_{t, i<j} 1 / (‖μ̂_t^i - μ̂_t^j‖ + ε)
```

- Mirrors Social-STGCNN's own adjacency kernel
- Strong penalty when close; gentle when far

### 5.4 Gaussian potential loss

```
L_gauss = Σ_{t, i<j} exp(-‖μ̂_t^i - μ̂_t^j‖² / (2σ²))
```

- Physics-inspired repulsive potential field
- σ controls effective radius of influence

### 5.5 Expected collision probability (ECP) — distribution-level

The prior four losses act only on predicted means. But the model outputs a Gaussian, and the metric samples from it. If the model learns to push means apart while keeping σ large, samples will still collide and the loss provides no gradient on σ.

ECP addresses this directly:

```
L_ecp ≈ Σ_{t, i<j} P(‖p̂_t^i - p̂_t^j‖ < d_min)
```

We estimate this via **reparameterized Monte Carlo** with K=10 samples per pair:

```
For k = 1..K:
    ε_i, ε_j ~ N(0, I)
    p̂_t^i[k] = μ̂_t^i + L_t^i · ε_i       (Cholesky of Σ_t^i)
    p̂_t^j[k] = μ̂_t^j + L_t^j · ε_j
collision_prob ≈ mean_k[ sigmoid(β · (d_min - ‖p̂_t^i[k] - p̂_t^j[k]‖)) ]
```

- Fully differentiable through means **and variances**
- β = 10 (soft indicator); tune if needed
- Training cost: ~1.3× slower than the other four due to K=10 samples

**Hypothesis:** ECP will reduce evaluation collision rate most effectively because it matches the evaluation process directly.

### Implementation notes

- Pairwise distances computed via `torch.cdist(μ, μ)` — O(N²) per timestep, cheap for ETH/UCY (N ≤ 60)
- Use upper-triangular mask to avoid double-counting and diagonal
- Mask out scenes with N < 2 (no pairs to compute loss)
- Normalize by number of pairs to keep λ scale-invariant across scenes

---

## 6. Experimental design

### 6.1 Phased exploration (30 runs total)

**Phase 1 — select best loss form** (10 runs)
- Fix: d_min = 0.4m, λ = 1.0, uniform time weighting
- Vary: 5 loss formulations × 2 scenes (**ETH** sparse, **UNIV** dense)
- Rationale: ETH alone is misleading — crowded scenes have different loss dynamics
- Select: top 2 formulations for Phase 2

**Phase 2 — 2D sweep on winning losses** (16 runs)
- Fix: top 2 loss forms from Phase 1
- Vary: λ ∈ {0.1, 1.0, 10.0} × d_min ∈ {0.3, 0.6, 0.9}m, but reduced to a diagonal/L-shaped grid to fit budget
- Run on all 5 scenes (leave-one-out)
- Output: **Pareto frontier** (ADE vs collision rate) — central figure of the report

**Phase 3 — seed stability** (remaining budget)
- Re-run top 2 configurations with 3 random seeds
- Report mean ± std for all primary results

**Dropped from scope** (time-weighting, α/σ tuning, 6-point λ sweep): mention as future work.

### 6.2 Compute budget

- 30 training runs × ~45 min/run on GPU = ~22 hours
- Plus inference/evaluation: ~5 hours
- Plus buffer: ~10 hours
- **Total: ~37 hours** — achievable in 4 weeks with 1 GPU

### 6.3 Metrics reported

| Metric | Definition | Purpose |
|---|---|---|
| **minADE_20** | Min L2 error across 20 samples, averaged over frames | Best-case trajectory accuracy (standard in field) |
| **minFDE_20** | Min L2 error at final frame across 20 samples | Long-horizon accuracy (standard) |
| **ColRate_avg** | Fraction of 20 samples containing any collision | Overall distribution safety |
| **ColRate_minADE** | Whether the sample that achieved minADE_20 contains a collision | Deployment-relevant: users ship the best-ADE sample |

All numeric results reported as mean ± std over 3 seeds.

---

## 7. Four-week timeline

### Week 1 — Baseline reproduction + metric implementation

**Tasks:**
- Set up environment (Python 3.9, PyTorch w/ CUDA); fix syntax/compat issues in upstream repo
- Reproduce reported ADE/FDE on all 5 ETH/UCY scenes (Social-STGCNN)
- Implement `collision_rate()` metric function
- Implement constant-velocity baseline (~10 lines)
- Implement **post-hoc filtering baseline** on Social-STGCNN
- Measure baseline collision rates; fill Table 1

**Deliverables:** Working reproduction + Table 1 (baselines across 5 scenes × 4 metrics).

**Key early finding:** Baseline collision rates (expected 5–20%).

### Week 2 — Implement losses + Phase 1

**Tasks:**
- Implement 5 loss functions (`collision_losses.py`)
- Modify training loop: `L_total = L_NLL + λ · L_collision`
- Run Phase 1 (5 losses × 2 scenes = 10 runs)
- Compare, select top 2 for Phase 2
- Begin Social-GAN reproduction (for collision-rate reference only; do not retrain)

**Deliverables:** 5 loss implementations, Phase 1 comparison table, decision memo on which 2 losses advance.

### Week 3 — Phase 2 sweep + Pareto frontier

**Tasks:**
- Run Phase 2 (16 runs across λ, d_min, scenes)
- Run Phase 3 seed stability on top 2 configs
- Plot Pareto frontier (5 scenes; one curve per scene, averaged curve as main figure)
- Ablation tables

**Deliverables:** Complete experimental data, Pareto plot, ablation tables.

### Week 4 — Qualitative analysis + report

**Tasks:**
- Visualize 5–10 scenes: baseline vs post-hoc filter vs our model
- Failure-case analysis (when does collision loss still fail?)
- Write final report (~8 pages)
- Prepare presentation

**Deliverables:** Final report, slides, open-source code (modified Social-STGCNN + losses + metric).

---

## 8. Report structure

1. **Introduction** — SOTA models achieve low ADE/FDE but ignore collisions
2. **Related work** — Social-LSTM → Social-GAN → SoPhie → Social-BiGAT → Social-STGCNN → Trajectron++; prior collision-aware attempts and what they missed
3. **Background** — Social-STGCNN architecture summary
4. **Observation: baseline collision rates** (Finding 1)
5. **Method** — 5 collision loss formulations + collision-rate metric + post-hoc filter baseline
6. **Experiments**
   - 6.1 Phase 1 loss function comparison (Finding 2)
   - 6.2 λ × d_min sweep and Pareto frontier (Finding 3)
   - 6.3 Comparison vs post-hoc filter (Finding 4)
   - 6.4 Qualitative analysis and failure cases
7. **Discussion and limitations** — including distribution-vs-mean gap, ground-truth near-misses, scene-density dependence
8. **Conclusion**

---

## 9. Expected findings

1. **Baseline SOTA models produce collisions** at 5–20% across scenes, despite strong ADE/FDE.
2. **ECP (distribution-level) loss outperforms mean-only losses** on collision rate because it aligns training with the sampling-based metric.
3. **A meaningful Pareto trade-off exists** between minADE and ColRate. Modest λ values substantially reduce collisions with small ADE cost.
4. **Collision-aware training beats post-hoc filtering** when λ is tuned — demonstrating training-time learning > rejection sampling.
5. **Optimal d_min depends on scene density** — dense scenes benefit from smaller d_min; sparse scenes tolerate larger d_min.

---

## 10. Team task allocation (3-person team)

| Member | Primary responsibilities |
|---|---|
| **Member A** — Infrastructure | Baseline reproduction, environment setup, metric implementation, experiment logging (wandb), post-hoc filter baseline |
| **Member B** — Methods | 5 collision loss implementations, training loop modification, Phase 1 execution, loss comparison writeup |
| **Member C** — Analysis | Phase 2/3 execution, Pareto plotting, qualitative visualization, failure-case analysis |
| **All** | Week 4 report writing and presentation |

Work is modular — any member can pick up another's tasks mid-project.

---

## 11. Tools and infrastructure

- **Language:** Python 3.9, PyTorch ≥2.0 (with CUDA)
- **GPU:** Local (RTX-class) or Google Colab Pro
- **Experiment tracking:** Weights & Biases (free tier) — logs ADE/FDE/ColRate per epoch
- **Version control:** GitHub private repo; branch per member
- **Data:** ETH/UCY (bundled with Social-STGCNN repo)
- **Baseline code:**
  - Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN
  - Social-GAN: https://github.com/agrimgupta92/sgan

---

## 12. Risk mitigation

| Risk | Mitigation |
|---|---|
| Baseline code doesn't reproduce paper numbers | 2-day buffer in Week 1; check GitHub issues; accept ±5% deviation |
| Collision loss makes ADE much worse at all λ | Expected — that's why we sweep λ. Falling back to post-hoc filter is still a valid negative result |
| ECP numerically unstable (Cholesky on near-singular Σ) | Add σ² floor (1e-4); clamp correlation to ±0.99 |
| GPU budget overrun | Drop Phase 3 seeds or reduce to 3 scenes (ETH + UNIV + ZARA1) |
| Social-GAN retraining fails | Already scoped as reference-only; no retraining needed |
| Team member unavailable | Tasks are modular; pair-program checkpoints at end of each week |
| Ground-truth collisions in dataset pollute loss | Detect and mask frames where GT pedestrians are already <0.2m apart |

---

## 13. Key references

1. Mohamed et al., *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction*, CVPR 2020.
2. Kosaraju et al., *Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks*, NeurIPS 2019.
3. Gupta et al., *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks*, CVPR 2018.
4. Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*, CVPR 2016.
5. Salzmann et al., *Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data*, ECCV 2020.
6. Mangalam et al., *It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction* (PECNet), ECCV 2020.
7. Sadeghian et al., *SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints*, CVPR 2019.
8. Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
9. Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018.
10. Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014 — reparameterization trick used in ECP loss.
