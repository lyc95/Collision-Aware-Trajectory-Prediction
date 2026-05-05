#!/usr/bin/env python3
"""
Mock collision scenario demo for Social-STGCNN.

Generates synthetic head-on collision scenarios with 2, 3, and 4 agents,
then compares predicted trajectories from the baseline vs. collision-aware
models (hinge loss, ECP loss).  All models used here were trained on the
'univ' dataset.

Output: mock_collision_demo.png  (3 rows × 3 columns grid)
"""

import os, sys, pickle
import numpy as np
import torch
import torch.distributions as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import social_stgcnn
from utils import seq_to_graph

# ── Constants ──────────────────────────────────────────────────────────────────
T_OBS   = 8
T_PRED  = 12
SPEED   = 0.15   # m/frame ≈ 0.375 m/s — collision lands at pred frame ~5, not frame 0
D_COL   = 0.2    # collision radius (m)
K       = 20     # prediction samples
SEED    = 42

AGENT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

MODEL_DIRS = {
    'Baseline':  './baseline/social-stgcnn-univ',
    'Hinge':     './hinge/social-stgcnn-univ',
    'Exp':       './exp/social-stgcnn-univ',
    'Inv':       './inv/social-stgcnn-univ',
    'Gauss':     './gauss/social-stgcnn-univ',
    'ECP':       './ecp/social-stgcnn-univ',
}
N_AGENTS_LIST = [2, 3, 4]


# ── Scenario generators ────────────────────────────────────────────────────────

def make_scenario(n_agents, noise_std=0.015, seed=SEED):
    """
    Return obs_abs (N, 2, T_OBS) and gt_abs (N, 2, T_PRED).

    All agents converge toward the origin.  The collision occurs at
    prediction frame 8 (≈ 3.2 s into the future).
    """
    rng = np.random.default_rng(seed)
    s = SPEED
    configs = {
        2: dict(
            starts=[(-2.0,  0.0), ( 2.0,  0.0)],
            vels  =[( s,    0.0), (-s,    0.0)],
        ),
        3: dict(
            starts=[(-2.0,  0.0), ( 2.0,  0.0), ( 0.0,  2.0)],
            vels  =[( s,    0.0), (-s,    0.0), ( 0.0,  -s)],
        ),
        4: dict(
            starts=[(-2.0,  0.0), ( 2.0,  0.0), ( 0.0, -2.0), ( 0.0,  2.0)],
            vels  =[( s,    0.0), (-s,    0.0), ( 0.0,   s),  ( 0.0,  -s)],
        ),
    }
    cfg = configs[n_agents]
    T = T_OBS + T_PRED
    trajs = []
    for (sx, sy), (vx, vy) in zip(cfg['starts'], cfg['vels']):
        t = np.arange(T, dtype=float)
        xs = sx + t * vx + rng.standard_normal(T) * noise_std
        ys = sy + t * vy + rng.standard_normal(T) * noise_std
        trajs.append(np.stack([xs, ys]))   # (2, T)
    traj_all = np.stack(trajs)             # (N, 2, T)
    return traj_all[:, :, :T_OBS].copy(), traj_all[:, :, T_OBS:].copy()


# ── Data prep ─────────────────────────────────────────────────────────────────

def build_graph_tensors(obs_abs):
    """
    obs_abs : (N, 2, T_OBS)
    Returns V_obs (T_OBS, N, 2) and A_obs (T_OBS, N, N) as float Tensors.
    """
    obs_rel = np.zeros_like(obs_abs)
    obs_rel[:, :, 1:] = obs_abs[:, :, 1:] - obs_abs[:, :, :-1]
    return seq_to_graph(obs_abs, obs_rel)   # (T, N, 2), (T, N, N)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_dir):
    with open(os.path.join(model_dir, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    m = social_stgcnn(
        n_stgcnn   = args.n_stgcnn,
        n_txpcnn   = args.n_txpcnn,
        output_feat= args.output_size,
        seq_len    = args.obs_seq_len,
        kernel_size= args.kernel_size,
        pred_seq_len= args.pred_seq_len,
    )
    m.load_state_dict(torch.load(
        os.path.join(model_dir, 'val_best.pth'), map_location='cpu'))
    m.eval()
    return m


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, V_obs, A_obs, obs_abs, seed=SEED):
    """
    V_obs   : (T_OBS, N, 2) Tensor
    A_obs   : (T_OBS, N, N) Tensor
    obs_abs : (N, 2, T_OBS) numpy
    Returns:
      samples  – list of K arrays (T_PRED, N, 2) absolute positions
      mean_abs – (T_PRED, N, 2) array, mean predicted path
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        # (T, N, 2) → (1, 2, T, N)
        V_in = V_obs.unsqueeze(0).permute(0, 3, 1, 2)
        V_raw, _ = model(V_in, A_obs)       # (1, 5, T_PRED, N)
        V_raw = V_raw.permute(0, 2, 3, 1).squeeze(0)  # (T_PRED, N, 5)

    sx   = torch.exp(V_raw[:, :, 2])
    sy   = torch.exp(V_raw[:, :, 3])
    corr = torch.tanh(V_raw[:, :, 4])

    cov = torch.zeros(*V_raw.shape[:2], 2, 2)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean_rel = V_raw[:, :, :2]  # (T_PRED, N, 2)

    mvn = tdist.MultivariateNormal(mean_rel.cpu(), cov.cpu())

    start = obs_abs[:, :, -1]     # (N, 2) — last observed position
    samples = []
    for _ in range(K):
        rel = mvn.sample().numpy()                     # (T_PRED, N, 2)
        samples.append(np.cumsum(rel, axis=0) + start) # absolute

    mean_abs = np.cumsum(mean_rel.numpy(), axis=0) + start
    return samples, mean_abs


def collision_count(samples):
    """Return number of K samples containing a collision (dist < D_COL)."""
    count = 0
    for s in samples:           # s: (T, N, 2)
        T, N, _ = s.shape
        if N < 2:
            continue
        diff = s[:, :, None, :] - s[:, None, :, :]    # (T, N, N, 2)
        dist = np.sqrt((diff**2).sum(-1))               # (T, N, N)
        iu = np.triu_indices(N, k=1)
        if (dist[:, iu[0], iu[1]] < D_COL).any():
            count += 1
    return count


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_panel(ax, obs_abs, gt_abs, samples, mean_abs, n_agents, model_label):
    """Draw one subplot."""
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)

    for n in range(n_agents):
        c = AGENT_COLORS[n]
        ox, oy = obs_abs[n, 0], obs_abs[n, 1]   # (T_OBS,)
        gx = np.concatenate([[ox[-1]], gt_abs[n, 0]])
        gy = np.concatenate([[oy[-1]], gt_abs[n, 1]])

        # Observed (solid)
        ax.plot(ox, oy, color=c, linewidth=2.2, zorder=4)
        ax.plot(ox[0], oy[0], 's', color=c, markersize=5, zorder=5)  # start
        ax.plot(ox[-1], oy[-1], 'o', color=c, markersize=6, zorder=5) # end obs

        # Ground truth future (dashed)
        ax.plot(gx, gy, color=c, linewidth=1.5, linestyle='--', alpha=0.65, zorder=3)

        # K samples (thin, transparent)
        for s in samples:
            px = np.concatenate([[ox[-1]], s[:, n, 0]])
            py = np.concatenate([[oy[-1]], s[:, n, 1]])
            ax.plot(px, py, color=c, linewidth=0.35, alpha=0.18, zorder=2)

        # Mean predicted path (dotted + arrow)
        mx = np.concatenate([[ox[-1]], mean_abs[:, n, 0]])
        my = np.concatenate([[oy[-1]], mean_abs[:, n, 1]])
        ax.plot(mx, my, color=c, linewidth=2.0, linestyle=':', zorder=6)
        ax.annotate('', xy=(mx[-1], my[-1]), xytext=(mx[-2], my[-2]),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5), zorder=7)

    # Collision zone indicator at origin
    circle = plt.Circle((0, 0), D_COL, color='crimson', fill=True,
                         alpha=0.12, zorder=1)
    ax.add_patch(circle)
    circle2 = plt.Circle((0, 0), D_COL, color='crimson', fill=False,
                          linestyle=':', linewidth=1.2, zorder=3)
    ax.add_patch(circle2)

    n_col = collision_count(samples)
    ax.set_title(f'collisions: {n_col}/{K}', fontsize=8, pad=3)
    ax.tick_params(labelsize=7)


def main():
    # ── Load all models ───────────────────────────────────────────────────────
    models = {}
    for label, d in MODEL_DIRS.items():
        print(f"  Loading {label} ({d}) ...", flush=True)
        models[label] = load_model(d)
    print()

    # ── Pre-generate scenarios (identical for all models) ─────────────────────
    scenarios = {n: make_scenario(n) for n in N_AGENTS_LIST}

    # ── Build figure: rows = models, columns = agent counts ──────────────────
    n_rows = len(MODEL_DIRS)
    n_cols = len(N_AGENTS_LIST)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        'Mock Collision Scenarios: Baseline vs Collision-Aware Models\n'
        '(trained on univ dataset)',
        fontsize=13, y=1.002,
    )

    # Column headers (agent counts)
    for col_i, n_agents in enumerate(N_AGENTS_LIST):
        axes[0][col_i].set_title(f'{n_agents} agents', fontsize=11, pad=6)

    for row_i, (label, model) in enumerate(models.items()):
        for col_i, n_agents in enumerate(N_AGENTS_LIST):
            ax = axes[row_i][col_i]
            obs_abs, gt_abs = scenarios[n_agents]
            V_obs, A_obs = build_graph_tensors(obs_abs)
            samples, mean_abs = predict(model, V_obs, A_obs, obs_abs)

            draw_panel(ax, obs_abs, gt_abs, samples, mean_abs, n_agents, label)

            # Row label (left column only)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=11, labelpad=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.2,
               label='Observed trajectory'),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--',
               label='Ground truth (straight-line)'),
        Line2D([0], [0], color='gray', linewidth=0.5, alpha=0.5,
               label=f'Predicted samples (K={K})'),
        Line2D([0], [0], color='gray', linewidth=2.0, linestyle=':',
               label='Mean predicted path'),
        mpatches.Patch(facecolor='crimson', alpha=0.3,
                       label=f'Collision zone (r={D_COL} m)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.025), fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'mock_collision_demo.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
