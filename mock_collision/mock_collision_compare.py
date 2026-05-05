#!/usr/bin/env python3
"""
Mock collision comparison: Baseline vs Specialized-Trained Social-STGCNN.

Visualization strategy
----------------------
For each agent we plot, per prediction timestep, the 1-σ confidence ellipse
derived from the model's own predicted bivariate Gaussian (μ, Σ).  This is
exactly what the model is "saying" about where the agent will be at each
moment in the future — not a sample envelope inflated by variance.

Where two agents' time-matched ellipses overlap = collision risk surface.
The baseline drives the means through the origin so the time-matched ellipses
collide.  The specialized model pushes the means apart so the time-matched
ellipses no longer share space, even if the variance is large.

Models compared:
  - Baseline (zara2, val_best)
  - ECP d_min=2.0, lambda=10, specialized (epoch_final.pth)
      ADE 0.94 / ColRate 0.242  -- ~44% collision reduction with
      still-meaningful trajectories.

Output: mock_collision_compare.png
"""

import os, sys, pickle
import numpy as np
import torch
import torch.distributions as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate, translate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from model import social_stgcnn
from utils import seq_to_graph

# ── Constants ──────────────────────────────────────────────────────────────────
T_OBS  = 8
T_PRED = 12
SPEED  = 0.15     # m/frame ≈ 0.375 m/s — typical walking pace in zara2
D_COL  = 0.2
K      = 20
SEED   = 42
SIGMA  = 2.0      # ellipse scale for visual confidence region (≈86% mass)

AGENT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# (label, model_dir, ckpt_filename, short_tag)
MODELS = [
    ('Baseline (zara2, val_best)\n'
     'ADE 0.30  |  ColRate 0.43',
     './baseline/social-stgcnn-zara2',
     'val_best.pth',
     'baseline'),
    ('HINGE  d=2.0  λ=10  (specialized round 2, val_best)\n'
     'ADE 0.44  |  ColRate 0.37   (-15% vs baseline; moderate swerve)',
     './finetune_specialized2_zara2/social-stgcnn-zara2-hinge-dmin2.0-lam10-spec',
     'val_best.pth',
     'hinge_d20_l10'),
]
# Three scene geometries, all evaluated at a fixed time-to-collision.
# t_col=8 (≈3.2 s out) is the sweet spot: baseline shows clear collisions
# while the prediction window is still long enough for a polite swerve.
SCENE_FAMILIES = ['head_on', 't_cross', 'lane_merge']
FIXED_T_COL    = 8


# ── Scenario generators ────────────────────────────────────────────────────────

def _starts_for_family(family, t_col, s):
    """Return (starts, vels, zones, family_title) for a given family + t_col.

    The geometry is parameterised so that for *every* family, both agents
    arrive at the collision zone (origin) at absolute frame T_OBS + t_col.
    """
    A = s * (T_OBS + t_col)            # backward-walk distance from origin

    if family == 'head_on':
        # Near head-on, ±9 cm lateral offset → 18 cm closest approach.
        offset_y = 0.09
        starts = [(-A,  offset_y), ( A, -offset_y)]
        vels   = [( s,  0.0     ), (-s,  0.0     )]
        zones  = [(0.0, 0.0)]
        title  = 'Head-on  (18 cm offset)'

    elif family == 't_cross':
        # 90° crossing — agent A walks east, agent B walks north.
        # Slight cross-axis offsets keep them just inside the 0.2 m threshold.
        starts = [(-A, 0.10), (0.10, -A)]
        vels   = [( s, 0.0 ), (0.0,   s)]
        zones  = [(0.0, 0.0)]
        title  = 'T-intersection crossing'

    elif family == 'lane_merge':
        # Agent A walks east in the upper lane (y=+0.05).
        # Agent B converges from below at a 20° angle, merging into agent A's
        # lane.  Both arrive at the origin at frame T_OBS + t_col.
        ang = np.deg2rad(20.0)
        Ax  = s * np.cos(ang) * (T_OBS + t_col)
        Ay  = s * np.sin(ang) * (T_OBS + t_col)
        starts = [(-A,  0.05), (-Ax, -Ay)]
        vels   = [( s,  0.0 ), ( s * np.cos(ang),  s * np.sin(ang))]
        zones  = [(0.0, 0.0)]
        title  = 'Lane merge  (20° converge)'

    else:
        raise ValueError(family)

    return starts, vels, zones, title


def make_scenario(family, t_col, noise_std=0.010, seed=SEED):
    """Return (obs_abs, gt_abs, n_agents, col_zones, scene_title)."""
    rng = np.random.default_rng(seed)
    starts, vels, zones, fam_title = _starts_for_family(family, t_col, SPEED)

    n_agents = len(starts)
    T = T_OBS + T_PRED
    trajs = []
    for (sx, sy), (vx, vy) in zip(starts, vels):
        t = np.arange(T, dtype=float)
        xs = sx + t * vx + rng.standard_normal(T) * noise_std
        ys = sy + t * vy + rng.standard_normal(T) * noise_std
        trajs.append(np.stack([xs, ys]))
    traj_all = np.stack(trajs)
    obs = traj_all[:, :, :T_OBS].copy()
    gt  = traj_all[:, :, T_OBS:].copy()

    if t_col == 4:
        ttc_tag = '4 frames  (imminent)'
    elif t_col == 8:
        ttc_tag = '8 frames  (medium)'
    elif t_col == 12:
        ttc_tag = '12 frames  (distant)'
    else:
        ttc_tag = f'{t_col} frames'
    title = f'Time-to-collision: {ttc_tag}'

    return obs, gt, n_agents, zones, title, fam_title


# ── Data prep ──────────────────────────────────────────────────────────────────

def build_graph_tensors(obs_abs):
    obs_rel = np.zeros_like(obs_abs)
    obs_rel[:, :, 1:] = obs_abs[:, :, 1:] - obs_abs[:, :, :-1]
    return seq_to_graph(obs_abs, obs_rel)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_dir, ckpt_filename):
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
    ckpt = os.path.join(model_dir, ckpt_filename)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(model, V_obs, A_obs, obs_abs, seed=SEED):
    """Returns mean_abs, cov_abs (per-timestep absolute covariance), samples."""
    torch.manual_seed(seed)
    with torch.no_grad():
        V_in = V_obs.unsqueeze(0).permute(0, 3, 1, 2)
        V_raw, _ = model(V_in, A_obs)
        V_raw = V_raw.permute(0, 2, 3, 1).squeeze(0)         # (T_PRED, N, 5)

    sx   = torch.exp(V_raw[:, :, 2])
    sy   = torch.exp(V_raw[:, :, 3])
    corr = torch.tanh(V_raw[:, :, 4])
    cov_rel = torch.zeros(*V_raw.shape[:2], 2, 2)
    cov_rel[:, :, 0, 0] = sx * sx
    cov_rel[:, :, 0, 1] = corr * sx * sy
    cov_rel[:, :, 1, 0] = corr * sx * sy
    cov_rel[:, :, 1, 1] = sy * sy
    mean_rel = V_raw[:, :, :2]                                # (T_PRED, N, 2)

    # Cumulative covariance for the absolute (cumsum) prediction:
    #   pos_t = sum_{s=1..t} delta_s   →   Cov(pos_t) = sum_{s=1..t} Cov(delta_s)
    cov_abs = torch.cumsum(cov_rel, dim=0).numpy()            # (T_PRED, N, 2, 2)

    start = obs_abs[:, :, -1]
    mean_abs = np.cumsum(mean_rel.numpy(), axis=0) + start    # (T_PRED, N, 2)

    mvn = tdist.MultivariateNormal(mean_rel.cpu(), cov_rel.cpu())
    samples = []
    for _ in range(K):
        rel = mvn.sample().numpy()
        samples.append(np.cumsum(rel, axis=0) + start)

    return mean_abs, cov_abs, samples


def collision_count(samples):
    count = 0
    for s in samples:
        T, N, _ = s.shape
        if N < 2:
            continue
        diff = s[:, :, None, :] - s[:, None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        iu = np.triu_indices(N, k=1)
        if (dist[:, iu[0], iu[1]] < D_COL).any():
            count += 1
    return count


# ── Confidence ellipse helpers ────────────────────────────────────────────────

def cov_to_ellipse(mean, cov, n_sigma=1.0):
    """Return (cx, cy, w, h, angle_deg) describing the n-σ ellipse."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    order = vals.argsort()[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_sigma * np.sqrt(vals)
    return mean[0], mean[1], w, h, angle


def ellipse_polygon(cx, cy, w, h, angle_deg, n_pts=48):
    """Shapely polygon for an ellipse (for intersection math)."""
    unit = Point(0, 0).buffer(1.0, resolution=n_pts // 4)
    e = scale(unit, xfact=w/2, yfact=h/2)
    e = rotate(e, angle_deg, origin=(0, 0), use_radians=False)
    e = translate(e, xoff=cx, yoff=cy)
    return e


# ── Plot ───────────────────────────────────────────────────────────────────────

def draw_panel(ax, obs_abs, gt_abs, mean_abs, cov_abs, samples, n_agents,
               col_zones, lim=4.0):
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # 1. Per-timestep 1.5-σ confidence ellipses (light fill, agent-coloured)
    ellipse_polys = [[] for _ in range(n_agents)]
    for n in range(n_agents):
        for t in range(T_PRED):
            cx, cy, w, h, ang = cov_to_ellipse(
                mean_abs[t, n], cov_abs[t, n], n_sigma=SIGMA)
            ax.add_patch(Ellipse(
                (cx, cy), w, h, angle=ang,
                facecolor=AGENT_COLORS[n], alpha=0.10,
                edgecolor=AGENT_COLORS[n], linewidth=0.4, zorder=2))
            ellipse_polys[n].append(ellipse_polygon(cx, cy, w, h, ang))

    # 2. Time-aligned overlap of confidence ellipses (the actual collision risk)
    overlap_total = 0.0
    overlap_geoms = []
    for t in range(T_PRED):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                inter = ellipse_polys[i][t].intersection(ellipse_polys[j][t])
                if inter.is_empty:
                    continue
                overlap_total += inter.area
                overlap_geoms.append(inter)

    for g in overlap_geoms:
        geoms = [g] if g.geom_type == 'Polygon' else list(g.geoms)
        for sub in geoms:
            if hasattr(sub, 'exterior'):
                xs, ys = sub.exterior.xy
                ax.fill(xs, ys, color='black', alpha=0.55, zorder=4,
                        hatch='///', edgecolor='black', linewidth=0.5)

    # 3. Endpoint markers (where the model thinks each agent ends up)
    for n in range(n_agents):
        ax.plot(mean_abs[-1, n, 0], mean_abs[-1, n, 1], '*',
                color=AGENT_COLORS[n], markersize=12,
                markeredgecolor='black', markeredgewidth=0.6, zorder=8)

    # 4. Trajectories on top
    for n in range(n_agents):
        c = AGENT_COLORS[n]
        ox, oy = obs_abs[n, 0], obs_abs[n, 1]
        gx = np.concatenate([[ox[-1]], gt_abs[n, 0]])
        gy = np.concatenate([[oy[-1]], gt_abs[n, 1]])

        ax.plot(ox, oy, color=c, linewidth=2.4, zorder=5)
        ax.plot(ox[0],  oy[0],  's', color=c, markersize=6, zorder=6)
        ax.plot(ox[-1], oy[-1], 'o', color=c, markersize=7, zorder=6)
        ax.plot(gx, gy, color=c, linewidth=1.2, linestyle='--',
                alpha=0.55, zorder=4)

        mx = np.concatenate([[ox[-1]], mean_abs[:, n, 0]])
        my = np.concatenate([[oy[-1]], mean_abs[:, n, 1]])
        ax.plot(mx, my, color=c, linewidth=2.0, linestyle=':', zorder=7)

    # 5. Collision zones (one per intended collision point in this scene)
    for (cx, cy) in col_zones:
        ax.add_patch(plt.Circle((cx, cy), D_COL, color='crimson',
                                fill=True, alpha=0.18, zorder=1))
        ax.add_patch(plt.Circle((cx, cy), D_COL, color='crimson',
                                fill=False, linestyle=':', linewidth=1.4,
                                zorder=4))

    # 6. Title
    n_col = collision_count(samples)
    ax.set_title(
        f'collisions: {n_col}/{K}   |   '
        f'overlap area: {overlap_total:.2f} m²',
        fontsize=9, pad=4,
    )
    ax.tick_params(labelsize=7)


def render_combined(models):
    """Render a single figure: 2 models × 3 scene geometries at fixed t_col."""
    scenarios = {fam: make_scenario(fam, FIXED_T_COL) for fam in SCENE_FAMILIES}

    n_rows = len(models)
    n_cols = len(SCENE_FAMILIES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.4 * n_cols, 5.0 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f'Predicted Avoidance Across Geometries — '
        f'Baseline vs Specialized Social-STGCNN  '
        f'(time-to-collision = {FIXED_T_COL} frames ≈ '
        f'{FIXED_T_COL * 0.4:.1f} s)\n'
        f'Light fills = per-step {SIGMA}-σ ellipses from (μ, Σ).   '
        f'Black hatched = time-aligned overlap (collision risk).   '
        f'Goal: keep ellipses overlapping (similar trajectories) '
        f'but eliminate collisions (point-to-point < 0.2 m).',
        fontsize=11, y=1.005,
    )

    # Column headers — the geometry name from each scenario.
    for col_i, fam in enumerate(SCENE_FAMILIES):
        fam_title = scenarios[fam][5]
        axes[0][col_i].annotate(
            fam_title,
            xy=(0.5, 1.08), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    for row_i, (label, model, tag) in enumerate(models):
        for col_i, fam in enumerate(SCENE_FAMILIES):
            ax = axes[row_i][col_i]
            obs_abs, gt_abs, n_agents, zones, _, _ = scenarios[fam]
            V_obs, A_obs = build_graph_tensors(obs_abs)
            mean_abs, cov_abs, samples = predict(model, V_obs, A_obs, obs_abs)
            draw_panel(ax, obs_abs, gt_abs, mean_abs, cov_abs, samples,
                       n_agents, zones)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=10, labelpad=10)

    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.4,
               label='Observed trajectory'),
        Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--',
               label='Ground truth'),
        Line2D([0], [0], color='gray', linewidth=2.0, linestyle=':',
               label='Mean predicted path'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=11, linewidth=0,
               label='Predicted endpoint'),
        mpatches.Patch(facecolor='gray', alpha=0.2, edgecolor='gray',
                       label=f'{SIGMA}-σ confidence ellipses (per agent, per t)'),
        mpatches.Patch(facecolor='black', alpha=0.55, hatch='///',
                       edgecolor='black',
                       label='Time-aligned ellipse overlap (collision risk)'),
        mpatches.Patch(facecolor='crimson', alpha=0.3,
                       label=f'Ground-truth collision zone (r={D_COL} m)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'mock_collision_compare.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    models = []
    for label, d, ckpt, tag in MODELS:
        print(f"  Loading [{tag}] {label.splitlines()[0]}")
        print(f"     dir : {d}")
        print(f"     ckpt: {ckpt}", flush=True)
        models.append((label, load_model(d, ckpt), tag))
    print()

    render_combined(models)


if __name__ == '__main__':
    main()
