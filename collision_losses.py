"""Collision-aware auxiliary losses for Social-STGCNN.

Five formulations operating on the predicted bivariate-Gaussian output
of Social-STGCNN. The first four act on predicted means only; ECP acts
on the full distribution via reparameterized Monte-Carlo sampling.

Shape conventions
-----------------
V_pred    : (T, N, 5)  bivariate Gaussian params per (timestep, ped):
            [mu_x, mu_y, sigma_x_raw, sigma_y_raw, rho_raw]
            where sigma = exp(raw) and rho = tanh(raw).
            Means are RELATIVE displacements (delta per frame).
start_pos : (N, 2)     last observed ABSOLUTE position for each ped.

All losses return a scalar tensor that can be added to L_NLL:
    L_total = L_NLL + lambda_col * collision_loss(...)
"""
import torch


def _rel_means_to_abs(mu_rel, start_pos):
    """Convert relative-displacement means to absolute positions.
    mu_rel: (T, N, 2), start_pos: (N, 2) -> (T, N, 2)"""
    return torch.cumsum(mu_rel, dim=0) + start_pos.unsqueeze(0)


def _pair_distances(abs_pos, eps=1e-8):
    """Upper-triangular pairwise distances per timestep.
    abs_pos: (T, N, 2) -> (T, P) with P = N*(N-1)/2."""
    T, N, _ = abs_pos.shape
    iu = torch.triu_indices(N, N, offset=1, device=abs_pos.device)
    a = abs_pos[:, iu[0], :]
    b = abs_pos[:, iu[1], :]
    return torch.sqrt(((a - b) ** 2).sum(-1) + eps)


def _zero(V_pred):
    return torch.zeros((), device=V_pred.device, dtype=V_pred.dtype)


def hinge_collision_loss(V_pred, start_pos, d_min=0.2):
    """L = mean over (t, pairs) of max(0, d_min - d)^2.

    Hard margin — activates only when d < d_min.
    """
    N = V_pred.shape[1]
    if N < 2:
        return _zero(V_pred)
    abs_pos = _rel_means_to_abs(V_pred[..., :2], start_pos)
    d = _pair_distances(abs_pos)
    return torch.clamp(d_min - d, min=0.0).pow(2).mean()


def exp_collision_loss(V_pred, start_pos, alpha=5.0):
    """L = mean over (t, pairs) of exp(-alpha * d).

    Smooth, non-zero at all distances. alpha ~ 1/d_min is a reasonable default.
    """
    N = V_pred.shape[1]
    if N < 2:
        return _zero(V_pred)
    abs_pos = _rel_means_to_abs(V_pred[..., :2], start_pos)
    d = _pair_distances(abs_pos)
    return torch.exp(-alpha * d).mean()


def inv_collision_loss(V_pred, start_pos, eps=0.1):
    """L = mean over (t, pairs) of 1 / (d + eps).

    Mirrors the 1/||.|| adjacency kernel used inside Social-STGCNN itself.
    """
    N = V_pred.shape[1]
    if N < 2:
        return _zero(V_pred)
    abs_pos = _rel_means_to_abs(V_pred[..., :2], start_pos)
    d = _pair_distances(abs_pos)
    return (1.0 / (d + eps)).mean()


def gauss_collision_loss(V_pred, start_pos, sigma=0.3):
    """L = mean over (t, pairs) of exp(-d^2 / (2*sigma^2)).

    Physics-inspired repulsive potential. sigma sets the effective radius.
    """
    N = V_pred.shape[1]
    if N < 2:
        return _zero(V_pred)
    abs_pos = _rel_means_to_abs(V_pred[..., :2], start_pos)
    d = _pair_distances(abs_pos)
    return torch.exp(-(d ** 2) / (2.0 * sigma * sigma)).mean()


def ecp_collision_loss(V_pred, start_pos, d_min=0.2, K=10, beta=10.0):
    """Expected Collision Probability via reparameterized sampling.

    Acts on the full predicted distribution (means AND variances), unlike
    the other four losses. Uses K Monte-Carlo samples per (t, ped) and a
    soft collision indicator for differentiability.

    L ~= E_samples[ sigmoid(beta * (d_min - d)) ]  averaged over (t, pairs).
    """
    T, N, _ = V_pred.shape
    if N < 2:
        return _zero(V_pred)

    mu = V_pred[..., :2]                            # (T, N, 2)
    sx = torch.exp(V_pred[..., 2])                  # (T, N)
    sy = torch.exp(V_pred[..., 3])                  # (T, N)
    rho = torch.tanh(V_pred[..., 4]).clamp(-0.99, 0.99)  # (T, N)

    # Reparameterized bivariate normal sampling (Cholesky-free form):
    #   x = mu_x + sx * eps1
    #   y = mu_y + sy * (rho * eps1 + sqrt(1 - rho^2) * eps2)
    eps1 = torch.randn(K, T, N, device=V_pred.device, dtype=V_pred.dtype)
    eps2 = torch.randn(K, T, N, device=V_pred.device, dtype=V_pred.dtype)
    sqrt_term = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-6))

    dx = sx.unsqueeze(0) * eps1
    dy = sy.unsqueeze(0) * (rho.unsqueeze(0) * eps1 + sqrt_term.unsqueeze(0) * eps2)
    rel = torch.stack([mu[..., 0].unsqueeze(0) + dx,
                       mu[..., 1].unsqueeze(0) + dy], dim=-1)   # (K, T, N, 2)

    # Relative -> absolute (cumsum along time axis per sample).
    abs_pos = torch.cumsum(rel, dim=1) + start_pos.view(1, 1, N, 2)   # (K, T, N, 2)

    # Pairwise distances per sample, timestep, pair.
    iu = torch.triu_indices(N, N, offset=1, device=V_pred.device)
    a = abs_pos[:, :, iu[0], :]
    b = abs_pos[:, :, iu[1], :]
    d = torch.sqrt(((a - b) ** 2).sum(-1) + 1e-8)            # (K, T, P)

    soft_collision = torch.sigmoid(beta * (d_min - d))
    return soft_collision.mean()


LOSS_REGISTRY = {
    'hinge': hinge_collision_loss,
    'exp':   exp_collision_loss,
    'inv':   inv_collision_loss,
    'gauss': gauss_collision_loss,
    'ecp':   ecp_collision_loss,
}


def collision_loss(name, V_pred, start_pos, **kwargs):
    """Dispatch helper. name in {hinge, exp, inv, gauss, ecp, none}."""
    if name is None or name == 'none':
        return _zero(V_pred)
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown collision loss '{name}'. "
                         f"Options: {list(LOSS_REGISTRY)} or 'none'.")
    return LOSS_REGISTRY[name](V_pred, start_pos, **kwargs)
