"""Additional baseline models for comparison against Social-STGCNN.

Both new models share the Social-STGCNN I/O convention:
    Input:  v of shape (B, input_feat=2, T_obs, N), a of shape (T_obs, N, N)
    Output: (V_pred of shape (B, output_feat=5, T_pred, N), a_passthrough)

Output features are bivariate-Gaussian params per (timestep, ped):
    [mu_x, mu_y, sigma_x_raw, sigma_y_raw, rho_raw]
where sigma = exp(raw) and rho = tanh(raw), matching social_stgcnn.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import social_stgcnn


class social_lstm(nn.Module):
    """Per-pedestrian encoder-decoder LSTM. No social interaction.

    Each pedestrian's observed displacement sequence is encoded independently;
    the decoder autoregressively produces predicted displacements for the
    pred_seq_len horizon. Provides a no-graph baseline.
    """

    def __init__(self, input_feat=2, output_feat=5,
                 hidden_dim=64, num_layers=1,
                 seq_len=8, pred_seq_len=12, dropout=0.0):
        super().__init__()
        self.pred_seq_len = pred_seq_len
        # nn.LSTM dropout only applies between stacked layers
        ld = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_feat, hidden_dim, num_layers,
                               batch_first=True, dropout=ld)
        self.decoder = nn.LSTM(input_feat, hidden_dim, num_layers,
                               batch_first=True, dropout=ld)
        self.out = nn.Linear(hidden_dim, output_feat)

    def forward(self, v, a):
        # v: (B, F=2, T, N) -> (B*N, T, F)
        B, F_in, T, N = v.shape
        x = v.permute(0, 3, 2, 1).reshape(B * N, T, F_in)

        _, (h, c) = self.encoder(x)

        # Autoregressive decoder, seeded with the last observed displacement.
        cur = x[:, -1:, :]
        outs = []
        for _ in range(self.pred_seq_len):
            o, (h, c) = self.decoder(cur, (h, c))
            p = self.out(o)                 # (B*N, 1, output_feat)
            outs.append(p)
            cur = p[..., :2]                # next input = predicted Δ-mean

        out = torch.cat(outs, dim=1)        # (B*N, T_pred, output_feat)
        # Reshape back to social_stgcnn convention (B, output_feat, T_pred, N)
        out = out.view(B, N, self.pred_seq_len, -1).permute(0, 3, 2, 1)
        return out, a


class _GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al.) with optional adjacency mask.

    Uses the factored attention formulation:
        e_ij^h = LeakyReLU(attn_src_h · W_h h_i  +  attn_dst_h · W_h h_j)
        alpha_ij^h = softmax_j(e_ij^h)
        h'_i = concat_h [ sum_j alpha_ij^h * W_h h_j ]

    When mask is provided (1 = edge exists), attention to absent pedestrians is
    set to -inf before softmax, focusing each node on its spatial neighbours.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, alpha=0.2):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.out_dim   = out_dim
        self.W         = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.alpha = alpha

    def forward(self, x, mask=None):
        # x: (..., N, in_dim);  mask: (..., N, N) binary, 1 = edge exists
        *prefix, N, _ = x.shape
        h = self.W(x).view(*prefix, N, self.num_heads, self.head_dim)  # (..., N, H, D)

        # e[..., i, j, h] = attention score from node i to node j for head h
        e = (h * self.attn_src).sum(-1).unsqueeze(-2) \
          + (h * self.attn_dst).sum(-1).unsqueeze(-3)   # (..., N_i, N_j, H)
        e = F.leaky_relu(e, self.alpha)

        if mask is not None:
            e = e.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

        # softmax over N_j; nan_to_num handles isolated nodes (all-inf rows)
        attn = torch.nan_to_num(F.softmax(e, dim=-2), nan=0.0)  # (..., N_i, N_j, H)

        out = torch.einsum('...ijh,...jhd->...ihd', attn, h)     # (..., N_i, H, D)
        return out.reshape(*prefix, N, self.out_dim)


class social_gat(nn.Module):
    """Per-frame multi-head GAT for spatial interactions + temporal LSTM.

    Pipeline per scene:
        1. Stack of GAT layers over pedestrians, applied independently to each
           observed frame, using the adjacency matrix as an attention mask so
           each pedestrian only attends to its spatial neighbours.
        2. Temporal LSTM encoder over the GAT-features per pedestrian.
        3. Autoregressive LSTM decoder produces pred_seq_len bivariate-Gaussian
           outputs per pedestrian.
    """

    def __init__(self, input_feat=2, output_feat=5,
                 hidden_dim=64, seq_len=8, pred_seq_len=12, dropout=0.0,
                 num_gat_layers=2, num_heads=4):
        super().__init__()
        self.pred_seq_len = pred_seq_len

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(_GATLayer(input_feat, hidden_dim, num_heads=num_heads))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(_GATLayer(hidden_dim, hidden_dim, num_heads=num_heads))

        self.lstm_enc   = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_dec   = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.input_proj = nn.Linear(input_feat, hidden_dim)
        self.out        = nn.Linear(hidden_dim, output_feat)
        self.drop       = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, v, a):
        # v: (B, F=2, T, N) -> (B*T, N, F) for per-frame GAT
        B, F_in, T, N = v.shape
        x = v.permute(0, 2, 3, 1).reshape(B * T, N, F_in)

        # Build binary adjacency mask from the graph kernel stack
        # a: (K, N, N) after squeeze in training loop  →  collapse K → (N, N)
        if a.dim() == 3:    # (K, N, N)
            mask = (a.sum(0) > 0).float().unsqueeze(0).expand(B * T, -1, -1)
        elif a.dim() == 2:  # (N, N)
            mask = a.unsqueeze(0).expand(B * T, -1, -1)
        else:
            mask = None

        h = x
        for gat in self.gat_layers:
            h = F.elu(gat(h, mask))
            h = self.drop(h)

        # (B*T, N, hidden) -> (B*N, T, hidden) for temporal LSTM
        h = h.view(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)

        _, (he, ce) = self.lstm_enc(h)

        # Autoregressive decoder
        cur = h[:, -1:, :]
        outs = []
        for _ in range(self.pred_seq_len):
            o, (he, ce) = self.lstm_dec(cur, (he, ce))
            p = self.out(o)                                      # (B*N, 1, output_feat)
            outs.append(p)
            cur = self.input_proj(p[..., :2])                    # project mean to hidden

        out = torch.cat(outs, dim=1)                             # (B*N, T_pred, output_feat)
        out = out.view(B, N, self.pred_seq_len, -1).permute(0, 3, 2, 1)
        return out, a


def build_model_from_args(args):
    """Factory: instantiate the right model class based on args.model_type.

    Recognised model_type values: 'stgcnn', 'lstm', 'gat'.
    Default (unset) is 'stgcnn' for backward compatibility.
    """
    mt = getattr(args, 'model_type', 'stgcnn')
    if mt == 'stgcnn':
        return social_stgcnn(
            n_stgcnn=args.n_stgcnn,
            n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size,
            seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size,
            pred_seq_len=args.pred_seq_len,
        )
    if mt == 'lstm':
        return social_lstm(
            input_feat=args.input_size,
            output_feat=args.output_size,
            hidden_dim=getattr(args, 'hidden_dim', 64),
            num_layers=getattr(args, 'num_layers', 1),
            seq_len=args.obs_seq_len,
            pred_seq_len=args.pred_seq_len,
            dropout=getattr(args, 'dropout', 0.0),
        )
    if mt == 'gat':
        return social_gat(
            input_feat=args.input_size,
            output_feat=args.output_size,
            hidden_dim=getattr(args, 'hidden_dim', 64),
            seq_len=args.obs_seq_len,
            pred_seq_len=args.pred_seq_len,
            dropout=getattr(args, 'dropout', 0.0),
            num_gat_layers=getattr(args, 'num_gat_layers', 2),
            num_heads=getattr(args, 'num_heads', 4),
        )
    raise ValueError(f"Unknown model_type: {mt}. Expected 'stgcnn', 'lstm', or 'gat'.")
