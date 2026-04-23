"""Evaluate all Phase 1 checkpoints and produce a comparison table.

Loads every ./checkpoint/phase1-* directory, runs test-time sampling,
and prints a table of ADE / FDE / ColRate_avg / ColRate_minADE.

Results are also written to phase1_results.txt.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import copy
import glob
import pickle

import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist
from torch.utils.data import DataLoader

from metrics import (
    ade, fde, seq_to_nodes, nodes_rel_to_nodes_abs, collision_metrics,
)
from model import social_stgcnn
from utils import TrajectoryDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(checkpoint_dir, ksteps=20, d_col=0.2):
    model_path = os.path.join(checkpoint_dir, 'val_best.pth')
    args_path = os.path.join(checkpoint_dir, 'args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    data_set = f'./datasets/{args.dataset}/test/'
    dset = TrajectoryDataset(data_set, obs_len=args.obs_seq_len,
                             pred_len=args.pred_seq_len, skip=1,
                             norm_lap_matr=True)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                          output_feat=args.output_size,
                          seq_len=args.obs_seq_len,
                          kernel_size=args.kernel_size,
                          pred_seq_len=args.pred_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ade_all, fde_all, col_avg_all, col_min_all = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = [t.to(device) for t in batch]
            (obs_traj, pred_traj_gt, _, _, _, _, V_obs, A_obs, V_tr, _) = batch
            num_objs = obs_traj.shape[1]

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1).squeeze()[:, :num_objs, :]
            V_tr = V_tr.squeeze()[:, :num_objs, :]

            sx = torch.exp(V_pred[:, :, 2])
            sy = torch.exp(V_pred[:, :, 3])
            corr = torch.tanh(V_pred[:, :, 4])
            cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2, device=device)
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy
            mvn = torchdist.MultivariateNormal(V_pred[:, :, :2], cov)

            V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
            V_y_abs = nodes_rel_to_nodes_abs(
                V_tr.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())

            ade_per_ped = {n: [] for n in range(num_objs)}
            fde_per_ped = {n: [] for n in range(num_objs)}
            samples = []
            for _ in range(ksteps):
                v = mvn.sample()
                v_abs = nodes_rel_to_nodes_abs(
                    v.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
                samples.append(v_abs[:, :num_objs, :])
                for n in range(num_objs):
                    pred = [v_abs[:, n:n+1, :]]
                    tgt = [V_y_abs[:, n:n+1, :]]
                    ade_per_ped[n].append(ade(pred, tgt, [1]))
                    fde_per_ped[n].append(fde(pred, tgt, [1]))

            for n in range(num_objs):
                ade_all.append(min(ade_per_ped[n]))
                fde_all.append(min(fde_per_ped[n]))

            target_scene = V_y_abs[:, :num_objs, :]
            col = collision_metrics(samples, target_scene, d_col=d_col)
            col_avg_all.append(col['ColRate_avg'])
            col_min_all.append(col['ColRate_minADE'])

    return {
        'ADE': np.mean(ade_all),
        'FDE': np.mean(fde_all),
        'ColRate_avg': np.mean(col_avg_all),
        'ColRate_minADE': np.mean(col_min_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='./checkpoint/phase1-*')
    ap.add_argument('--d_col', type=float, default=0.2)
    ap.add_argument('--ksteps', type=int, default=20)
    ap.add_argument('--output', default='phase1_results.txt')
    args = ap.parse_args()

    ckpt_dirs = sorted(glob.glob(args.pattern))
    if not ckpt_dirs:
        print(f'No checkpoints matched {args.pattern}')
        return

    rows = []
    for ck in ckpt_dirs:
        if not os.path.exists(os.path.join(ck, 'val_best.pth')):
            print(f'[skip] {ck} (no val_best.pth)')
            continue
        print(f'[eval] {ck}', flush=True)
        try:
            m = evaluate(ck, ksteps=args.ksteps, d_col=args.d_col)
        except Exception as e:
            print(f'  FAILED: {e}')
            continue
        name = os.path.basename(ck)
        rows.append((name, m['ADE'], m['FDE'], m['ColRate_avg'], m['ColRate_minADE']))

    # Print / write table.
    header = f'{"Run":35s} {"ADE":>8s} {"FDE":>8s} {"ColAvg":>8s} {"ColMin":>8s}'
    lines = [header, '-' * len(header)]
    for r in rows:
        lines.append(f'{r[0]:35s} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>8.4f} {r[4]:>8.4f}')

    table = '\n'.join(lines)
    print('\n' + table)
    with open(args.output, 'w') as f:
        f.write(table + '\n')
    print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
