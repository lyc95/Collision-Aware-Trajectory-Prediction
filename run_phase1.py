"""Run Phase 1 experiments: 5 collision losses x 2 scenes (+ 2 baselines).

Works identically on local machine and Google Colab.
Each experiment writes to ./checkpoint/phase1-<scene>-<loss>/.
Results are appended to phase1_results.txt.

Usage:
    python run_phase1.py                    # run everything (~8 hours)
    python run_phase1.py --scenes eth        # just one scene
    python run_phase1.py --losses hinge ecp  # just two losses
    python run_phase1.py --epochs 50         # quick smoke test
"""
import argparse
import os
import subprocess
import sys
import time

DEFAULT_LOSSES = ['hinge', 'exp', 'inv', 'gauss', 'ecp']
DEFAULT_SCENES = ['eth', 'univ']


def run(cmd, log_file=None):
    """Run a training command, streaming output and optionally mirroring to log."""
    print('>>> ' + ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    status = 'OK' if result.returncode == 0 else f'FAILED({result.returncode})'
    line = f'{status}  {" ".join(cmd)}  elapsed={elapsed/60:.1f}min'
    print(line, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + '\n')
    return result.returncode == 0


def train_one(scene, loss, epochs, lambda_col, d_min, log_file):
    tag = f'phase1-{scene}-{loss}'
    cmd = [
        sys.executable, 'train.py',
        '--dataset', scene,
        '--tag', tag,
        '--collision_loss', loss,
        '--lambda_col', str(lambda_col if loss != 'none' else 0.0),
        '--d_min', str(d_min),
        '--lr', '0.01',
        '--n_stgcnn', '1',
        '--n_txpcnn', '5',
        '--num_epochs', str(epochs),
        '--use_lrschd',
    ]
    return run(cmd, log_file=log_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=DEFAULT_SCENES,
                    choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'])
    ap.add_argument('--losses', nargs='+', default=DEFAULT_LOSSES + ['none'],
                    choices=['hinge', 'exp', 'inv', 'gauss', 'ecp', 'none'])
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--lambda_col', type=float, default=1.0)
    ap.add_argument('--d_min', type=float, default=0.4)
    ap.add_argument('--log', default='phase1_runs.log')
    args = ap.parse_args()

    print(f'Running {len(args.scenes)} scenes x {len(args.losses)} losses '
          f'= {len(args.scenes) * len(args.losses)} runs', flush=True)

    ok = 0
    total = 0
    for scene in args.scenes:
        for loss in args.losses:
            total += 1
            success = train_one(scene, loss, args.epochs, args.lambda_col,
                                args.d_min, args.log)
            if success:
                ok += 1

    print(f'\n=== Phase 1 complete: {ok}/{total} runs succeeded ===', flush=True)
    print(f'Log: {args.log}', flush=True)
    print('Evaluate with: python run_phase1_eval.py', flush=True)


if __name__ == '__main__':
    main()
