#!/bin/bash
# Phase 1: compare 5 collision losses on 2 scenes (sparse + dense).
# 10 runs total. Each ~45 min on a single GPU = ~7.5 hours total.
# Fixed hyperparameters: lambda_col=1.0, d_min=0.4
# Checkpoints saved to ./checkpoint/phase1-<scene>-<loss>/

set -e
echo " Phase 1: 5 losses x 2 scenes = 10 runs"

LOSSES=(hinge exp inv gauss ecp)
SCENES=(eth univ)
EPOCHS=250
LAMBDA=1.0
DMIN=0.4

for scene in "${SCENES[@]}"; do
  for loss in "${LOSSES[@]}"; do
    TAG="phase1-${scene}-${loss}"
    echo "============================================================"
    echo " Training: scene=${scene} loss=${loss} tag=${TAG}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=0 python train.py \
      --dataset "${scene}" \
      --tag "${TAG}" \
      --collision_loss "${loss}" \
      --lambda_col "${LAMBDA}" \
      --d_min "${DMIN}" \
      --lr 0.01 --n_stgcnn 1 --n_txpcnn 5 \
      --num_epochs "${EPOCHS}" --use_lrschd \
      && echo " ${TAG} Done."
  done
done

# Also train baselines (no collision loss) on the same scenes for clean comparison.
for scene in "${SCENES[@]}"; do
  TAG="phase1-${scene}-baseline"
  echo "============================================================"
  echo " Training baseline: scene=${scene} tag=${TAG}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset "${scene}" \
    --tag "${TAG}" \
    --collision_loss none --lambda_col 0.0 \
    --lr 0.01 --n_stgcnn 1 --n_txpcnn 5 \
    --num_epochs "${EPOCHS}" --use_lrschd \
    && echo " ${TAG} Done."
done

echo "============================================================"
echo " Phase 1 complete. Evaluate with: python run_phase1_eval.py"
echo "============================================================"
