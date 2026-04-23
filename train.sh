#!/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 250 && echo "eth Done."

CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag social-stgcnn-hotel --use_lrschd --num_epochs 250 && echo "hotel Done."

CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag social-stgcnn-univ --use_lrschd --num_epochs 250 && echo "univ Done."

CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag social-stgcnn-zara1 --use_lrschd --num_epochs 250 && echo "zara1 Done."

CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag social-stgcnn-zara2 --use_lrschd --num_epochs 250 && echo "zara2 Done."