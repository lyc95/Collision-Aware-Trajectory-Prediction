import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_,max_nodes = 88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def _pairwise_min_distance(traj, eps=1e-8):
    """Return shape (T,) — min pairwise distance across peds at each timestep.
    traj: (T, N, 2) absolute positions. Returns +inf for frames with <2 peds."""
    T, N, _ = traj.shape
    if N < 2:
        return np.full(T, np.inf)
    diff = traj[:, :, None, :] - traj[:, None, :, :]       # (T, N, N, 2)
    dist = np.sqrt((diff ** 2).sum(-1) + eps)              # (T, N, N)
    iu = np.triu_indices(N, k=1)
    return dist[:, iu[0], iu[1]].min(axis=1)               # (T,)


def has_collision(traj, d_col=0.2):
    """Does this single sampled trajectory contain any collision?
    traj: (T, N, 2) absolute positions. Returns bool."""
    return bool((_pairwise_min_distance(traj) < d_col).any())


def collision_rate(samples, d_col=0.2):
    """Fraction of K sampled trajectories containing at least one collision.
    samples: list of K arrays of shape (T, N, 2)."""
    if len(samples) == 0:
        return 0.0
    return np.mean([has_collision(s, d_col) for s in samples])


def scene_ade(sample, target):
    """Scene-level ADE: L2 error averaged over peds and timesteps.
    sample, target: (T, N, 2). Used for picking the best-of-K sample."""
    return float(np.sqrt(((sample - target) ** 2).sum(-1)).mean())


def collision_metrics(samples, target, d_col=0.2):
    """Full collision evaluation for one scene.
    samples: list of K (T, N, 2) sampled predictions.
    target:  (T, N, 2) ground-truth future positions.

    Returns dict with:
      ColRate_avg    — fraction of K samples that contain any collision
      ColRate_minADE — does the sample with min scene-ADE contain a collision?
    """
    flags = [has_collision(s, d_col) for s in samples]
    ades = [scene_ade(s, target) for s in samples]
    best = int(np.argmin(ades))
    return {
        'ColRate_avg': float(np.mean(flags)),
        'ColRate_minADE': float(flags[best]),
    }


def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result
   