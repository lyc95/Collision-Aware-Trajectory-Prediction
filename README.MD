# Reference Paper: Social-STGCNN  
### *A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction*  
**Authors:** Abduallah Mohamed, Kun Qian, Mohamed Elhoseiny, Christian Claudel  
**Conference:** CVPR 2020  

Read the full paper: https://arxiv.org/abs/2002.11927

---

## Overview  

In this project, we take reference from the **Social-STGCNN** framework, which formulates human trajectory prediction as a **spatio-temporal graph learning problem**.  

The model captures both:
- **Temporal dynamics** (trajectory evolution over time)  
- **Social interactions** (influence between agents)  

The original work demonstrates:
- ~20% improvement in **Final Displacement Error (FDE)**  
- Improved **Average Displacement Error (ADE)**  
- ~7.6K parameters (lightweight model)  
- Up to 48× faster inference (~0.002s per frame)  
- Strong performance even with only 20% of training data  

---

## Model Architecture  

The Social-STGCNN model consists of two main components:

### 1. ST-GCNN (Spatio-Temporal Graph CNN)  
- Builds a **spatio-temporal graph representation** from observed trajectories  
- Encodes spatial and temporal dependencies  
- Uses a kernel function to model **social interactions** via the adjacency matrix  

### 2. TXP-CNN (Time Extrapolator CNN)  
- Takes the learned graph embeddings  
- Predicts **future trajectories** of agents  

---

## Key Idea  

Pedestrians are modeled as nodes in a graph:
- Edges represent **interactions between agents**  
- The adjacency matrix captures **social influence**  
- Graph convolutions learn motion patterns over space and time  
- Quantifying and Mitigating Collision Failures in Graph-Based Social Models

---

## Implementation Setup  

The reference implementation is based on:
- Python 3.6  
- PyTorch  
- NetworkX  
- NumPy  
- tqdm  

Install dependencies:
```bash
pip install -r requirements.txt