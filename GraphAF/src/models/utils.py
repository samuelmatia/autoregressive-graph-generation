import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# tf32 for faster matmuls
def setup_gpu():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# bonds of node i are modeled against the previous `unroll` nodes only (BFS order)
def edge_window(i: int, unroll: int):
    return (0, i) if i < unroll else (i - unroll, unroll)


# one generation step per node, then one per (new node, previous node) pair inside the window
# node_mask[s] / adj_mask[s]: part of the graph visible at step s
# link[e] = (older node, new node) of edge step e, edge_id[i, a] = e, edge_sel = modeled entries of A
def build_masks(n: int, unroll: int) -> dict:
    unroll = min(unroll, n)
    n_edge = sum(edge_window(i, unroll)[1] for i in range(n))
    steps = n + n_edge
    node_m = np.zeros((steps, n), bool)
    adj_m = np.zeros((steps, n, n), bool)
    link = np.zeros((n_edge, 2), np.int64)
    edge_id = np.full((n, n), -1, np.int64)
    edge_sel = np.zeros((n, n), bool)

    for i in range(n):
        node_m[i, :i] = True
        adj_m[i, :i, :i] = True
    s, e = n, 0
    for i in range(n):
        start, tot = edge_window(i, unroll)
        for j in range(tot):
            node_m[s, :i + 1] = True
            if j == 0:
                adj_m[s] = adj_m[i]
                adj_m[s, i, i] = True
            else:
                adj_m[s] = adj_m[s - 1]
                adj_m[s, i, start + j - 1] = True
                adj_m[s, start + j - 1, i] = True
            link[e] = (start + j, i)
            edge_id[i, start + j] = e
            edge_sel[i, start + j] = True
            s += 1
            e += 1
    return dict(node_mask=node_m, adj_mask=adj_m, link=link, edge_id=edge_id, edge_sel=edge_sel,
                unroll=unroll, n_node=n, n_edge=n_edge)


# types (B,N) int with -1 for padding, bonds (B,N,N) uint8 with 0 = no bond, 1..3 = single/double/triple
# returns x (B,N,d) one-hot and adj (B,4,N,N): 3 bond channels, "no bond" channel, identity on every channel
def featurize(types: torch.Tensor, bonds: torch.Tensor, node_dim: int):
    types, bonds = types.long(), bonds.long()
    n = types.size(1)
    x = F.one_hot(types.clamp(min=0), node_dim).float() * (types >= 0).unsqueeze(-1)
    adj = torch.stack([(bonds == k).float() for k in (1, 2, 3)], dim=1)
    adj = torch.cat([adj, 1.0 - adj.sum(1, keepdim=True)], dim=1)
    return x, adj + torch.eye(n, device=adj.device)
