import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utils import build_masks

LOG_2PI = math.log(2 * math.pi)


class RelGraphConv(nn.Module):
    def __init__(self, din: int, dout: int, n_rel: int = 3, relu: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_rel, din, dout))
        nn.init.xavier_uniform_(self.weight)
        self.relu = relu

    # x (M,N,din), adj (M,R,N,N) -> (M,N,dout), messages summed over relations
    def forward(self, x, adj):
        h = torch.matmul(adj, torch.einsum("bnd,rdh->brnh", x, self.weight))
        return (F.relu(h) if self.relu else h).sum(1)


class RGCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nout: int, n_rel: int = 3, num_layers: int = 3):
        super().__init__()
        self.emb = nn.Linear(nfeat, nfeat, bias=False)
        nn.init.xavier_uniform_(self.emb.weight)
        self.layers = nn.ModuleList(
            [RelGraphConv(nfeat, nhid, n_rel)]
            + [RelGraphConv(nhid, nhid, n_rel) for _ in range(num_layers - 2)]
            + [RelGraphConv(nhid, nout, n_rel, relu=False)])

    def forward(self, x, adj):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, adj)
        return x


class Rescale(nn.Module):
    def __init__(self, n: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return torch.exp(self.weight) * x


# predicts scale s > 0 and shift t of the affine transform from the step embedding
class STNet(nn.Module):
    def __init__(self, kind: str, din: int, dout: int, hid: int, shift: float):
        super().__init__()
        self.kind, self.dout, self.shift = kind, dout, shift
        self.l1 = nn.Linear(din, hid)
        if kind == "softplus":
            self.l2 = nn.Linear(hid, hid)
            self.l3 = nn.Linear(hid, 2 * dout)
            self.rescale1 = Rescale(dout)
        else:
            self.l3 = nn.Linear(hid, 2 * dout)
            self.rescale1 = Rescale()
            if kind == "sigmoid":
                self.rescale2 = Rescale()
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.constant_(self.l3.weight, 1e-10)
        for m in (self.l1, self.l3) + ((self.l2,) if kind == "softplus" else ()):
            nn.init.constant_(m.bias, 0.0)
        if kind == "softplus":
            nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, e):
        if self.kind == "softplus":
            h = self.l3(torch.tanh(self.l2(F.relu(self.l1(e)))))
            return self.rescale1(F.softplus(h[:, :self.dout])), h[:, self.dout:]
        h = self.l3(torch.tanh(self.l1(e)))
        if self.kind == "exp":
            return self.rescale1(torch.tanh(h[:, :self.dout])).exp(), h[:, self.dout:]
        h = self.rescale1(h)
        return self.rescale2(torch.sigmoid(h[:, :self.dout] + self.shift)), h[:, self.dout:]


class GraphAF(nn.Module):
    def __init__(self, max_size: int, node_dim: int, bond_dim: int = 4, edge_unroll: int = 12,
                 num_flow_layer: int = 6, gcn_layer: int = 3, nhid: int = 128, nout: int = 128,
                 st_type: str = "sigmoid", sigmoid_shift: float = 2.0, deq_coeff: float = 0.9,
                 is_bn: bool = False, learn_prior: bool = False, atom_decoder: Optional[list] = None):
        super().__init__()
        assert st_type in ("sigmoid", "exp", "softplus")
        self.cfg = dict(max_size=max_size, node_dim=node_dim, bond_dim=bond_dim, edge_unroll=edge_unroll,
                        num_flow_layer=num_flow_layer, gcn_layer=gcn_layer, nhid=nhid, nout=nout, st_type=st_type,
                        sigmoid_shift=sigmoid_shift, deq_coeff=deq_coeff, is_bn=is_bn, learn_prior=learn_prior,
                        atom_decoder=atom_decoder)
        self.max_size, self.node_dim, self.bond_dim = max_size, node_dim, bond_dim
        self.edge_unroll = min(edge_unroll, max_size)
        self.deq, self.nout, self.st_type = deq_coeff, nout, st_type
        self.atom_decoder = atom_decoder

        m = build_masks(max_size, self.edge_unroll)
        self.n_edge = m["n_edge"]
        self.register_buffer("node_mask", torch.from_numpy(m["node_mask"]), persistent=False)
        self.register_buffer("adj_mask", torch.from_numpy(m["adj_mask"]), persistent=False)
        self.register_buffer("link", torch.from_numpy(m["link"]), persistent=False)
        self.register_buffer("edge_sel", torch.from_numpy(m["edge_sel"]), persistent=False)
        self.len_node = max_size * node_dim
        self.len_edge = self.n_edge * bond_dim

        self.prior_ln_var = nn.Parameter(torch.zeros(1), requires_grad=learn_prior)
        self.rgcn = RGCN(node_dim, nhid, nout, bond_dim - 1, gcn_layer)
        self.bn = nn.BatchNorm1d(nout) if is_bn else None
        self.node_st = nn.ModuleList([STNet(st_type, nout, node_dim, nhid, sigmoid_shift)
                                      for _ in range(num_flow_layer)])
        self.edge_st = nn.ModuleList([STNet(st_type, 3 * nout, bond_dim, nhid, sigmoid_shift)
                                      for _ in range(num_flow_layer)])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- embeddings ----
    # x (M,N,d), adj (M,>=3,N,N) -> node embeddings (M,N,nout)
    def _gcn(self, x, adj):
        h = self.rgcn(x, adj[:, :self.bond_dim - 1])
        if self.bn is not None:
            h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return h

    # graph embedding used to generate the next node, (M,nout)
    def node_emb(self, x, adj):
        return self._gcn(x, adj).sum(1)

    # graph embedding + embeddings of the two end nodes, (M,3*nout); index (M,2) = (older node, new node)
    def edge_emb(self, x, adj, index):
        h = self._gcn(x, adj)
        idx = index.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.cat([torch.gather(h, 1, idx).flatten(1), h.sum(1)], dim=1)

    # embeddings of every step of the sequence at once (masked copies of the graph)
    def _step_embs(self, x, adj):
        B, N, _ = x.shape
        R = self.node_mask.size(0)
        xs = x.unsqueeze(1) * self.node_mask.view(1, R, N, 1)
        as_ = adj[:, :self.bond_dim - 1].unsqueeze(1) * self.adj_mask.view(1, R, 1, N, N)
        h = self._gcn(xs.reshape(B * R, N, -1), as_.reshape(B * R, -1, N, N)).view(B, R, N, -1)
        g = h.sum(2)
        he = h[:, N:]
        idx = self.link.view(1, -1, 2, 1).expand(B, -1, -1, h.size(-1))
        pair = torch.gather(he, 2, idx)
        node_in = g[:, :N].reshape(B * N, -1)
        edge_in = torch.cat([pair, g[:, N:].unsqueeze(2)], dim=2).reshape(B * self.n_edge, -1)
        return node_in, edge_in

    # ---- flow ----
    def _forward_flow(self, z, emb, nets):
        logdet = 0.0
        for net in nets:
            s, t = net(emb)
            z = z * s + t if self.st_type == "sigmoid" else (z + t) * s
            logdet = logdet + (s.abs() + 1e-20).log()
        return z, logdet

    def _inverse_flow(self, z, emb, nets):
        for net in reversed(nets):
            s, t = net(emb)
            z = (z - t) / s if self.st_type == "sigmoid" else z / s - t
        return z

    def _prior_ll(self, z):
        return (-0.5 * (LOG_2PI + self.prior_ln_var + torch.exp(-self.prior_ln_var) * z ** 2)).sum(-1)

    # x (B,N,d) one-hot, adj (B,4,N,N); latents of the dequantized graph and log-determinants
    def encode(self, x, adj):
        B = x.size(0)
        xd = x + self.deq * torch.rand_like(x)
        ed = adj[:, :, self.edge_sel].permute(0, 2, 1).contiguous()
        ed = ed + self.deq * torch.rand_like(ed)
        node_in, edge_in = self._step_embs(x, adj)
        zn, ldn = self._forward_flow(xd.reshape(B * self.max_size, -1), node_in, self.node_st)
        ze, lde = self._forward_flow(ed.reshape(B * self.n_edge, -1), edge_in, self.edge_st)
        return (zn.view(B, self.max_size, -1), ze.view(B, self.n_edge, -1),
                ldn.view(B, -1).sum(-1), lde.view(B, -1).sum(-1))

    # exact -log p(G) per graph (up to the dequantization constant), (B,)
    def nll(self, x, adj):
        zn, ze, ldn, lde = self.encode(x, adj)
        ll = self._prior_ll(zn.flatten(1)) + ldn - self.len_node
        ll = ll + self._prior_ll(ze.flatten(1)) + lde - self.len_edge
        return -ll

    @torch.no_grad()
    def per_sample_nll(self, x, adj, per_dim: bool = False):
        out = self.nll(x, adj)
        return out / (self.len_node + self.len_edge) if per_dim else out

    # ---- sampling (one new node / one new bond per call, batched) ----
    def reverse_node(self, emb, latent):
        return self._inverse_flow(latent, emb, self.node_st)

    def reverse_edge(self, emb, latent):
        return self._inverse_flow(latent, emb, self.edge_st)

    # ---- single-step log densities, used by RL ----
    def logp_node(self, x, adj, action):
        z, ld = self._forward_flow(self._cont(action, self.node_dim), self.node_emb(x, adj), self.node_st)
        return self._prior_ll(z) + ld.sum(-1)

    def logp_edge(self, x, adj, index, action):
        z, ld = self._forward_flow(self._cont(action, self.bond_dim), self.edge_emb(x, adj, index), self.edge_st)
        return self._prior_ll(z) + ld.sum(-1)

    def _cont(self, action, dim):
        a = F.one_hot(action, dim).float()
        return a + self.deq * torch.rand_like(a)


# self-contained checkpoint: weights + model config + dataset meta
def save_checkpoint(path: str, model: GraphAF, meta: dict, step: int = 0, optimizer=None, **extra):
    state = {"model": model.state_dict(), "model_cfg": model.cfg, "meta": meta, "step": step, **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device="cpu") -> Tuple[GraphAF, dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GraphAF(**ck["model_cfg"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return model.to(device), ck["meta"], ck
