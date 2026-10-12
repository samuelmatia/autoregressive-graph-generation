# python -m src.generate --ckpt logs/zinc250k/base/best.pt --num_samples 10000
# python -m src.generate --ckpt logs/zinc250k/base/best.pt --no_check          # without valency check
# python -m src.generate --ckpt logs/zinc250k/base/best.pt --temperature 0.75 --min_atoms 5 --max_atoms 48

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from .chem import BOND_TYPES, atomic_numbers, check_valency, final_smiles
from .models.model import GraphAF, load_checkpoint
from .models.utils import edge_window, featurize, set_seed, setup_gpu


@dataclass
class Sample:
    smiles: Optional[str]    # None if the molecule is invalid
    num_atoms: int
    n_resample: int          # rejected bonds, 0 means the valency check was never needed


# init: optional list of (types (n0,), bonds (n0,n0)) starting subgraphs in BFS order (constrained optimization)
# record: also return the trajectory (final graphs and every attempted action) for REINFORCE
@torch.no_grad()
def sample_batch(model: GraphAF, num: int, temperature: float = 0.75, max_atoms: int = 48, check: bool = True,
                 init: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None, record: bool = False,
                 progress: bool = False):
    model.eval()
    dev = next(model.parameters()).device
    d, bd, U, T = model.node_dim, model.bond_dim, model.edge_unroll, max_atoms
    anum = atomic_numbers(model.atom_decoder)

    x = torch.zeros(num, T, d, device=dev)
    adj = torch.zeros(num, bd, T, T, device=dev)
    rw = [Chem.RWMol() for _ in range(num)]
    n0 = torch.zeros(num, dtype=torch.long, device=dev)
    for b, (types, bonds) in enumerate(init or []):
        n = len(types)
        n0[b] = n
        ar = torch.arange(n, device=dev)
        x[b, ar, torch.as_tensor(types, device=dev).long()] = 1
        bt = torch.as_tensor(np.asarray(bonds), device=dev).long()
        for k in range(1, bd):
            adj[b, k - 1, :n, :n] = (bt == k).float()
        adj[b, :, ar, ar] = 1
        for a in types:
            rw[b].AddAtom(Chem.Atom(anum[int(a)]))
        for i, j in zip(*np.nonzero(np.triu(np.asarray(bonds), 1))):
            rw[b].AddBond(int(i), int(j), BOND_TYPES[int(bonds[i, j]) - 1])
    n0_list = n0.tolist()

    alive = [True] * num
    n_res = [0] * num
    n_steps = [int(n) for n in n0_list]
    logs = [[] for _ in range(num)]

    for i in tqdm(range(T), desc="decoding", leave=False, disable=not progress):
        if not any(alive):
            break
        act = [b for b in range(num) if alive[b] and n0_list[b] <= i]
        if not act:
            continue
        at = torch.tensor(act, device=dev)

        # new node
        z = torch.randn(len(act), d, device=dev) * temperature
        typ = model.reverse_node(model.node_emb(x[at], adj[at]), z).argmax(-1)
        x[at, i, typ] = 1.0
        adj[at, :, i, i] = 1.0
        mark = {}
        for b, t in zip(act, typ.tolist()):
            rw[b].AddAtom(Chem.Atom(anum[t]))
            mark[b] = len(logs[b])
            logs[b].append((0, i, 0, t, True))
            n_steps[b] = i + 1
        connected = {b: i == 0 for b in act}

        # bonds to the previous nodes inside the window
        start, total = edge_window(i, U)
        for j in range(total):
            a = start + j
            index = torch.tensor([[a, i]], device=dev).expand(len(act), 2)
            emb = model.edge_emb(x[at], adj[at], index)
            pending = list(range(len(act)))
            tried = [set() for _ in act]
            tries = [0] * len(act)
            upd = []
            while pending:
                forced = {q for q in pending if len(tried[q]) >= 3 or tries[q] > 50}
                free = [q for q in pending if q not in forced]
                choice = {q: 3 for q in forced}
                if free:
                    fq = torch.tensor(free, device=dev)
                    zz = torch.randn(len(free), bd, device=dev) * temperature
                    out = model.reverse_edge(emb[fq], zz).argmax(-1).tolist()
                    choice.update(zip(free, out))
                nxt = []
                for q in pending:
                    b, e = act[q], choice[q]
                    ok = True
                    if e != 3:
                        rw[b].AddBond(i, a, BOND_TYPES[e])
                        ok = (not check) or check_valency(rw[b])
                        if not ok:
                            rw[b].RemoveBond(i, a)
                            n_res[b] += 1
                            tries[q] += 1
                            tried[q].add(e)
                            nxt.append(q)
                        else:
                            connected[b] = True
                    logs[b].append((1, i, a, e, ok))
                    if ok:
                        upd.append((b, e))
                pending = nxt
            if upd:
                ub = torch.tensor([u[0] for u in upd], device=dev)
                ue = torch.tensor([u[1] for u in upd], device=dev)
                adj[ub, ue, i, a] = 1.0
                adj[ub, ue, a, i] = 1.0

        # a node left unbonded stops the generation and is discarded
        for b in act:
            if not connected[b]:
                alive[b] = False
                rw[b].RemoveAtom(i)
                del logs[b][mark[b]:]

    samples = []
    for b in range(num):
        mol = rw[b].GetMol()
        samples.append(Sample(final_smiles(mol), mol.GetNumAtoms(), n_res[b]))
    if not record:
        return samples, None
    return samples, dict(x=x, adj=adj, logs=logs, n_steps=n_steps, n0=n0_list)


# sample until num molecules with at least min_atoms atoms are collected
def generate_molecules(model: GraphAF, num: int, batch_size: int = 512, temperature: float = 0.75,
                       max_atoms: int = 48, min_atoms: int = 5, check: bool = True,
                       verbose: bool = True) -> Tuple[List[Sample], float]:
    out: List[Sample] = []
    t0 = time.time()
    with tqdm(total=num, desc="generate", disable=not verbose) as bar:
        while len(out) < num:
            samples, _ = sample_batch(model, min(batch_size, num - len(out) + 16), temperature, max_atoms, check)
            keep = [s for s in samples if s.num_atoms >= min_atoms][:num - len(out)]
            out.extend(keep)
            bar.update(len(keep))
    dt = (time.time() - t0) / max(num, 1)
    if verbose:
        print(f"[generate] {dt * 1000:.2f} ms / molecule")
    return out, dt


# encode graphs to latents, decode them back and count exact matches (real nodes and modeled bonds)
@torch.no_grad()
def reconstruction_rate(model: GraphAF, loader, max_batches: int = 10) -> float:
    model.eval()
    dev = next(model.parameters()).device
    N, d, bd, U = model.max_size, model.node_dim, model.bond_dim, model.edge_unroll
    ok = tot = 0
    for bi, (types, bonds) in enumerate(loader):
        if bi >= max_batches:
            break
        x, adj = featurize(types.to(dev), bonds.to(dev), d)
        zn, ze, _, _ = model.encode(x, adj)
        B = x.size(0)
        xr = torch.zeros_like(x)
        ar = torch.zeros_like(adj)
        ids = torch.arange(B, device=dev)
        e = 0
        for i in range(N):
            xr[ids, i] = torch.nn.functional.one_hot(
                model.reverse_node(model.node_emb(xr, ar), zn[:, i]).argmax(-1), d).float()
            ar[:, :, i, i] = 1.0
            start, total = edge_window(i, U)
            for j in range(total):
                idx = torch.tensor([[start + j, i]], device=dev).expand(B, 2)
                k = model.reverse_edge(model.edge_emb(xr, ar, idx), ze[:, e]).argmax(-1)
                ar[ids, k, i, start + j] = 1.0
                ar[ids, k, start + j, i] = 1.0
                e += 1
        real = (types.to(dev) >= 0)
        n_ok = ((xr.argmax(-1) == x.argmax(-1)) | ~real).all(1)
        pair = real.unsqueeze(1) & real.unsqueeze(2) & model.edge_sel.unsqueeze(0)
        e_ok = ((ar.argmax(1) == adj.argmax(1)) | ~pair).flatten(1).all(1)
        ok += int((n_ok & e_ok).sum())
        tot += B
    return ok / max(tot, 1)


def main():
    from .data import load_smiles
    from .evaluate import evaluate_samples, summarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--min_atoms", type=int, default=5)
    ap.add_argument("--max_atoms", type=int, default=48)
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    setup_gpu()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, ck = load_checkpoint(a.ckpt, device)
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "gen")
    os.makedirs(out, exist_ok=True)

    samples, dt = generate_molecules(model, a.num_samples, a.batch_size, a.temperature, a.max_atoms,
                                     a.min_atoms, check=not a.no_check)
    res = evaluate_samples(samples, load_smiles(meta["name"], a.root, "train"))
    print(summarize(res))
    with open(os.path.join(out, "generated_smiles.txt"), "w") as f:
        f.write("\n".join(s.smiles if s.smiles is not None else "None" for s in samples))
    metrics = dict(res, sec_per_mol=dt, temperature=a.temperature, check=not a.no_check, ckpt=a.ckpt,
                   step=ck.get("step"))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"[generate] results in {out}")


if __name__ == "__main__":
    main()
