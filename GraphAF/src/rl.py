# reward functions, REINFORCE update and constrained-optimization helpers used by finetune.py

import json
import math
import multiprocessing as mp
import os.path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from .chem import (check_valency, min_penalized_logp, penalized_logp, qed, similarity, steric_strain_filter,
                   zinc_molecule_filter)
from .generate import sample_batch
from .models.utils import build_masks


# ---------------- rewards ----------------
# cfg: property, reward_type (exp | linear | imp), exp_temperature, exp_bias, qed_coeff, plogp_coeff
# returns (total reward, property reward, raw property score or None)
def score_molecule(args):
    smiles, cfg, org = args
    if smiles is None:
        return -3.0, 0.0, None
    mol = Chem.MolFromSmiles(smiles)
    r_valid = 2.0
    if not steric_strain_filter(mol):
        r_valid -= 1
    if not zinc_molecule_filter(mol):
        r_valid -= 1
    try:
        if cfg["property"] == "qed":
            score = qed(mol)
            r_prop = score * cfg["qed_coeff"]
        else:
            rt = cfg["reward_type"]
            score = min_penalized_logp(mol) if rt == "imp" else penalized_logp(mol)
            if rt == "exp":
                r_prop = math.exp(score / cfg["exp_temperature"]) - cfg["exp_bias"]
            elif rt == "linear":
                r_prop = score * cfg["plogp_coeff"]
            else:
                r_prop = score - org
    except Exception:
        return r_valid, 0.0, None
    return r_valid + r_prop, r_prop, score


def _plogp_worker(smi):
    try:
        return min_penalized_logp(Chem.MolFromSmiles(smi))
    except Exception:
        return float("inf")


def _co_worker(args):
    smi, org_smi, org_plogp = args
    try:
        m, o = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(org_smi)
        return min_penalized_logp(m) - org_plogp, similarity(m, o)
    except Exception:
        return None


# process pool for the (slow) rdkit scoring, n_jobs <= 1 runs inline
class Pool:
    def __init__(self, n_jobs: int):
        self.pool = mp.get_context("spawn").Pool(n_jobs) if n_jobs > 1 else None

    def map(self, fn, items, chunksize: int = 8, **kw):
        if self.pool is None:
            return [fn(i) for i in tqdm(items, **kw)]
        return list(tqdm(self.pool.imap(fn, items, chunksize=chunksize), total=len(items), **kw))

    def close(self):
        if self.pool is not None:
            self.pool.close()


# ---------------- REINFORCE ----------------
# masks for graphs of size T, used to rebuild the state seen at any step from the final graph
class StepMasks:
    def __init__(self, T: int, unroll: int, device):
        m = build_masks(T, unroll)
        self.T = T
        self.node_mask = torch.from_numpy(m["node_mask"]).to(device)
        self.adj_mask = torch.from_numpy(m["adj_mask"]).to(device)
        self.link = torch.from_numpy(m["link"]).to(device)
        self.edge_id = m["edge_id"]


# loss = -sum_t log p(a_t | s_t) * (r_t - b_t) / num_molecules, gradients are accumulated chunk by chunk
# reward of a step: final reward discounted by its distance to the end, -1 for a rejected bond
def reinforce_update(model, traj: dict, final_rewards: Sequence[float], baseline: Optional[torch.Tensor],
                     masks: StepMasks, decay: float, moving_coeff: float, penalty: bool = True,
                     chunk: int = 2048):
    dev = traj["x"].device
    n_mols = len(final_rewards)
    cols = {0: [], 1: []}   # kind -> (mol, step, a, i, action, reward, baseline index)
    for b, log in enumerate(traj["logs"]):
        n_tot, R = traj["n_steps"][b], final_rewards[b]
        for kind, i, a, action, ok in log:
            if ok:
                r = R * decay ** (n_tot - 1 - i)
            elif penalty:
                r = -1.0
            else:
                continue
            step = i if kind == 0 else masks.T + int(masks.edge_id[i, a])
            cols[kind].append((b, step, a, i, action, r, i + 1))

    size = masks.T + 5
    s, c = torch.zeros(size), torch.zeros(size)
    for rows in cols.values():
        for row in rows:
            s[row[6]] += row[5]
            c[row[6]] += 1
    cur = s / c.clamp(min=1)
    new_baseline = cur if baseline is None else cur * (1 - moving_coeff) + baseline * moving_coeff
    new_baseline = new_baseline.to(dev)

    model.train()
    for m in model.modules():                 # batch norm must keep its running statistics
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
    total = 0.0
    for kind, rows in cols.items():
        if not rows:
            continue
        t = torch.tensor([r[:5] for r in rows], device=dev)
        rew = torch.tensor([r[5] for r in rows], device=dev)
        adv = rew - new_baseline[torch.tensor([r[6] for r in rows], device=dev)]
        for k in range(0, len(rows), chunk):
            mb, st, a_, i_, act = t[k:k + chunk].unbind(1)
            xs = traj["x"][mb] * masks.node_mask[st].unsqueeze(-1)
            as_ = traj["adj"][mb] * masks.adj_mask[st].unsqueeze(1)
            if kind == 0:
                lp = model.logp_node(xs, as_, act)
            else:
                lp = model.logp_edge(xs, as_, masks.link[st - masks.T], act)
            loss = -(lp * adv[k:k + chunk]).sum() / n_mols
            loss.backward()
            total += loss.item()
    return total, new_baseline.cpu()


# ---------------- constrained optimization ----------------
# 800 ZINC250k molecules with the lowest penalized logP (cached in processed/co_set.json)
def build_co_set(root: str, pool: Pool, n: int = 800) -> dict:
    from .data import load_smiles, load_split
    path = osp.join(root, "zinc250k", "processed", "co_set.json")
    if osp.exists(path):
        return json.load(open(path))
    smiles = load_smiles("zinc250k", root, "train")
    scores = np.array(pool.map(_plogp_worker, smiles, chunksize=200, desc="plogp train set"))
    store = load_split("zinc250k", root, "train")
    chosen = []
    for i in np.argsort(scores):
        types, edges = store.graph(int(i))
        mol = Chem.RWMol()
        from .chem import BOND_TYPES, atomic_numbers
        from .data import ATOM_DECODERS
        an = atomic_numbers(ATOM_DECODERS["zinc250k"])
        for t in types:
            mol.AddAtom(Chem.Atom(an[int(t)]))
        for u, v, k in edges:
            mol.AddBond(int(u), int(v), BOND_TYPES[int(k)])
        if check_valency(mol):        # skip molecules whose charged atoms break the neutral valence check
            chosen.append(int(i))
        if len(chosen) == n:
            break
    out = dict(indices=chosen, smiles=[smiles[i] for i in chosen], plogp=[float(scores[i]) for i in chosen])
    json.dump(out, open(path, "w"))
    return out


# random BFS prefix of a molecule: the model must complete it (keep all nodes half of the time)
def make_init(ds, k: int, rng: np.random.Generator, modify_size: int = 6):
    types, bonds = ds.raw(k, rng)
    n = int((types >= 0).sum())
    m = int(rng.integers(0, modify_size)) if rng.random() < 0.5 else 0
    keep = max(n - m, 1)
    return types[:keep].astype(np.int64), bonds[:keep, :keep]


# best improvement per molecule under similarity >= delta, mean/std over molecules with a candidate;
# success = fraction of molecules whose best candidate improves the property
def evaluate_constrained(model, ds, co: dict, pool: Pool, repeat: int, n_mols: int, modify_size: int,
                         max_atoms: int, temperature: float, deltas=(0.0, 0.2, 0.4, 0.6), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    best = {d: [] for d in deltas}
    for k in tqdm(range(min(n_mols, len(co["smiles"]))), desc="constrained eval"):
        inits = [make_init(ds, k, rng, modify_size) for _ in range(repeat)]
        samples, _ = sample_batch(model, repeat, temperature, max_atoms, True, init=inits)
        cands = sorted({s.smiles for s in samples if s.smiles})
        res = [r for r in pool.map(_co_worker, [(c, co["smiles"][k], co["plogp"][k]) for c in cands],
                                   disable=True) if r is not None]
        for d in deltas:
            ok = [(imp, sim) for imp, sim in res if sim >= d]
            best[d].append(max(ok) if ok else None)
    out = {}
    for d, rows in best.items():
        found = [r for r in rows if r is not None]
        imp = np.array([r[0] for r in found]) if found else np.zeros(1)
        sim = np.array([r[1] for r in found]) if found else np.zeros(1)
        out[str(d)] = dict(improvement_mean=float(imp.mean()), improvement_std=float(imp.std()),
                           similarity_mean=float(sim.mean()), similarity_std=float(sim.std()),
                           success=float(np.mean([bool(r is not None and r[0] > 0) for r in rows])))
    return out
