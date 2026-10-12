# python -m src.evaluate --dataset qm9 --graphs logs/qm9/gen/graphs.pkl
# python -m src.evaluate --dataset zinc --graphs logs/zinc/gen/graphs.pkl --official --nspdk
# --official: MOSES metrics (FCD, Scaf, SNN, Frag), needs `pip install molsets`
# --nspdk: NSPDK MMD, needs EDeN (`pip install git+https://github.com/fabriziocosta/EDeN.git`)

import argparse
import pickle
import re
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

from .data import MolGraph, load_smiles, prepare_dataset, DATASET_NAMES

RDLogger.DisableLog("rdApp.*")

BOND_DECODER = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE,
                2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
ATOM_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 3, "S": 2, "Cl": 1, "Br": 1, "I": 1, "B": 3, "Se": 6, "Si": 4}
ORDER_TO_BOND = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        msg = str(e)
        return False, [int(v) for v in re.findall(r"\d+", msg[msg.find("#"):])]


# build the molecule edge by edge (larger index first), +1 charge on over-valent N/O/S
def construct_mol(g: MolGraph, atoms: Sequence[Sequence[int]]):
    mol = Chem.RWMol()
    for a in g.x:
        z, charge, nh = atoms[int(a)]
        atom = Chem.Atom(int(z))
        atom.SetFormalCharge(int(charge))
        atom.SetNumExplicitHs(int(nh))
        if nh > 0:
            atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    ei = g.edge_index
    for k in np.lexsort((ei[0], ei[1])):
        i, j = int(ei[0][k]), int(ei[1][k])
        mol.AddBond(j, i, BOND_DECODER[int(g.edge_attr[k])])
        ok, info = check_valency(mol)
        if not ok and info is not None and len(info) == 2:
            idx, v = info
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() in (7, 8, 16) and (v - ATOM_VALENCY.get(atom.GetSymbol(), -99)) == 1:
                atom.SetFormalCharge(1)
    return mol


# lower the highest non-aromatic bond order of the offending atom until the valence is valid
def correct_mol(mol):
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True
    while True:
        flag, info = check_valency(mol)
        if flag:
            break
        if info is None or len(info) != 2:
            return None, no_correct
        idx = info[0]
        queue, check_idx = [], 0
        for b in mol.GetAtomWithIdx(idx).GetBonds():
            t = int(b.GetBondType())
            queue.append((b.GetIdx(), t, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            if t == 12:
                check_idx += 1
        if not queue:
            return None, no_correct
        queue.sort(key=lambda tup: tup[1], reverse=True)
        if queue[-1][1] == 12:
            return None, no_correct
        start, end, t = queue[check_idx][2], queue[check_idx][3], queue[check_idx][1] - 1
        mol.RemoveBond(start, end)
        if t >= 1:
            mol.AddBond(start, end, ORDER_TO_BOND[t])
    return mol, no_correct


# keep the largest connected component
def valid_mol_can_with_seg(m):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if "." in sm:
        sm = max(sm.split("."), key=len)
    return Chem.MolFromSmiles(sm)


# list of (final molecule or None, valid without correction)
def graphs_to_mols(graphs: Sequence[Optional[MolGraph]], atoms, progress: bool = True):
    out = []
    for g in tqdm(graphs, desc="build molecules", disable=not progress):
        if g is None:
            out.append((None, False))
            continue
        try:
            mol, no_correct = correct_mol(construct_mol(g, atoms))
            out.append((valid_mol_can_with_seg(mol), no_correct))
        except Exception:
            out.append((None, False))
    return out


def mol_to_nx(mol):
    import networkx as nx
    G = nx.Graph()
    for a in mol.GetAtoms():
        G.add_node(a.GetIdx(), label=f"{a.GetAtomicNum()}_{a.GetFormalCharge()}_{a.GetNumExplicitHs()}")
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), label=str(b.GetBondTypeAsDouble()))
    return G


# linear-kernel MMD between NSPDK feature means
def nspdk_mmd(ref_mols, gen_mols) -> float:
    from eden.graph import vectorize
    X = vectorize([mol_to_nx(m) for m in tqdm(ref_mols, desc="nspdk ref")], complexity=4, discrete=True)
    Y = vectorize([mol_to_nx(m) for m in tqdm(gen_mols, desc="nspdk gen")], complexity=4, discrete=True)
    mx = np.asarray(X.mean(0)).ravel()
    my = np.asarray(Y.mean(0)).ravel()
    return float(np.sum((mx - my) ** 2))


# validity w/o correction (over all samples), validity after correction, unique, novelty, optional MOSES / NSPDK
def evaluate_graphs(graphs: Sequence[Optional[MolGraph]], atoms, train_smiles: Optional[Sequence[str]] = None,
                    test_smiles: Optional[Sequence[str]] = None, official: bool = False, nspdk: bool = False,
                    device: str = "cuda:0", n_jobs: int = 8, progress: bool = True) -> dict:
    res = graphs_to_mols(graphs, atoms, progress)
    N = max(len(graphs), 1)
    pairs = [(m, nc) for m, nc in res if m is not None]
    mols = [m for m, _ in pairs]
    smiles = [s for s in (Chem.MolToSmiles(m) for m in mols) if len(s)]
    n_nc = sum(nc for _, nc in pairs)
    out = {
        "num_samples": len(graphs),
        "validity_wo_cor": n_nc / N,
        "validity_wo_cor_geel": n_nc / max(len(pairs), 1),
        "validity_corrected": len(pairs) / N,
        "uniqueness": len(set(smiles)) / max(len(smiles), 1),
        "novelty": -1.0,
        "smiles": smiles,
    }
    if train_smiles is not None and smiles:
        train_set = set(train_smiles)
        uniq = set(smiles)
        out["novelty"] = sum(s not in train_set for s in uniq) / len(uniq)
    if official and smiles and test_smiles is not None:
        from moses.metrics.metrics import get_all_metrics
        scores = get_all_metrics(gen=smiles, n_jobs=n_jobs, device=device, test=list(test_smiles),
                                 train=list(train_smiles), k=len(smiles))
        out["moses"] = {k: float(v) for k, v in scores.items()}
    if nspdk and mols and test_smiles is not None:
        ref = [Chem.MolFromSmiles(s) for s in test_smiles]
        out["nspdk"] = nspdk_mmd([m for m in ref if m is not None], mols)
    return out


def summarize(res: dict) -> str:
    keys = ["validity_wo_cor", "validity_wo_cor_geel", "validity_corrected", "uniqueness", "novelty"]
    s = " | ".join(f"{k}={100 * res[k]:.2f}" for k in keys if k in res and res[k] != -1.0)
    if "nspdk" in res:
        s += f" | nspdk={res['nspdk']:.4f}"
    if "moses" in res:
        s += " | " + " | ".join(f"{k}={res['moses'][k]:.4f}" for k in ("FCD/Test", "Scaf/Test", "SNN/Test", "Frag/Test")
                                if k in res["moses"])
    return s


def load_eval_data(name: str, root: str):
    meta = prepare_dataset(name, root)
    return meta, load_smiles(meta["name"], root, "train"), load_smiles(meta["name"], root, "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_NAMES))
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--nspdk", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=8)
    a = ap.parse_args()

    meta, train_smiles, test_smiles = load_eval_data(a.dataset, a.root)
    graphs = pickle.load(open(a.graphs, "rb"))
    res = evaluate_graphs(graphs, meta["atoms"], train_smiles, test_smiles, a.official, a.nspdk, n_jobs=a.n_jobs)
    print(summarize(res))


if __name__ == "__main__":
    main()
