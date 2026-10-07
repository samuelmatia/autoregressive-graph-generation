# python -m src.evaluate --dataset qm9 --graphs logs/qm9/s/gen/graphs.pkl
# python -m src.evaluate --dataset moses --graphs logs/moses/s/gen/graphs.pkl --official

import argparse
import json
import os.path as osp
import pickle
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

from .data import MolGraph, load_smiles, prepare_dataset, ATOM_DECODERS

RDLogger.DisableLog("rdApp.*")

BOND_DICT = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
ALLOWED_BONDS = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Al": 3, "Si": 4, "P": [3, 5],
                 "S": 4, "Cl": 1, "As": 3, "Br": 1, "I": 1, "Hg": [1, 2], "Bi": [3, 5], "Se": [2, 4, 6]}


def _check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        msg = str(e)
        return False, [int(v) for v in re.findall(r"\d+", msg[msg.find("#"):])]


# partial_charges=True adds +1 to over-valent N/O/S (DiGress relaxation)
def graph_to_mol(g: MolGraph, atom_decoder: Sequence[str], partial_charges: bool = False):
    mol = Chem.RWMol()
    for a in g.x:
        mol.AddAtom(Chem.Atom(atom_decoder[int(a)]))
    for i, j, t in zip(g.edge_index[0], g.edge_index[1], g.edge_attr):
        mol.AddBond(int(i), int(j), BOND_DICT[int(t)])
        if partial_charges:
            ok, info = _check_valency(mol)
            if not ok and info is not None and len(info) == 2:
                idx, v = info
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def mol_to_smiles(mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def _largest_fragment_smiles(mol) -> Optional[str]:
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        return mol_to_smiles(max(frags, default=mol, key=lambda m: m.GetNumAtoms()))
    except Exception:
        return None


# atom and molecule stability (QM9 with hydrogens)
def stability(graphs: Sequence[Optional[MolGraph]], atom_decoder: Sequence[str]) -> Dict[str, float]:
    mol_stable = n_stable_atoms = n_atoms = 0
    for g in graphs:
        if g is None:
            continue
        bonds = np.zeros(g.num_nodes, dtype=np.int64)
        for (i, j), t in zip(g.edge_index.T, g.edge_attr):
            bonds[i] += t + 1
            bonds[j] += t + 1
        ok = 0
        for a, b in zip(g.x, bonds):
            allowed = ALLOWED_BONDS[atom_decoder[int(a)]]
            ok += int(b == allowed) if isinstance(allowed, int) else int(b in allowed)
        n_stable_atoms += ok
        n_atoms += g.num_nodes
        mol_stable += int(ok == g.num_nodes)
    return {"mol_stable": mol_stable / max(len(graphs), 1), "atm_stable": n_stable_atoms / max(n_atoms, 1)}


# validity (largest fragment), relaxed validity, uniqueness, novelty, optional stability
def evaluate_graphs(graphs: Sequence[Optional[MolGraph]], atom_decoder: Sequence[str],
                    train_smiles: Optional[Sequence[str]] = None, check_stability: bool = False,
                    progress: bool = True) -> dict:
    strict, relaxed, ncomp = [], [], []
    for g in tqdm(graphs, desc="evaluate", disable=not progress):
        if g is None or g.num_nodes == 0:
            strict.append(None)
            relaxed.append(None)
            continue
        m = graph_to_mol(g, atom_decoder, partial_charges=False)
        try:
            ncomp.append(len(Chem.GetMolFrags(m)))
        except Exception:
            pass
        strict.append(_largest_fragment_smiles(m) if mol_to_smiles(m) is not None else None)
        m2 = graph_to_mol(g, atom_decoder, partial_charges=True)
        relaxed.append(_largest_fragment_smiles(m2) if mol_to_smiles(m2) is not None else None)

    n = max(len(graphs), 1)
    valid = [s for s in strict if s is not None]
    rvalid = [s for s in relaxed if s is not None]
    uniq = set(rvalid)
    out = {
        "num_samples": len(graphs),
        "validity": len(valid) / n,
        "relaxed_validity": len(rvalid) / n,
        "uniqueness": len(uniq) / len(rvalid) if rvalid else 0.0,
        "novelty": -1.0,
        "nc_mu": float(np.mean(ncomp)) if ncomp else 0.0,
        "nc_max": int(np.max(ncomp)) if ncomp else 0,
        "smiles": strict,
        "relaxed_smiles": relaxed,
    }
    if train_smiles is not None and uniq:
        train_set = set(train_smiles)
        out["novelty"] = sum(s not in train_set for s in uniq) / len(uniq)
    if check_stability:
        out.update(stability([g for g in graphs if g is not None], atom_decoder))
    return out


def summarize(res: dict) -> str:
    keys = ["validity", "relaxed_validity", "uniqueness", "novelty", "mol_stable", "atm_stable", "nc_mu"]
    return " | ".join(f"{k}={100 * res[k]:.2f}" if k != "nc_mu" else f"{k}={res[k]:.2f}"
                      for k in keys if k in res and res[k] != -1.0)


# invalid molecules stay in the list so they count in validity
def _clean_for_official(smiles: Sequence[Optional[str]]) -> List[str]:
    return [s if s is not None else "invalid" for s in smiles]


def official_moses(smiles: Sequence[Optional[str]], root: str = "data", n_jobs: int = 8,
                   device: str = "cuda:0") -> dict:
    import moses
    import pandas as pd
    raw = osp.join(root, "moses", "raw")
    rd = lambda n: pd.read_csv(osp.join(raw, n))["SMILES"].values
    # val_moses.csv is test_scaffolds.csv, test_moses.csv is test.csv
    return moses.get_all_metrics(_clean_for_official(smiles), n_jobs=n_jobs, device=device,
                                 test=rd("test_moses.csv"), test_scaffolds=rd("val_moses.csv"),
                                 train=rd("train_moses.csv"))


def official_guacamol(smiles: Sequence[Optional[str]], root: str = "data", out_json: str = "guacamol_dist.json") -> dict:
    from guacamol.assess_distribution_learning import assess_distribution_learning
    from guacamol.distribution_matching_generator import DistributionMatchingGenerator
    from .data import _download

    class Mock(DistributionMatchingGenerator):
        def __init__(self, mols):
            self.mols, self.cursor = list(mols), 0

        def generate(self, number_samples):
            out = self.mols[self.cursor:self.cursor + number_samples]
            self.cursor += number_samples
            return out

    chembl = osp.join(root, "guacamol", "raw", "guacamol_v1_all.smiles")
    _download("https://figshare.com/ndownloader/files/13612745", chembl)
    # each benchmark consumes ~10k valid molecules, generate at least 50-60k
    assess_distribution_learning(Mock(_clean_for_official(smiles)), chembl_training_file=chembl,
                                 json_output_file=out_json, benchmark_version="v1")
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ATOM_DECODERS))
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=8)
    a = ap.parse_args()

    meta = prepare_dataset(a.dataset, a.root)
    graphs = pickle.load(open(a.graphs, "rb"))
    res = evaluate_graphs(graphs, meta["atom_decoder"], load_smiles(a.dataset, a.root, "train"),
                          check_stability=(a.dataset == "qm9"))
    print(summarize(res))
    if a.official:
        smi = res["smiles"]
        if a.dataset == "moses":
            print(official_moses(smi, a.root, a.n_jobs))
        elif a.dataset == "guacamol":
            print(official_guacamol(smi, a.root))
        else:
            print("No official benchmark for QM9, use the metrics above (DiGress protocol)")


if __name__ == "__main__":
    main()