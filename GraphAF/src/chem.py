# rdkit helpers shared by generation, evaluation and RL rewards (ported from the GraphAF environment, GCPN-style)

import copy
from typing import Optional

import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
_PT = Chem.GetPeriodicTable()
_SA = None
_ZINC = None


def atomic_numbers(atom_decoder):
    return [_PT.GetAtomicNumber(s) for s in atom_decoder]


# no atom exceeds its allowed valency
def check_valency(mol) -> bool:
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


# smiles round trip, implicit sanitization
def check_validity(mol) -> bool:
    try:
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) is not None
    except Exception:
        return False


def radicals_to_hydrogens(mol):
    m = copy.deepcopy(mol)
    if Descriptors.NumRadicalElectrons(m) == 0:
        return m
    for a in m.GetAtoms():
        r = a.GetNumRadicalElectrons()
        if r > 0:
            a.SetNumRadicalElectrons(0)
            a.SetNumExplicitHs(r)
    return m


# final smiles of a generated molecule, None if invalid
def final_smiles(mol) -> Optional[str]:
    try:
        if not check_validity(mol):
            return None
        s = Chem.MolToSmiles(radicals_to_hydrogens(mol), isomericSmiles=True)
        return s if Chem.MolFromSmiles(s) is not None else None
    except Exception:
        return None


def _sascorer():
    global _SA
    if _SA is None:
        try:
            from rdkit.Contrib.SA_Score import sascorer
        except Exception:
            from . import sascorer
        _SA = sascorer
    return _SA


def sa_score(mol) -> float:
    return _sascorer().calculateScore(mol)


def qed(mol) -> float:
    return QED.qed(mol)


# logP penalized by SA and long cycles (Kusner et al. 2017), normalized on ZINC250k statistics
def penalized_logp(mol) -> float:
    logp_mean, logp_std = 2.4570953396190123, 1.434324401111988
    sa_mean, sa_std = -3.0525811293166134, 0.8335207024513095
    cyc_mean, cyc_std = -0.0485696876403053, 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    sa = -sa_score(mol)
    cycles = nx.cycle_basis(nx.from_numpy_array(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    longest = max((len(c) for c in cycles), default=0)
    cycle_score = -(longest - 6 if longest > 6 else 0)
    return (log_p - logp_mean) / logp_std + (sa - sa_mean) / sa_std + (cycle_score - cyc_mean) / cyc_std


# plogp is not invariant to the smiles flavor, take the minimum over them (as in the official code)
def min_penalized_logp(mol) -> float:
    vals = [penalized_logp(mol)]
    for iso in (True, False):
        vals.append(penalized_logp(Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=iso))))
    return min(vals)


# Tanimoto similarity of Morgan fingerprints (radius 2, 2048 bits)
def similarity(a, b) -> float:
    fa = rdMolDescriptors.GetMorganFingerprintAsBitVect(a, radius=2, nBits=2048, useChirality=True)
    fb = rdMolDescriptors.GetMorganFingerprintAsBitVect(b, radius=2, nBits=2048, useChirality=True)
    return DataStructs.TanimotoSimilarity(fa, fb)


# True if the molecule embeds and its minimized average angle strain is below the cutoff
def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20, max_num_iters=200) -> bool:
    if mol.GetNumAtoms() <= 2:
        return True
    m_h = Chem.AddHs(copy.deepcopy(mol))
    try:
        if AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed) == -1:
            return False
        AllChem.MMFFSanitizeMolecule(m_h)
        if not AllChem.MMFFHasAllMoleculeParams(m_h):
            return False
        props = AllChem.MMFFGetMoleculeProperties(m_h)
        ff = AllChem.MMFFGetMoleculeForceField(m_h, props)
        ff.Minimize(maxIts=max_num_iters)
        props.SetMMFFBondTerm(False)
        props.SetMMFFAngleTerm(True)
        props.SetMMFFStretchBendTerm(False)
        props.SetMMFFOopTerm(False)
        props.SetMMFFTorsionTerm(False)
        props.SetMMFFVdWTerm(False)
        props.SetMMFFEleTerm(False)
        ff = AllChem.MMFFGetMoleculeForceField(m_h, props)
        angle_e = ff.CalcEnergy()
        n_angles = 0
        for atom in m_h.GetAtoms():
            nb = [n.GetIdx() for n in atom.GetNeighbors()]
            for p in range(len(nb)):
                for q in range(p + 1, len(nb)):
                    if props.GetMMFFAngleBendParams(m_h, nb[p], atom.GetIdx(), nb[q]):
                        n_angles += 1
        return angle_e / max(n_angles, 1) < cutoff
    except Exception:
        return False


# True if the molecule matches none of the ZINC problematic functional groups
def zinc_molecule_filter(mol) -> bool:
    global _ZINC
    if _ZINC is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
        _ZINC = FilterCatalog(params)
    return not _ZINC.HasMatch(mol)
