# python -c "from src.data import prepare_dataset; prepare_dataset('zinc250k')"   # download + preprocess only
# python -c "from src.data import prepare_dataset; prepare_dataset('qm9')"
# python -c "from src.data import prepare_dataset; prepare_dataset('moses')"
# (train.py does this automatically on first run)

import json
import multiprocessing as mp
import os
import os.path as osp
import urllib.request
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ATOM_DECODERS = {
    "zinc250k": ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"],
    "qm9": ["C", "N", "O", "F"],
    "moses": ["C", "N", "S", "O", "F", "Cl", "Br"],
}
BOND_IDX = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}
NUM_BOND_TYPES = 4          # single, double, triple + "no bond"
EDGE_UNROLL = 12            # bonds are modeled against the previous 12 nodes in BFS order
VAL_SIZE = 2000

ZINC_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/"
            "zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/"
QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"


def _download(url: str, dst: str):
    if osp.exists(dst):
        return
    os.makedirs(osp.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=osp.basename(dst)) as bar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(dst + ".part", dst)


# all graphs of a split concatenated in flat arrays, edges stored once as (i, j, bond type)
class GraphStore:
    FIELDS = ("x", "node_ptr", "edges", "edge_ptr")

    def __init__(self, **arrays):
        for k in self.FIELDS:
            setattr(self, k, arrays[k])

    @classmethod
    def build(cls, records: Sequence[Tuple[np.ndarray, np.ndarray]]) -> "GraphStore":
        node_ptr = np.zeros(len(records) + 1, np.int64)
        edge_ptr = np.zeros(len(records) + 1, np.int64)
        node_ptr[1:] = np.cumsum([len(r[0]) for r in records])
        edge_ptr[1:] = np.cumsum([len(r[1]) for r in records])
        return cls(x=np.concatenate([r[0] for r in records]).astype(np.int8), node_ptr=node_ptr,
                   edges=np.concatenate([r[1] for r in records]).astype(np.int8), edge_ptr=edge_ptr)

    def save(self, d: str):
        os.makedirs(d, exist_ok=True)
        for k in self.FIELDS:
            np.save(osp.join(d, f"{k}.npy"), getattr(self, k))

    @classmethod
    def load(cls, d: str) -> "GraphStore":
        return cls(**{k: np.load(osp.join(d, f"{k}.npy"), mmap_mode="r") for k in cls.FIELDS})

    def __len__(self) -> int:
        return len(self.node_ptr) - 1

    def num_nodes(self) -> np.ndarray:
        return np.diff(np.asarray(self.node_ptr))

    def graph(self, g: int):
        n0, n1 = int(self.node_ptr[g]), int(self.node_ptr[g + 1])
        e0, e1 = int(self.edge_ptr[g]), int(self.edge_ptr[g + 1])
        return np.array(self.x[n0:n1], dtype=np.int64), np.array(self.edges[e0:e1], dtype=np.int64)


# random BFS: random start, neighbors visited in a random order
def bfs_order(n: int, edges: np.ndarray, rng: np.random.Generator) -> List[int]:
    nbrs = [[] for _ in range(n)]
    for i, j, _ in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    prio = rng.permutation(n)
    start = int(rng.integers(n))
    seen = np.zeros(n, bool)
    seen[start] = True
    order, head = [start], 0
    while head < len(order):
        for v in sorted(nbrs[order[head]], key=lambda v: prio[v]):
            if not seen[v]:
                seen[v] = True
                order.append(v)
        head += 1
    return order


# types (max_size,) int8 with -1 padding, bonds (max_size, max_size) uint8 with 0 = no bond, 1..3 = bond type + 1
def pack_graph(types: np.ndarray, edges: np.ndarray, order: Sequence[int], max_size: int):
    n = len(order)
    pos = np.empty(len(types), np.int64)
    pos[list(order)] = np.arange(n)
    t = np.full(max_size, -1, np.int8)
    t[:n] = types[list(order)]
    b = np.zeros((max_size, max_size), np.uint8)
    for i, j, k in edges:
        b[pos[i], pos[j]] = b[pos[j], pos[i]] = k + 1
    return t, b


# a new random BFS order is sampled at every access
class MolBFSDataset(Dataset):
    def __init__(self, store: GraphStore, max_size: int, indices: Optional[np.ndarray] = None,
                 deterministic: bool = False, base_seed: int = 0):
        self.store, self.max_size = store, max_size
        self.indices = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.deterministic, self.base_seed = deterministic, base_seed
        self._rng = None

    def __len__(self):
        return len(self.indices)

    def _generator(self, g: int) -> np.random.Generator:
        # fixed order per graph for validation, fresh stream per worker for training
        if self.deterministic:
            return np.random.default_rng((g * 2654435761 + self.base_seed * 40503 + 12345) % (2 ** 32))
        if self._rng is None:
            self._rng = np.random.default_rng((torch.initial_seed() + self.base_seed) % (2 ** 32))
        return self._rng

    def raw(self, i: int, rng: Optional[np.random.Generator] = None):
        g = int(self.indices[i])
        types, edges = self.store.graph(g)
        rng = rng or self._generator(g)
        return pack_graph(types, edges, bfs_order(len(types), edges, rng), self.max_size)

    def __getitem__(self, i):
        t, b = self.raw(i)
        return torch.from_numpy(t), torch.from_numpy(b)


def _smiles_worker(smi: str):
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumAtoms() > _W["max_atoms"]:
            return None
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        types = np.empty(mol.GetNumAtoms(), np.int64)
        for a in mol.GetAtoms():
            k = _W["index"].get(a.GetSymbol())
            if k is None:
                return None
            types[a.GetIdx()] = k
        can = Chem.MolToSmiles(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        edges = []
        for b in mol.GetBonds():
            t = BOND_IDX.get(str(b.GetBondType()))
            if t is None:
                return None
            edges.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), t))
        if not edges:
            return None
        return (types, np.array(edges, np.int64)), can
    except Exception:
        return None


_W: dict = {}


def _worker_init(atom_decoder, max_atoms):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _W["index"] = {a: i for i, a in enumerate(atom_decoder)}
    _W["max_atoms"] = max_atoms


def _process_smiles(smiles: Sequence[str], atom_decoder, max_atoms: int, n_jobs: int, desc: str):
    recs, kept = [], []
    with mp.Pool(n_jobs, initializer=_worker_init, initargs=(atom_decoder, max_atoms)) as pool:
        for res in tqdm(pool.imap(_smiles_worker, smiles, chunksize=1000), total=len(smiles), desc=desc):
            if res is not None:
                recs.append(res[0])
                kept.append(res[1])
    return recs, kept


def _raw_smiles(name: str, root: str) -> List[str]:
    import pandas as pd
    raw = osp.join(root, name, "raw")
    if name == "zinc250k":
        path = osp.join(raw, "250k_rndm_zinc_drugs_clean_3.csv")
        _download(ZINC_URL, path)
        return [s.strip() for s in pd.read_csv(path)["smiles"].values]
    if name == "moses":
        out = []
        for fn in ("train.csv", "test.csv", "test_scaffolds.csv"):
            _download(MOSES_URL + fn, osp.join(raw, fn))
            out += pd.read_csv(osp.join(raw, fn))["SMILES"].values.tolist()
        return out
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    zpath = osp.join(raw, "qm9.zip")
    if not osp.exists(osp.join(raw, "gdb9.sdf")):
        _download(QM9_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
    out = []
    suppl = Chem.SDMolSupplier(osp.join(raw, "gdb9.sdf"), removeHs=True, sanitize=True)
    for mol in tqdm(suppl, total=len(suppl), desc="qm9 sdf"):
        if mol is not None:
            out.append(Chem.MolToSmiles(mol))
    return out


MAX_ATOMS = {"zinc250k": 38, "qm9": 9, "moses": 30}


# download + process if needed, returns dataset meta
def prepare_dataset(name: str, root: str = "data", n_jobs: int = 8, seed: int = 42) -> dict:
    name = name.lower()
    if name not in ATOM_DECODERS:
        raise ValueError(f"Unknown dataset '{name}', choose from {tuple(ATOM_DECODERS)}")
    pdir = osp.join(root, name, "processed")
    meta_path = osp.join(pdir, "meta.json")
    if osp.exists(meta_path):
        return json.load(open(meta_path))

    smiles = _raw_smiles(name, root)
    recs, kept = _process_smiles(smiles, ATOM_DECODERS[name], MAX_ATOMS[name], n_jobs, f"{name} preprocess")
    perm = np.random.RandomState(seed).permutation(len(recs))
    parts = {"val": perm[:VAL_SIZE], "train": perm[VAL_SIZE:]}
    for split, ids in parts.items():
        GraphStore.build([recs[i] for i in ids]).save(osp.join(pdir, split))
        with open(osp.join(pdir, f"{split}_smiles.txt"), "w") as f:
            f.write("\n".join(kept[i] for i in ids))

    max_size = int(max(len(r[0]) for r in recs))
    ds = MolBFSDataset(GraphStore.load(osp.join(pdir, "train")), max_size, deterministic=True)
    window = 0
    for i in tqdm(range(min(len(ds), 20000)), desc="bfs window"):
        t, b = ds.raw(i)
        ii, jj = np.nonzero(np.triu(b, 1))
        if len(ii):
            window = max(window, int((jj - ii).max()))
    meta = dict(name=name, atom_decoder=ATOM_DECODERS[name], node_dim=len(ATOM_DECODERS[name]),
                bond_dim=NUM_BOND_TYPES, max_size=max_size, edge_unroll=min(EDGE_UNROLL, max_size),
                max_bfs_window=window, sizes={s: len(ids) for s, ids in parts.items()})
    json.dump(meta, open(meta_path, "w"), indent=2)
    if window > meta["edge_unroll"]:
        print(f"[data] warning: bonds span up to {window} nodes in BFS order, only {meta['edge_unroll']} are modeled")
    print(f"[data] {name} ready: {meta}")
    return meta


def load_split(name: str, root: str, split: str) -> GraphStore:
    return GraphStore.load(osp.join(root, name.lower(), "processed", split))


def load_smiles(name: str, root: str, split: str = "train") -> List[str]:
    with open(osp.join(root, name.lower(), "processed", f"{split}_smiles.txt")) as f:
        return f.read().split("\n")


# returns (meta, train_ds, val_ds); exclude_idx removes train graphs (retain-only retraining)
def build_datasets(name: str, root: str = "data", n_jobs: int = 8, exclude_idx: Optional[np.ndarray] = None,
                   val_max: int = VAL_SIZE, seed: int = 0):
    meta = prepare_dataset(name, root, n_jobs)
    train_store, val_store = load_split(name, root, "train"), load_split(name, root, "val")
    idx = np.arange(len(train_store))
    if exclude_idx is not None:
        idx = np.setdiff1d(idx, np.asarray(exclude_idx))
    vidx = np.arange(min(len(val_store), val_max))
    train_ds = MolBFSDataset(train_store, meta["max_size"], idx, deterministic=False, base_seed=seed)
    val_ds = MolBFSDataset(val_store, meta["max_size"], vidx, deterministic=True)
    return meta, train_ds, val_ds


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False,
                seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None, generator=g)
