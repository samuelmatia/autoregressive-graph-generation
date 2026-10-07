# python -c "from src.data import prepare_dataset; prepare_dataset('qm9')"   # download + preprocess only
# python -c "from src.data import prepare_dataset; prepare_dataset('moses')"
# (train.py does this automatically on first run)

import hashlib
import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# special tokens
SOS, RESET, LADJ, RADJ, EOS, PAD = 0, 1, 2, 3, 4, 5
IDX_OFFSET = 6

BOND_IDX = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
NUM_EDGE_TYPES = 4

ATOM_DECODERS = {
    "qm9": ["H", "C", "N", "O", "F"],
    "moses": ["C", "N", "S", "O", "F", "Cl", "Br", "H"],
    "guacamol": ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"],
}


@dataclass
class MolGraph:
    x: np.ndarray            # (N,) atom types
    edge_index: np.ndarray   # (2, E) unique undirected edges, i < j
    edge_attr: np.ndarray    # (E,) bond types

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


# xorshift32 rng, state is an explicit array so it is safe in workers
@njit(cache=True)
def _rand_int(state, n):
    s = state[0]
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    s &= 0xFFFFFFFF
    state[0] = s
    return s % n


# Algorithm 1: sample a causal Hamiltonian labeled SENT, returns number of tokens written
@njit(cache=True)
def _sample_sent(x, rowptr, indices, eattr, idx_off, node_off, edge_off,
                 reset, ladj, radj, seed, out):
    n = x.shape[0]
    state = np.empty(1, np.int64)
    s0 = ((seed & 0x7FFFFFFF) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    if s0 == 0:
        s0 = 1
    state[0] = s0
    for _ in range(3):
        _rand_int(state, 2)

    unvisited = np.ones(n, np.uint8)
    nmap = np.full(n, -1, np.int64)
    nbr = np.empty(n, np.int64)
    nbr_k = np.empty(n, np.int64)
    ns_idx = np.empty(n, np.int64)
    ns_lab = np.empty(n, np.int64)

    p = 0
    remaining = n
    cur = _rand_int(state, n)
    unvisited[cur] = 0
    nmap[cur] = 0
    counter = 1
    remaining -= 1
    out[p] = idx_off
    p += 1
    out[p] = x[cur] + node_off
    p += 1

    while remaining > 0:
        prev = cur
        cnt = 0
        for k in range(rowptr[cur], rowptr[cur + 1]):
            nb = indices[k]
            if unvisited[nb] == 1:
                nbr[cnt] = nb
                nbr_k[cnt] = k
                cnt += 1
        if cnt == 0:
            # dead end: start a new trail at a random unvisited node
            out[p] = reset
            p += 1
            r = _rand_int(state, remaining)
            c = -1
            sel = 0
            for i in range(n):
                if unvisited[i] == 1:
                    c += 1
                    if c == r:
                        sel = i
                        break
            cur = sel
        else:
            # extend the trail with a random unvisited neighbor
            j = _rand_int(state, cnt)
            cur = nbr[j]
            out[p] = eattr[nbr_k[j]] + edge_off
            p += 1

        unvisited[cur] = 0
        nmap[cur] = counter
        counter += 1
        remaining -= 1
        out[p] = nmap[cur] + idx_off
        p += 1
        out[p] = x[cur] + node_off
        p += 1

        # neighborhood set: visited neighbors except the predecessor
        m = 0
        for k in range(rowptr[cur], rowptr[cur + 1]):
            nb = indices[k]
            if unvisited[nb] == 0 and nb != prev and nb != cur:
                ns_idx[m] = nmap[nb]
                ns_lab[m] = eattr[k]
                m += 1
        if m > 0:
            # insertion sort by node index
            for a in range(1, m):
                ki = ns_idx[a]
                kl = ns_lab[a]
                b = a - 1
                while b >= 0 and ns_idx[b] > ki:
                    ns_idx[b + 1] = ns_idx[b]
                    ns_lab[b + 1] = ns_lab[b]
                    b -= 1
                ns_idx[b + 1] = ki
                ns_lab[b + 1] = kl
            out[p] = ladj
            p += 1
            for a in range(m):
                out[p] = ns_lab[a] + edge_off
                p += 1
                out[p] = ns_idx[a] + idx_off
                p += 1
            out[p] = radj
            p += 1
    return p


class SentTokenizer:
    sos, reset, ladj, radj, eos, pad = SOS, RESET, LADJ, RADJ, EOS, PAD
    idx_offset = IDX_OFFSET

    def __init__(self, max_num_nodes: int, num_node_types: int, num_edge_types: int = NUM_EDGE_TYPES,
                 atom_decoder: Optional[List[str]] = None):
        self.max_num_nodes = int(max_num_nodes)
        self.num_node_types = int(num_node_types)
        self.num_edge_types = int(num_edge_types)
        self.atom_decoder = list(atom_decoder) if atom_decoder is not None else None
        self.node_idx_offset = self.idx_offset + self.max_num_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    # saved in checkpoints
    def meta(self) -> dict:
        return dict(max_num_nodes=self.max_num_nodes, num_node_types=self.num_node_types,
                    num_edge_types=self.num_edge_types, atom_decoder=self.atom_decoder)

    @classmethod
    def from_meta(cls, meta: dict) -> "SentTokenizer":
        return cls(meta["max_num_nodes"], meta["num_node_types"], meta["num_edge_types"],
                   meta.get("atom_decoder"))

    def __len__(self) -> int:
        return self.edge_idx_offset + self.num_edge_types

    @property
    def vocab_size(self) -> int:
        return len(self)

    # graph (CSR) -> tokens
    def encode(self, x, rowptr, indices, eattr, seed: int, add_special: bool = True) -> np.ndarray:
        n = int(x.shape[0])
        out = np.empty(6 * n + int(indices.shape[0]) + 8, dtype=np.int64)
        p = _sample_sent(x, rowptr, indices, eattr, self.idx_offset, self.node_idx_offset,
                         self.edge_idx_offset, self.reset, self.ladj, self.radj, int(seed), out)
        if not add_special:
            return out[:p].copy()
        res = np.empty(p + 2, dtype=np.int64)
        res[0] = self.sos
        res[1:p + 1] = out[:p]
        res[p + 1] = self.eos
        return res

    # tokens -> graph, tolerant to malformed sequences
    def decode(self, tokens) -> Optional[MolGraph]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        seq = [int(t) for t in tokens if t not in (self.sos, self.eos, self.pad)]
        io, no, eo = self.idx_offset, self.node_idx_offset, self.edge_idx_offset
        is_idx = lambda t: io <= t < no
        is_node = lambda t: no <= t < eo
        is_edge = lambda t: t >= eo

        labels: Dict[int, int] = {}
        edges: Dict[Tuple[int, int], int] = {}

        def add_edge(a, b, t):
            if a != b:
                edges.setdefault((min(a, b), max(a, b)), t)

        n, i = len(seq), 0
        prev_node, pending = None, None
        while i < n:
            t = seq[i]
            if t == self.reset:
                prev_node, pending = None, None
                i += 1
                continue
            if not is_idx(t):
                i += 1
                continue
            node = t - io
            i += 1
            if i < n and is_node(seq[i]):
                labels.setdefault(node, seq[i] - no)
                i += 1
            else:
                labels.setdefault(node, 0)
            if prev_node is not None and pending is not None:
                add_edge(prev_node, node, pending)
            pending = None
            if i < n and seq[i] == self.ladj:
                i += 1
                while i < n and seq[i] != self.radj:
                    if is_edge(seq[i]) and i + 1 < n and is_idx(seq[i + 1]):
                        nb = seq[i + 1] - io
                        labels.setdefault(nb, 0)
                        add_edge(node, nb, seq[i] - eo)
                        i += 2
                    else:
                        i += 1
                i += 1
            if i < n and is_edge(seq[i]):
                pending = seq[i] - eo
                i += 1
            prev_node = node

        if not labels:
            return None
        order = sorted(labels)
        remap = {old: new for new, old in enumerate(order)}
        x = np.array([labels[o] if 0 <= labels[o] < self.num_node_types else 0 for o in order], dtype=np.int64)
        if edges:
            ei = np.array([[remap[a], remap[b]] for (a, b) in edges], dtype=np.int64).T
            ea = np.array([t if 0 <= t < self.num_edge_types else 0 for t in edges.values()], dtype=np.int64)
        else:
            ei, ea = np.zeros((2, 0), np.int64), np.zeros((0,), np.int64)
        return MolGraph(x=x, edge_index=ei, edge_attr=ea)


Record = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # x, rowptr, indices, eattr


# all graphs of a split concatenated in flat CSR arrays
class GraphStore:
    FIELDS = ("x", "node_ptr", "rowptr", "indices", "edge_ptr", "eattr")

    def __init__(self, **arrays):
        for k in self.FIELDS:
            setattr(self, k, arrays[k])

    @classmethod
    def build(cls, records: Sequence[Record]) -> "GraphStore":
        node_ptr = np.zeros(len(records) + 1, np.int64)
        edge_ptr = np.zeros(len(records) + 1, np.int64)
        node_ptr[1:] = np.cumsum([len(r[0]) for r in records])
        edge_ptr[1:] = np.cumsum([len(r[2]) for r in records])
        return cls(
            x=np.concatenate([r[0] for r in records]).astype(np.int8),
            node_ptr=node_ptr,
            rowptr=np.concatenate([r[1] for r in records]).astype(np.int32),
            indices=np.concatenate([r[2] for r in records]).astype(np.int32),
            edge_ptr=edge_ptr,
            eattr=np.concatenate([r[3] for r in records]).astype(np.int8),
        )

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
        return (np.array(self.x[n0:n1], dtype=np.int64),
                np.array(self.rowptr[n0 + g:n1 + g + 1], dtype=np.int64),
                np.array(self.indices[e0:e1], dtype=np.int64),
                np.array(self.eattr[e0:e1], dtype=np.int64))


# a new random SENT is sampled at every access
class MolSENTDataset(Dataset):
    def __init__(self, store: GraphStore, tok: SentTokenizer, indices: Optional[np.ndarray] = None,
                 deterministic: bool = False, base_seed: int = 0):
        self.store, self.tok = store, tok
        self.indices = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.deterministic, self.base_seed = deterministic, base_seed
        self._rng = None

    def __len__(self):
        return len(self.indices)

    def _seed(self, g: int) -> int:
        # fixed seed per graph for validation, fresh seed per worker for training
        if self.deterministic:
            return int((g * 2654435761 + self.base_seed * 40503 + 12345) % (2 ** 31 - 1)) + 1
        if self._rng is None:
            self._rng = np.random.default_rng((torch.initial_seed() + self.base_seed) % (2 ** 32))
        return int(self._rng.integers(1, 2 ** 31 - 1))

    def __getitem__(self, i):
        g = int(self.indices[i])
        x, rowptr, indices, eattr = self.store.graph(g)
        return torch.from_numpy(self.tok.encode(x, rowptr, indices, eattr, self._seed(g)))


class Collator:
    def __init__(self, pad_id: int = PAD, truncation_length: Optional[int] = 2048, pad_multiple: int = 16):
        self.pad_id, self.trunc, self.mult = pad_id, truncation_length, pad_multiple

    # pad to a multiple of 16, random crop if too long
    def __call__(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(b) for b in batch)
        if self.trunc is not None:
            max_len = min(max_len, self.trunc)
        max_len = ((max_len + self.mult - 1) // self.mult) * self.mult
        out = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(batch):
            if self.trunc is not None and len(s) > self.trunc:
                st = int(torch.randint(0, len(s) - self.trunc, (1,)))
                s = s[st:st + self.trunc]
            out[i, :len(s)] = s
        return out


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


# RDKit mol -> CSR record, None if unsupported atom/bond or no bonds
def mol_to_record(mol, atom_index: Dict[str, int]) -> Optional[Record]:
    n = mol.GetNumAtoms()
    x = np.empty(n, np.int64)
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s not in atom_index:
            return None
        x[a.GetIdx()] = atom_index[s]
    nbrs = [[] for _ in range(n)]
    for b in mol.GetBonds():
        t = BOND_IDX.get(str(b.GetBondType()))
        if t is None:
            return None
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        nbrs[i].append((j, t))
        nbrs[j].append((i, t))
    rowptr = np.zeros(n + 1, np.int64)
    rowptr[1:] = np.cumsum([len(l) for l in nbrs])
    if rowptr[-1] == 0:
        return None
    indices = np.array([j for l in nbrs for j, _ in l], np.int64)
    eattr = np.array([t for l in nbrs for _, t in l], np.int64)
    return x, rowptr, indices, eattr


def record_to_graph(rec: Record) -> MolGraph:
    x, rowptr, indices, eattr = rec
    src = np.repeat(np.arange(len(x)), np.diff(rowptr))
    keep = src < indices
    return MolGraph(x=x, edge_index=np.stack([src[keep], indices[keep]]), edge_attr=eattr[keep])


_W: dict = {}


def _worker_init(atom_decoder, rebuild_filter):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _W["decoder"] = atom_decoder
    _W["index"] = {a: i for i, a in enumerate(atom_decoder)}
    _W["filter"] = rebuild_filter


def _smiles_worker(smi: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    rec = mol_to_record(mol, _W["index"])
    if rec is None:
        return None
    can = Chem.MolToSmiles(mol)
    if _W["filter"]:
        # guacamol: keep only molecules that rebuild from the graph and are connected
        from .evaluate import graph_to_mol, mol_to_smiles
        m = graph_to_mol(record_to_graph(rec), _W["decoder"], partial_charges=True)
        s = mol_to_smiles(m)
        if s is None:
            return None
        try:
            if len(Chem.rdmolops.GetMolFrags(m, asMols=True, sanitizeFrags=True)) != 1:
                return None
        except Exception:
            return None
        can = s
    return rec, can


def _process_smiles(smiles: Sequence[str], atom_decoder, rebuild_filter: bool, n_jobs: int, desc: str = ""):
    recs, kept = [], []
    with mp.Pool(n_jobs, initializer=_worker_init, initargs=(atom_decoder, rebuild_filter)) as pool:
        for res in tqdm(pool.imap(_smiles_worker, smiles, chunksize=2000), total=len(smiles), desc=desc):
            if res is not None:
                recs.append(res[0])
                kept.append(res[1])
    return recs, kept


def _write_split(pdir: str, split: str, recs: List[Record], smiles: Optional[List[str]] = None):
    GraphStore.build(recs).save(osp.join(pdir, split))
    if smiles is not None:
        with open(osp.join(pdir, f"{split}_smiles.txt"), "w") as f:
            f.write("\n".join(smiles))


def _prepare_qm9(root: str, n_jobs: int):
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from .evaluate import graph_to_mol, mol_to_smiles

    raw, pdir = osp.join(root, "qm9", "raw"), osp.join(root, "qm9", "processed")
    zpath = osp.join(raw, "qm9.zip")
    if not osp.exists(osp.join(raw, "gdb9.sdf")):
        _download("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
    _download("https://ndownloader.figshare.com/files/3195404", osp.join(raw, "uncharacterized.txt"))

    # same splits as DiGress: 100k train, 10% test, rest val
    df = pd.read_csv(osp.join(raw, "gdb9.sdf.csv"))
    n = len(df)
    n_train, n_test = 100000, int(0.1 * n)
    n_val = n - (n_train + n_test)
    shuf = df.sample(frac=1, random_state=42).index.values
    parts = dict(zip(("train", "val", "test"), np.split(shuf, [n_train, n_val + n_train])))
    owner = {int(i): s for s, ids in parts.items() for i in ids}

    with open(osp.join(raw, "uncharacterized.txt")) as f:
        skip = {int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]}

    atom_decoder = ATOM_DECODERS["qm9"]
    index = {a: i for i, a in enumerate(atom_decoder)}
    recs = {s: [] for s in parts}
    suppl = Chem.SDMolSupplier(osp.join(raw, "gdb9.sdf"), removeHs=False, sanitize=False)
    for i, mol in enumerate(tqdm(suppl, total=len(suppl), desc="qm9 sdf")):
        if mol is None or i in skip or i not in owner:
            continue
        rec = mol_to_record(mol, index)
        if rec is not None:
            recs[owner[i]].append(rec)

    # smiles rebuilt from the graph, used for novelty
    for s in ("train", "val", "test"):
        smi = []
        for rec in tqdm(recs[s], desc=f"qm9 {s} smiles"):
            sm = mol_to_smiles(graph_to_mol(record_to_graph(rec), atom_decoder, partial_charges=True))
            if sm is not None:
                smi.append(sm)
        _write_split(pdir, s, recs[s], smi)
    return {s: len(recs[s]) for s in recs}, recs["train"]


def _prepare_moses(root: str, n_jobs: int):
    import pandas as pd
    raw, pdir = osp.join(root, "moses", "raw"), osp.join(root, "moses", "processed")
    base = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/"
    # val = test_scaffolds.csv, test = test.csv
    files = {"train": "train.csv", "val": "test_scaffolds.csv", "test": "test.csv"}
    sizes, train_recs = {}, None
    for split, fn in files.items():
        _download(base + fn, osp.join(raw, f"{split}_moses.csv"))
        smi = pd.read_csv(osp.join(raw, f"{split}_moses.csv"))["SMILES"].values.tolist()
        recs, kept = _process_smiles(smi, ATOM_DECODERS["moses"], False, n_jobs, f"moses {split}")
        _write_split(pdir, split, recs, kept)
        sizes[split] = len(recs)
        if split == "train":
            train_recs = recs
    return sizes, train_recs


def _prepare_guacamol(root: str, n_jobs: int):
    raw, pdir = osp.join(root, "guacamol", "raw"), osp.join(root, "guacamol", "processed")
    urls = {"train": "https://figshare.com/ndownloader/files/13612760",
            "val": "https://figshare.com/ndownloader/files/13612766",
            "test": "https://figshare.com/ndownloader/files/13612757"}
    hashes = {"train": "05ad85d871958a05c02ab51a4fde8530", "val": "e53db4bff7dc4784123ae6df72e3b1f0",
              "test": "677b757ccec4809febd83850b43e1616"}
    sizes, train_recs = {}, None
    for split in ("train", "val", "test"):
        path = osp.join(raw, f"guacamol_v1_{split}.smiles")
        _download(urls[split], path)
        if hashlib.md5(open(path, "rb").read()).hexdigest() != hashes[split]:
            raise SystemExit(f"Bad hash for {path}")
        smi = [s for s in open(path).read().split("\n") if s.strip()]
        recs, kept = _process_smiles(smi, ATOM_DECODERS["guacamol"], True, n_jobs, f"guacamol {split}")
        _write_split(pdir, split, recs, kept)
        sizes[split] = len(recs)
        if split == "train":
            train_recs = recs
    return sizes, train_recs


_PREPARE = {"qm9": _prepare_qm9, "moses": _prepare_moses, "guacamol": _prepare_guacamol}
DATASET_NAMES = tuple(_PREPARE)


# download + process if needed, returns dataset meta
def prepare_dataset(name: str, root: str = "data", n_jobs: int = 8) -> dict:
    name = name.lower()
    if name not in _PREPARE:
        raise ValueError(f"Unknown dataset '{name}', choose from {DATASET_NAMES}")
    meta_path = osp.join(root, name, "processed", "meta.json")
    if osp.exists(meta_path):
        return json.load(open(meta_path))
    sizes, train_recs = _PREPARE[name](root, n_jobs)
    atom_decoder = ATOM_DECODERS[name]
    meta = dict(name=name, atom_decoder=atom_decoder, num_node_types=len(atom_decoder),
                num_edge_types=NUM_EDGE_TYPES,
                max_num_nodes=int(max(len(r[0]) for r in train_recs)), sizes=sizes)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[data] {name} ready: {meta}")
    return meta


def load_split(name: str, root: str, split: str) -> GraphStore:
    return GraphStore.load(osp.join(root, name.lower(), "processed", split))


def load_smiles(name: str, root: str, split: str = "train") -> List[str]:
    with open(osp.join(root, name.lower(), "processed", f"{split}_smiles.txt")) as f:
        return f.read().split("\n")


def build_tokenizer(meta: dict) -> SentTokenizer:
    return SentTokenizer(meta["max_num_nodes"], meta["num_node_types"], meta["num_edge_types"],
                         meta["atom_decoder"])


# returns (meta, tokenizer, train_ds, val_ds); exclude_idx removes train graphs (retain-only retraining)
def build_datasets(name: str, root: str = "data", n_jobs: int = 8, exclude_idx: Optional[np.ndarray] = None,
                   val_max: int = 10000, seed: int = 0):
    meta = prepare_dataset(name, root, n_jobs)
    tok = build_tokenizer(meta)
    train_store, val_store = load_split(name, root, "train"), load_split(name, root, "val")
    idx = np.arange(len(train_store))
    if exclude_idx is not None:
        idx = np.setdiff1d(idx, np.asarray(exclude_idx))
    vidx = np.arange(len(val_store))
    if len(vidx) > val_max:
        vidx = np.sort(np.random.RandomState(0).permutation(len(vidx))[:val_max])
    train_ds = MolSENTDataset(train_store, tok, idx, deterministic=False, base_seed=seed)
    val_ds = MolSENTDataset(val_store, tok, vidx, deterministic=True)
    return meta, tok, train_ds, val_ds


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, truncation_length: int = 2048,
                drop_last: bool = False, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
        collate_fn=Collator(PAD, truncation_length), pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, generator=g,
    )