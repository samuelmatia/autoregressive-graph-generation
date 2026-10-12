# python -c "from src.data import prepare_dataset; prepare_dataset('qm9')"
# python -c "from src.data import prepare_dataset; prepare_dataset('zinc')"
# python -c "from src.data import prepare_dataset; prepare_dataset('moses')"
# (train.py does this automatically on first run)

import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


PAD, BOS, EOS = 0, 1, 2
EDGE_OFFSET = 3
BOND_IDX = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
NUM_BOND_TYPES = 4
BASE_ORDERS = 4

GDSS_URL = "https://raw.githubusercontent.com/harryjo97/GDSS/master/data/"
GDSS_FILES = {"qm9": ("qm9.csv", "valid_idx_qm9.json"), "zinc": ("zinc250k.csv", "valid_idx_zinc250k.json")}
MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/"
VAL_FRAC = {"qm9": 0.05, "zinc": 0.05, "moses": 0.01}
DATASET_NAMES = ("qm9", "zinc", "moses")


@dataclass
class MolGraph:
    x: np.ndarray            # (N,) atom token ids
    edge_index: np.ndarray   # (2, E) with i < j
    edge_attr: np.ndarray    # (E,) bond ids

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


# xorshift32 with an explicit state array
@njit(cache=True)
def _rand_int(state, n):
    s = state[0]
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    s &= 0xFFFFFFFF
    state[0] = s
    return s % n


@njit(cache=True)
def _init_state(seed):
    state = np.empty(1, np.int64)
    s0 = ((seed & 0x7FFFFFFF) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    if s0 == 0:
        s0 = 1
    state[0] = s0
    for _ in range(3):
        _rand_int(state, 2)
    return state


# BFS distances from src, queue holds discovery order, returns number reached
@njit(cache=True)
def _bfs(rowptr, indices, src, dist, queue):
    n = dist.shape[0]
    for i in range(n):
        dist[i] = -1
    dist[src] = 0
    queue[0] = src
    head = 0
    tail = 1
    while head < tail:
        u = queue[head]
        head += 1
        for k in range(rowptr[u], rowptr[u + 1]):
            w = indices[k]
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                queue[tail] = w
                tail += 1
    return tail


# random Cuthill-McKee order from a pseudo-peripheral start, returns number of ordered nodes
@njit(cache=True)
def _cm_order(rowptr, indices, seed, order):
    n = rowptr.shape[0] - 1
    state = _init_state(seed)
    dist = np.empty(n, np.int64)
    queue = np.empty(n, np.int64)
    v = _rand_int(state, n)
    lp = 0
    while True:
        cnt = _bfs(rowptr, indices, v, dist, queue)
        far = 0
        for i in range(cnt):
            if dist[queue[i]] > far:
                far = dist[queue[i]]
        if far <= lp:
            break
        lp = far
        best = -1
        bdeg = 1 << 30
        for i in range(cnt):
            w = queue[i]
            if dist[w] == far:
                d = rowptr[w + 1] - rowptr[w]
                if d < bdeg:
                    bdeg = d
                    best = w
        v = best

    visited = np.zeros(n, np.uint8)
    ch = np.empty(n, np.int64)
    chd = np.empty(n, np.int64)
    chr_ = np.empty(n, np.int64)
    visited[v] = 1
    queue[0] = v
    qh = 0
    qt = 1
    pos = 0
    while qh < qt:
        parent = queue[qh]
        qh += 1
        order[pos] = parent
        pos += 1
        m = 0
        for k in range(rowptr[parent], rowptr[parent + 1]):
            w = indices[k]
            if visited[w] == 0:
                ch[m] = w
                chd[m] = rowptr[w + 1] - rowptr[w]
                chr_[m] = _rand_int(state, 1 << 30)
                m += 1
        # sort children by (degree, random)
        for a in range(1, m):
            kw = ch[a]
            kd = chd[a]
            kr = chr_[a]
            b = a - 1
            while b >= 0 and (chd[b] > kd or (chd[b] == kd and chr_[b] > kr)):
                ch[b + 1] = ch[b]
                chd[b + 1] = chd[b]
                chr_[b + 1] = chr_[b]
                b -= 1
            ch[b + 1] = kw
            chd[b + 1] = kd
            chr_[b + 1] = kr
        for a in range(m):
            visited[ch[a]] = 1
            queue[qt] = ch[a]
            qt += 1
    return pos


@njit(cache=True)
def _bandwidth(rowptr, indices, order):
    n = order.shape[0]
    rank = np.empty(n, np.int64)
    for i in range(n):
        rank[order[i]] = i
    bw = 0
    for u in range(n):
        for k in range(rowptr[u], rowptr[u + 1]):
            d = abs(rank[indices[k]] - rank[u])
            if d > bw:
                bw = d
    return bw


# gap encoded edge list with node / bond types, returns length or -1 if a gap exceeds bw
@njit(cache=True)
def _encode(x, rowptr, indices, eattr, order, bw, node_off, bond_off, out):
    n = x.shape[0]
    rank = np.empty(n, np.int64)
    for i in range(n):
        rank[order[i]] = i
    tg = np.empty(n, np.int64)
    tb = np.empty(n, np.int64)
    out[0] = 1
    p = 1
    for s in range(n):
        old = order[s]
        m = 0
        for k in range(rowptr[old], rowptr[old + 1]):
            w = rank[indices[k]]
            if w > s:
                tg[m] = w
                tb[m] = eattr[k]
                m += 1
        for a in range(1, m):
            kt = tg[a]
            kb = tb[a]
            b = a - 1
            while b >= 0 and tg[b] > kt:
                tg[b + 1] = tg[b]
                tb[b + 1] = tb[b]
                b -= 1
            tg[b + 1] = kt
            tb[b + 1] = kb
        out[p] = node_off + x[old]
        p += 1
        if m == 0:
            # dummy self-loop for nodes without larger neighbors
            out[p] = 3
            p += 1
            out[p] = bond_off
            p += 1
        else:
            for j in range(m):
                gap = tg[j] - s
                if gap > bw:
                    return -1
                ni = 1 if j == 0 else 0
                out[p] = 2 + 2 * gap + ni
                p += 1
                out[p] = bond_off + tb[j]
                p += 1
    out[p] = 2
    p += 1
    return p


# vocabulary: [pad, bos, eos] + edge tuples (ni, gap) without (0, 0) + atom tokens + bond tokens
class GeelTokenizer:
    pad, bos, eos = PAD, BOS, EOS
    edge_offset = EDGE_OFFSET

    def __init__(self, bw: int, atoms: Sequence[Sequence[int]], max_nodes: int, max_len: int):
        self.bw = int(bw)
        self.atoms = [tuple(int(v) for v in a) for a in atoms]
        self.max_nodes = int(max_nodes)
        self.max_len = int(max_len)
        self.num_atoms = len(self.atoms)
        self.node_offset = EDGE_OFFSET + 2 * self.bw + 1
        self.bond_offset = self.node_offset + self.num_atoms

    def meta(self) -> dict:
        return dict(bw=self.bw, atoms=[list(a) for a in self.atoms], max_nodes=self.max_nodes, max_len=self.max_len)

    @classmethod
    def from_meta(cls, meta: dict) -> "GeelTokenizer":
        return cls(meta["bw"], meta["atoms"], meta["max_nodes"], meta["max_len"])

    def __len__(self) -> int:
        return self.bond_offset + NUM_BOND_TYPES

    @property
    def vocab_size(self) -> int:
        return len(self)

    @staticmethod
    def edge_id(ni: int, gap: int) -> int:
        return 2 + 2 * gap + ni

    # graph (CSR) -> tokens, retries other orderings if the bandwidth exceeds the vocabulary
    def encode(self, x, rowptr, indices, eattr, seed: int) -> np.ndarray:
        n = int(x.shape[0])
        m = int(indices.shape[0]) // 2
        out = np.empty(3 * n + 2 * m + 4, np.int64)
        order = np.empty(n, np.int64)
        seeds = [seed + i * 104729 for i in range(8)] + list(range(1, BASE_ORDERS + 1))
        for s in seeds:
            if _cm_order(rowptr, indices, s, order) != n:
                continue
            p = _encode(x, rowptr, indices, eattr, order, self.bw, self.node_offset, self.bond_offset, out)
            if p > 0:
                return out[:p].copy()
        raise ValueError("no ordering fits the bandwidth vocabulary")

    # tokens -> graph, None if the sequence breaks the grammar
    def decode(self, tokens) -> Optional[MolGraph]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        seq = [int(t) for t in tokens]
        if not seq or seq[0] != BOS or EOS not in seq:
            return None
        seq = seq[1:seq.index(EOS)]
        if BOS in seq or PAD in seq:
            return None
        no, bo = self.node_offset, self.bond_offset
        state, src, pending = 0, -1, None
        atoms: List[int] = []
        edges = {}
        for t in seq:
            if state == 0:
                if not (no <= t < bo):
                    return None
                atoms.append(t - no)
                state = 1
            elif state == 1:
                if not (EDGE_OFFSET <= t < no) or ((t - 2) & 1) != 1:
                    return None
                src += 1
                if src != len(atoms) - 1:
                    return None
                gap = (t - 2) >> 1
                pending = (src, src + gap) if gap > 0 else None
                state = 2
            elif state == 2:
                if not (bo <= t < bo + NUM_BOND_TYPES):
                    return None
                if pending is not None:
                    edges[pending] = t - bo
                pending = None
                state = 3
            else:
                if no <= t < bo:
                    atoms.append(t - no)
                    state = 1
                elif EDGE_OFFSET <= t < no and ((t - 2) & 1) == 0:
                    pending = (src, src + ((t - 2) >> 1))
                    state = 2
                else:
                    return None
        if state != 3 or len(atoms) < 2 or not edges:
            return None
        if max(max(e) for e in edges) + 1 != len(atoms):
            return None
        ei = np.array(list(edges.keys()), dtype=np.int64).T
        ea = np.array(list(edges.values()), dtype=np.int64)
        return MolGraph(x=np.array(atoms, dtype=np.int64), edge_index=ei, edge_attr=ea)


Record = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# all graphs of a split in flat CSR arrays
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
            x=np.concatenate([r[0] for r in records]).astype(np.int16),
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

    def num_edges(self) -> np.ndarray:
        return np.diff(np.asarray(self.edge_ptr)) // 2

    def graph(self, g: int):
        n0, n1 = int(self.node_ptr[g]), int(self.node_ptr[g + 1])
        e0, e1 = int(self.edge_ptr[g]), int(self.edge_ptr[g + 1])
        return (np.array(self.x[n0:n1], dtype=np.int64),
                np.array(self.rowptr[n0 + g:n1 + g + 1], dtype=np.int64),
                np.array(self.indices[e0:e1], dtype=np.int64),
                np.array(self.eattr[e0:e1], dtype=np.int64))


# a new random Cuthill-McKee ordering is sampled at every access
class MolGEELDataset(Dataset):
    def __init__(self, store: GraphStore, tok: GeelTokenizer, indices: Optional[np.ndarray] = None,
                 deterministic: bool = False, base_seed: int = 0):
        self.store, self.tok = store, tok
        self.indices = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.deterministic, self.base_seed = deterministic, base_seed
        self._rng = None

    def __len__(self):
        return len(self.indices)

    def _seed(self, g: int) -> int:
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
    def __init__(self, pad_id: int = PAD, pad_multiple: int = 8):
        self.pad_id, self.mult = pad_id, pad_multiple

    def __call__(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(b) for b in batch)
        max_len = ((max_len + self.mult - 1) // self.mult) * self.mult
        out = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(batch):
            out[i, :len(s)] = s
        return out


def _download(url: str, dst: str):
    if osp.exists(dst):
        return
    os.makedirs(osp.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
            total = int(r.headers.get("Content-Length", 0)) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=osp.basename(dst)) as bar:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        if osp.exists(dst + ".part"):
            os.remove(dst + ".part")
        raise SystemExit(f"Download failed for {url} ({e}). Put the file manually at {dst}")
    os.replace(dst + ".part", dst)


def _atom_code(a) -> int:
    return a.GetAtomicNum() * 10000 + (a.GetFormalCharge() + 50) * 100 + a.GetNumExplicitHs()


_W: dict = {}


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


# SMILES -> (record, canonical smiles, bandwidth, was disconnected)
def _smiles_worker(smi: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    can = Chem.MolToSmiles(mol)
    frags = Chem.GetMolFrags(mol, asMols=True)
    multi = len(frags) > 1
    if multi:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    n = mol.GetNumAtoms()
    x = np.array([_atom_code(a) for a in mol.GetAtoms()], np.int64)
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
    indices = np.array([j for l in nbrs for j, _ in l], np.int64)
    eattr = np.array([t for l in nbrs for _, t in l], np.int64)
    order = np.empty(n, np.int64)
    bw = 0
    for s in range(1, BASE_ORDERS + 1):
        _cm_order(rowptr, indices, s, order)
        bw = max(bw, int(_bandwidth(rowptr, indices, order)))
    return (x, rowptr, indices, eattr), can, bw, multi


def _process(smiles: Sequence[str], n_jobs: int, desc: str):
    recs, canon, bw, multi = [], [], 0, 0
    with mp.Pool(n_jobs, initializer=_worker_init) as pool:
        for r in tqdm(pool.imap(_smiles_worker, smiles, chunksize=1000), total=len(smiles), desc=desc):
            if r is None:
                continue
            recs.append(r[0])
            canon.append(r[1])
            bw = max(bw, r[2])
            multi += int(r[3])
    return recs, canon, bw, multi


# GDSS split for QM9 / ZINC250k (test indices from the json), validation held out from the rest
def _load_gdss(name: str, root: str):
    import pandas as pd
    raw = osp.join(root, name, "raw")
    csv, idx = GDSS_FILES[name]
    for f in (csv, idx):
        _download(GDSS_URL + f, osp.join(raw, f))
    df = pd.read_csv(osp.join(raw, csv))
    col = next(c for c in ("SMILES1", "smiles", "SMILES") if c in df.columns)
    smiles = df[col].astype(str).tolist()
    js = json.load(open(osp.join(raw, idx)))
    if isinstance(js, dict):
        js = js.get("valid_idxs", next(iter(js.values())))
    test_idx = np.array(sorted(int(i) for i in js))
    mask = np.ones(len(smiles), bool)
    mask[test_idx] = False
    rest = np.nonzero(mask)[0]
    perm = np.random.RandomState(42).permutation(len(rest))
    n_val = int(round(VAL_FRAC[name] * len(rest)))
    val_idx, train_idx = rest[perm[:n_val]], np.sort(rest[perm[n_val:]])
    pick = lambda ids: [smiles[i] for i in ids]
    return {"train": pick(train_idx), "val": pick(val_idx), "test": pick(test_idx)}


def _load_moses(root: str):
    import pandas as pd
    raw = osp.join(root, "moses", "raw")
    for f in ("train.csv", "test.csv"):
        _download(MOSES_URL + f, osp.join(raw, f))
    train = pd.read_csv(osp.join(raw, "train.csv"))["SMILES"].tolist()
    test = pd.read_csv(osp.join(raw, "test.csv"))["SMILES"].tolist()
    perm = np.random.RandomState(42).permutation(len(train))
    n_val = int(round(VAL_FRAC["moses"] * len(train)))
    return {"train": [train[i] for i in np.sort(perm[n_val:])], "val": [train[i] for i in np.sort(perm[:n_val])],
            "test": test}


def _atom_name(z: int, charge: int, nh: int) -> str:
    from rdkit import Chem
    sym = Chem.GetPeriodicTable().GetElementSymbol(int(z))
    h = "" if nh == 0 else ("H" if nh == 1 else f"H{nh}")
    c = "" if charge == 0 else ("+" if charge == 1 else "-" if charge == -1 else f"{charge:+d}")
    return f"[{sym}{h}{c}]"


# download + process if needed, returns dataset meta
def prepare_dataset(name: str, root: str = "data", n_jobs: int = 8) -> dict:
    name = {"zinc250k": "zinc"}.get(name.lower(), name.lower())
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}', choose from {DATASET_NAMES}")
    pdir = osp.join(root, name, "processed")
    meta_path = osp.join(pdir, "meta.json")
    if osp.exists(meta_path):
        return json.load(open(meta_path))

    raw = _load_moses(root) if name == "moses" else _load_gdss(name, root)
    out = {s: _process(raw[s], n_jobs, f"{name} {s}") for s in ("train", "val", "test")}

    codes = np.unique(np.concatenate([r[0] for s in out for r in out[s][0]]))
    atoms = [[int(c // 10000), int((c % 10000) // 100 - 50), int(c % 100)] for c in codes]
    names = [_atom_name(*a) for a in atoms]
    bw = max(out[s][2] for s in out)
    max_nodes, max_len, sizes = 0, 0, {}
    for s in ("train", "val", "test"):
        recs, canon, _, multi = out[s]
        recs = [(np.searchsorted(codes, r[0]), r[1], r[2], r[3]) for r in recs]
        store = GraphStore.build(recs)
        store.save(osp.join(pdir, s))
        with open(osp.join(pdir, f"{s}_smiles.txt"), "w") as f:
            f.write("\n".join(canon))
        n, m = store.num_nodes(), store.num_edges()
        max_nodes = max(max_nodes, int(n.max()))
        max_len = max(max_len, int((2 + 3 * n + 2 * m).max()))
        sizes[s] = len(recs)
        print(f"[data] {name} {s}: {len(recs)} graphs ({multi} disconnected, largest fragment kept)")
    meta = dict(name=name, atoms=atoms, atom_names=names, bw=bw, max_nodes=max_nodes, max_len=max_len,
                num_bond_types=NUM_BOND_TYPES, base_orders=BASE_ORDERS, sizes=sizes)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[data] {name} ready: {len(atoms)} atom tokens, bandwidth {bw}, max nodes {max_nodes}, max_len {max_len}")
    return meta


def load_split(name: str, root: str, split: str) -> GraphStore:
    return GraphStore.load(osp.join(root, name, "processed", split))


def load_smiles(name: str, root: str, split: str = "train") -> List[str]:
    with open(osp.join(root, name, "processed", f"{split}_smiles.txt")) as f:
        return f.read().split("\n")


def build_tokenizer(meta: dict) -> GeelTokenizer:
    return GeelTokenizer(meta["bw"], meta["atoms"], meta["max_nodes"], meta["max_len"])


# returns (meta, tokenizer, train_ds, val_ds); exclude_idx removes train graphs (retain-only retraining)
def build_datasets(name: str, root: str = "data", n_jobs: int = 8, exclude_idx: Optional[np.ndarray] = None,
                   val_max: int = 10000, seed: int = 0):
    meta = prepare_dataset(name, root, n_jobs)
    name = meta["name"]
    tok = build_tokenizer(meta)
    train_store, val_store = load_split(name, root, "train"), load_split(name, root, "val")
    idx = np.arange(len(train_store))
    if exclude_idx is not None:
        idx = np.setdiff1d(idx, np.asarray(exclude_idx))
    vidx = np.arange(len(val_store))
    if len(vidx) > val_max:
        vidx = np.sort(np.random.RandomState(0).permutation(len(vidx))[:val_max])
    train_ds = MolGEELDataset(train_store, tok, idx, deterministic=False, base_seed=seed)
    val_ds = MolGEELDataset(val_store, tok, vidx, deterministic=True)
    return meta, tok, train_ds, val_ds


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False,
                seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
        collate_fn=Collator(PAD), pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, generator=g,
    )
