from __future__ import annotations
import json
import multiprocessing as mp
import os.path as osp
import shutil
import urllib.request
import zipfile
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from numba import njit
    HAS_NUMBA=True

except ImportError:
    HAS_NUMBA=False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and kwargs:
            return args[0]
        return lambda f: f



#Constants
SOS, RESET, LADJ, RADJ, EOS, PAD = 0, 1, 2, 3, 4, 5
IDX_OFFSET = 6

ATOM_DECODERS = {
    "qm9": ["H", "C", "N", "O", "F"],
    "moses": ["C", "N", "S", "O", "F", "Cl", "Br", "H"],
    "guacamol": ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"],
}


@dataclass
class MolGraph:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray

    @property
    def num_nodes(self):
        return int(self.x.shape[0])


#Constants
