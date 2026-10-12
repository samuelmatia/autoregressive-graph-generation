import random

import numpy as np
import torch

from ..data import PAD, BOS, EOS, EDGE_OFFSET, GeelTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# tf32 for faster matmuls
def setup_gpu():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# attributed GEEL grammar as a (previous token class) -> allowed tokens table
# bos -> node | node -> edge (ni=1) | edge -> bond | bond -> node, edge (ni=0) or eos
class GrammarConstraint:
    N_CLS = 7
    C_PAD, C_BOS, C_EOS, C_NODE, C_EDGE1, C_EDGE0, C_BOND = range(7)

    def __init__(self, tok: GeelTokenizer, device):
        V = len(tok)
        ar = torch.arange(V)
        is_edge = (ar >= EDGE_OFFSET) & (ar < tok.node_offset)
        odd = ((ar - 2) % 2) == 1
        cls = torch.empty(V, dtype=torch.long)
        cls[PAD], cls[BOS], cls[EOS] = self.C_PAD, self.C_BOS, self.C_EOS
        cls[is_edge & odd] = self.C_EDGE1
        cls[is_edge & ~odd] = self.C_EDGE0
        cls[(ar >= tok.node_offset) & (ar < tok.bond_offset)] = self.C_NODE
        cls[ar >= tok.bond_offset] = self.C_BOND
        m = lambda c: cls == c

        table = torch.zeros(self.N_CLS, V, dtype=torch.bool)
        table[self.C_BOS] = m(self.C_NODE)
        table[self.C_NODE] = m(self.C_EDGE1)
        table[self.C_EDGE1] = m(self.C_BOND)
        table[self.C_EDGE0] = m(self.C_BOND)
        table[self.C_BOND] = m(self.C_NODE) | m(self.C_EDGE0) | m(self.C_EOS)
        table[self.C_EOS] = True
        table[self.C_PAD] = True
        self.cls = cls.to(device)
        self.table = table.to(device)

    def mask_logits(self, logits: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~self.table[self.cls[prev]], float("-inf"))
