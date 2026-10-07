import math
import random

import numpy as np
import torch

from ..data import SentTokenizer, SOS, RESET, LADJ, RADJ, EOS, PAD


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


# linear warmup, cosine decay until max_steps, then constant at min_factor
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_factor=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        progress = min((step - warmup_steps) / max(1, max_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - min_factor) + min_factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# token grammar as a (previous token class, in neighborhood) -> allowed tokens table
class GrammarConstraint:
    N_CLS = 9
    C_SOS, C_RESET, C_LADJ, C_RADJ, C_EOS, C_PAD, C_IDX, C_NODE, C_EDGE = range(9)

    def __init__(self, tok: SentTokenizer, device):
        V = len(tok)
        ar = torch.arange(V)
        cls = torch.empty(V, dtype=torch.long)
        cls[SOS], cls[RESET], cls[LADJ], cls[RADJ], cls[EOS], cls[PAD] = range(6)
        cls[(ar >= tok.idx_offset) & (ar < tok.node_idx_offset)] = self.C_IDX
        cls[(ar >= tok.node_idx_offset) & (ar < tok.edge_idx_offset)] = self.C_NODE
        cls[ar >= tok.edge_idx_offset] = self.C_EDGE
        is_ = lambda c: cls == c
        m_idx, m_node, m_edge = is_(self.C_IDX), is_(self.C_NODE), is_(self.C_EDGE)
        m_reset, m_ladj, m_radj, m_eos = is_(self.C_RESET), is_(self.C_LADJ), is_(self.C_RADJ), is_(self.C_EOS)

        table = torch.zeros(self.N_CLS, 2, V, dtype=torch.bool)
        for b in (0, 1):
            table[self.C_SOS, b] = m_idx
            table[self.C_NODE, b] = m_edge | m_reset | m_ladj | m_eos
            table[self.C_EDGE, b] = m_idx
            table[self.C_LADJ, b] = m_edge
            table[self.C_RADJ, b] = m_edge | m_reset | m_eos
            table[self.C_RESET, b] = m_idx
            table[self.C_EOS, b] = True
            table[self.C_PAD, b] = True
        table[self.C_IDX, 0] = m_node
        table[self.C_IDX, 1] = m_edge | m_radj
        self.cls = cls.to(device)
        self.table = table.view(self.N_CLS * 2, V).to(device)

    def mask_logits(self, logits: torch.Tensor, prev: torch.Tensor, in_bracket: torch.Tensor) -> torch.Tensor:
        state = self.cls[prev] * 2 + in_bracket.long()
        return logits.masked_fill(~self.table[state], float("-inf"))

    @staticmethod
    def update_bracket(in_bracket: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        return (in_bracket | (sampled == LADJ)) & (sampled != RADJ)


# load an official Lightning .ckpt (model.model.* -> lm.*)
def load_official_state_dict(model, path: str, strict: bool = False):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.model."
    new = {"lm." + k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    res = model.load_state_dict(new, strict=strict)
    print(f"[official ckpt] missing: {res.missing_keys} | unexpected: {res.unexpected_keys}")
    return res