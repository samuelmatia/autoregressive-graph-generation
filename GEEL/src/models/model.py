import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..data import PAD, EDGE_OFFSET, GeelTokenizer


# token embedding + node positional embedding, 3-layer LSTM, linear head
class GeelLSTM(nn.Module):
    def __init__(self, tok: GeelTokenizer, emb_size: int = 512, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.cfg = dict(emb_size=emb_size, num_layers=num_layers, dropout=dropout)
        V = len(tok)
        self.max_nodes = tok.max_nodes
        self.scale = math.sqrt(emb_size)
        ids = torch.arange(V)
        is_edge = (ids >= EDGE_OFFSET) & (ids < tok.node_offset)
        is_ni = is_edge & (((ids - 2) % 2) == 1)
        self.register_buffer("is_edge", is_edge.long(), persistent=False)
        self.register_buffer("is_ni", is_ni.long(), persistent=False)
        self.embedding = nn.Embedding(V, emb_size)
        self.node_pe = nn.Embedding(tok.max_nodes + 3, emb_size)
        self.lstm = nn.LSTM(emb_size, emb_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(emb_size, V)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # node_count: number of new-source tokens already consumed (for step-by-step decoding)
    def forward(self, tokens: torch.Tensor, state=None, node_count: Optional[torch.Tensor] = None):
        if node_count is None:
            node_count = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        count = node_count[:, None] + self.is_ni[tokens].cumsum(1)
        pe = count * self.is_edge[tokens]
        pe = torch.where(pe > self.max_nodes + 1, torch.zeros_like(pe), pe)
        x = (self.embedding(tokens) + self.node_pe(pe)) * self.scale
        h, state = self.lstm(x, state)
        return self.out(h), state, count[:, -1]

    # next-token cross entropy, padding ignored
    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self(batch[:, :-1])[0]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1),
                               ignore_index=PAD)

    # -log p(sequence) per example, useful for unlearning metrics
    @torch.no_grad()
    def per_sequence_nll(self, batch: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        logits = self(batch[:, :-1])[0].float()
        tgt = batch[:, 1:]
        nll = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=PAD, reduction="none")
        s = nll.sum(1)
        return s if reduce == "sum" else s / (tgt != PAD).sum(1).clamp(min=1)


# self-contained checkpoint: weights + model config + tokenizer meta
def save_checkpoint(path: str, model: GeelLSTM, tok: GeelTokenizer, dataset: str, step: int = 0,
                    optimizer=None, **extra):
    state = {"model": model.state_dict(), "model_cfg": model.cfg, "tokenizer": tok.meta(),
             "dataset": dataset, "step": step, **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device="cpu") -> Tuple[GeelLSTM, GeelTokenizer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tok = GeelTokenizer.from_meta(ck["tokenizer"])
    model = GeelLSTM(tok, **ck["model_cfg"])
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return model.to(device), tok, ck
