import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from ..data import SentTokenizer, PAD, SOS, EOS

# (hidden, layers, heads); "s" (113M) is the model used in the paper
SIZES = {"xs": (384, 6, 12), "s": (768, 12, 12), "m": (1024, 24, 16)}


class AutoGraphLM(nn.Module):
    def __init__(self, vocab_size: int, size: str = "s", attention_dropout: float = 0.0,
                 attn_implementation: str = "sdpa"):
        super().__init__()
        hidden, layers, heads = SIZES[size]
        self.cfg = dict(vocab_size=vocab_size, size=size, attention_dropout=attention_dropout,
                        attn_implementation=attn_implementation)
        config = LlamaConfig(
            vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
            intermediate_size=4 * hidden, bos_token_id=SOS, eos_token_id=EOS, pad_token_id=PAD,
            attention_dropout=attention_dropout, attn_implementation=attn_implementation,
        )
        self.lm = LlamaForCausalLM(config)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits

    # next-token cross entropy, padding ignored
    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1),
                               ignore_index=PAD)

    # -log p(sequence) per example, useful for unlearning metrics
    @torch.no_grad()
    def per_sequence_nll(self, batch: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        logits = self(batch[:, :-1]).float()
        tgt = batch[:, 1:]
        nll = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=PAD, reduction="none")
        s = nll.sum(1)
        return s if reduce == "sum" else s / (tgt != PAD).sum(1).clamp(min=1)


# self-contained checkpoint: weights + model config + tokenizer meta
def save_checkpoint(path: str, model: AutoGraphLM, tok: SentTokenizer, dataset: str, step: int = 0,
                    optimizer=None, scheduler=None, **extra):
    state = {"model": model.state_dict(), "model_cfg": model.cfg, "tokenizer": tok.meta(),
             "dataset": dataset, "step": step, **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device="cpu", attention_dropout: Optional[float] = None
                    ) -> Tuple[AutoGraphLM, SentTokenizer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_cfg"])
    if attention_dropout is not None:
        cfg["attention_dropout"] = attention_dropout
    model = AutoGraphLM(**cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    return model.to(device), SentTokenizer.from_meta(ck["tokenizer"]), ck