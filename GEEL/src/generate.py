# python -m src.generate --ckpt logs/qm9/best.pt
# python -m src.generate --ckpt logs/zinc/best.pt --num_samples 10000 --batch_size 2000
# python -m src.generate --ckpt logs/zinc/best.pt --official --nspdk
# python -m src.generate --ckpt logs/qm9/best.pt --no_constraint   # without grammar mask

import argparse
import json
import os
import pickle
import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .data import BOS, EOS, PAD, MolGraph, GeelTokenizer
from .models.model import GeelLSTM, load_checkpoint
from .models.utils import GrammarConstraint, set_seed, setup_gpu


# sample sequences in parallel, the LSTM state is carried between steps (O(L) instead of O(L^2))
@torch.inference_mode()
def sample_sequences(model: GeelLSTM, tok: GeelTokenizer, num: int, temperature: float = 1.0, top_k: int = 0,
                     max_len: Optional[int] = None, constrained: bool = True, device=None, amp: bool = False,
                     progress: bool = False) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    max_len = max_len or tok.max_len
    constraint = GrammarConstraint(tok, device) if constrained else None
    cur = torch.full((num, 1), BOS, dtype=torch.long, device=device)
    prev = cur[:, 0]
    state, node_count = None, None
    finished = torch.zeros(num, dtype=torch.bool, device=device)
    out: List[torch.Tensor] = [cur]
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda")
    with ctx:
        for _ in tqdm(range(max_len - 1), desc="decoding", leave=False, disable=not progress):
            logits, state, node_count = model(cur, state, node_count)
            logits = logits[:, -1].float()
            logits[:, PAD] = float("-inf")
            logits[:, BOS] = float("-inf")
            if constraint is not None:
                logits = constraint.mask_logits(logits, prev)
            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            finished = finished | (nxt == EOS)
            out.append(nxt[:, None])
            cur, prev = nxt[:, None], nxt
            if bool(finished.all()):
                break
    return torch.cat(out, dim=1)


def sequences_to_graphs(seqs: torch.Tensor, tok: GeelTokenizer) -> List[Optional[MolGraph]]:
    return [tok.decode(row) for row in seqs.cpu().numpy()]


# returns (graphs, seconds per graph), None for sequences that do not decode
def generate_graphs(model: GeelLSTM, tok: GeelTokenizer, num_samples: int, batch_size: int = 1000,
                    temperature: float = 1.0, top_k: int = 0, constrained: bool = True, amp: bool = False,
                    verbose: bool = True) -> Tuple[List[Optional[MolGraph]], float]:
    device = next(model.parameters()).device
    graphs: List[Optional[MolGraph]] = []
    t0 = time.time()
    with tqdm(total=num_samples, desc="generate", disable=not verbose) as bar:
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            seqs = sample_sequences(model, tok, b, temperature, top_k, None, constrained, device, amp)
            graphs.extend(sequences_to_graphs(seqs, tok))
            bar.update(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(num_samples, 1)
    if verbose:
        print(f"[generate] {dt * 1000:.2f} ms / graph")
    return graphs, dt


def main():
    from .evaluate import evaluate_graphs, load_eval_data, summarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--no_constraint", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--nspdk", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    setup_gpu()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_checkpoint(a.ckpt, device)
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "gen")
    os.makedirs(out, exist_ok=True)

    graphs, dt = generate_graphs(model, tok, a.num_samples, a.batch_size, a.temperature, a.top_k,
                                 not a.no_constraint, a.amp)
    pickle.dump(graphs, open(os.path.join(out, "graphs.pkl"), "wb"))

    meta, train_smiles, test_smiles = load_eval_data(ck["dataset"], a.root)
    res = evaluate_graphs(graphs, meta["atoms"], train_smiles, test_smiles, a.official, a.nspdk,
                          device=str(device), n_jobs=a.n_jobs)
    print(summarize(res))
    with open(os.path.join(out, "generated_smiles.txt"), "w") as f:
        f.write("\n".join(res.pop("smiles")))
    res.update(sec_per_graph=dt, temperature=a.temperature, top_k=a.top_k, ckpt=a.ckpt, step=ck.get("step"))
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"[generate] results in {out}")


if __name__ == "__main__":
    main()
