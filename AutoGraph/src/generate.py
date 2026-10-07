# python -m src.generate --ckpt logs/qm9/s/best.pt --top_k 5
# python -m src.generate --ckpt logs/moses/s/best.pt --official
# python -m src.generate --ckpt logs/guacamol/s/best.pt --num_samples 60000 --official
# python -m src.generate --ckpt logs/qm9/s/best.pt --no_constraint   # without grammar mask

import argparse
import json
import os
import pickle
import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .data import SOS, EOS, PAD, MolGraph, SentTokenizer, load_smiles, prepare_dataset
from .models.model import AutoGraphLM, load_checkpoint
from .models.utils import GrammarConstraint, set_seed, setup_gpu


# sample num sequences in parallel with a KV cache; grammar mask is applied before top-k
@torch.inference_mode()
def sample_sequences(model: AutoGraphLM, tok: SentTokenizer, num: int, top_k: int = 5, temperature: float = 1.0,
                     max_length: int = 500, constrained: bool = True, device=None,
                     progress: bool = False) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    constraint = GrammarConstraint(tok, device) if constrained else None

    cur = torch.full((num, 1), SOS, dtype=torch.long, device=device)
    prev = cur[:, 0]
    in_bracket = torch.zeros(num, dtype=torch.bool, device=device)
    finished = torch.zeros(num, dtype=torch.bool, device=device)
    out: List[torch.Tensor] = [cur]
    cache = None

    for _ in tqdm(range(max_length - 1), desc="decoding", leave=False, disable=not progress):
        with amp:
            o = model.lm(input_ids=cur, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1, :].float()
        logits[:, SOS] = float("-inf")
        if constraint is not None:
            logits = constraint.mask_logits(logits, prev, in_bracket)
        logits = logits / max(temperature, 1e-8)
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
        in_bracket = GrammarConstraint.update_bracket(in_bracket, nxt)
        finished = finished | (nxt == EOS)
        out.append(nxt[:, None])
        cur, prev = nxt[:, None], nxt
        if bool(finished.all()):
            break
    return torch.cat(out, dim=1)


def sequences_to_graphs(seqs: torch.Tensor, tok: SentTokenizer) -> List[Optional[MolGraph]]:
    graphs = []
    for row in seqs.cpu().numpy():
        eos_pos = (row == EOS).nonzero()[0]
        row = row[:eos_pos[0]] if len(eos_pos) else row
        graphs.append(tok.decode(row))
    return graphs


# returns (graphs, seconds per graph)
def generate_graphs(model: AutoGraphLM, tok: SentTokenizer, num_samples: int, batch_size: int = 512,
                    top_k: int = 5, temperature: float = 1.0, max_length: int = 500,
                    constrained: bool = True, verbose: bool = True) -> Tuple[List[Optional[MolGraph]], float]:
    device = next(model.parameters()).device
    graphs: List[Optional[MolGraph]] = []
    t0 = time.time()
    with tqdm(total=num_samples, desc="generate", disable=not verbose) as bar:
        for i in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - i)
            seqs = sample_sequences(model, tok, b, top_k, temperature, max_length, constrained, device)
            graphs.extend(sequences_to_graphs(seqs, tok))
            bar.update(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(num_samples, 1)
    if verbose:
        print(f"[generate] {dt * 1000:.2f} ms / graph")
    return graphs, dt


DEFAULT_GEN = {"qm9": dict(num_samples=10000, max_length=500),
               "moses": dict(num_samples=30000, max_length=512),
               "guacamol": dict(num_samples=60000, max_length=1024)}


def main():
    from .evaluate import evaluate_graphs, summarize, official_moses, official_guacamol

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--no_constraint", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    setup_gpu()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_checkpoint(a.ckpt, device)
    ds = ck["dataset"]
    dflt = DEFAULT_GEN[ds]
    n = a.num_samples or dflt["num_samples"]
    max_len = a.max_length or dflt["max_length"]
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "gen")
    os.makedirs(out, exist_ok=True)

    graphs, dt = generate_graphs(model, tok, n, a.batch_size, a.top_k, a.temperature, max_len, not a.no_constraint)
    pickle.dump(graphs, open(os.path.join(out, "graphs.pkl"), "wb"))

    meta = prepare_dataset(ds, a.root)
    res = evaluate_graphs(graphs, meta["atom_decoder"], load_smiles(ds, a.root, "train"), check_stability=(ds == "qm9"))
    print(summarize(res))
    for name in ("smiles", "relaxed_smiles"):
        with open(os.path.join(out, f"generated_{name}.txt"), "w") as f:
            f.write("\n".join(s if s is not None else "None" for s in res[name]))
    metrics = {k: v for k, v in res.items() if k not in ("smiles", "relaxed_smiles")}
    metrics.update(sec_per_graph=dt, top_k=a.top_k, temperature=a.temperature, ckpt=a.ckpt, step=ck.get("step"))

    if a.official and ds == "moses":
        metrics["official"] = {k: float(v) for k, v in official_moses(res["smiles"], a.root).items()}
    elif a.official and ds == "guacamol":
        metrics["official"] = official_guacamol(res["smiles"], a.root, os.path.join(out, "guacamol_dist.json"))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"[generate] results in {out}")


if __name__ == "__main__":
    main()