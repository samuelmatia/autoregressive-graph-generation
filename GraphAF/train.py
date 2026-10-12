# python train.py --dataset zinc250k                                   # density modeling, 10 epochs
# python train.py --dataset zinc250k --epochs 300 --out logs/zinc250k/pretrain300   # pretraining before RL fine-tuning
# python train.py --dataset qm9 --st_type exp --num_flow_layer 12 --is_bn --learn_prior
# python train.py --dataset zinc250k --exclude_idx forget_idx.npy --out logs/zinc250k/retain   # retain-only retraining
# python train.py --dataset zinc250k --resume logs/zinc250k/base/last.pt
# python train.py --dataset zinc250k --batch_size 32 --grad_accum 4    # if the batch does not fit in GPU memory

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data import build_datasets, load_smiles, make_loader
from src.evaluate import evaluate_samples, summarize
from src.generate import generate_molecules, reconstruction_rate
from src.models.model import GraphAF, load_checkpoint, save_checkpoint
from src.models.utils import featurize, set_seed, setup_gpu


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["zinc250k", "qm9", "moses"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_flow_layer", type=int, default=6)
    ap.add_argument("--gcn_layer", type=int, default=3)
    ap.add_argument("--nhid", type=int, default=128)
    ap.add_argument("--nout", type=int, default=128)
    ap.add_argument("--st_type", default="sigmoid", choices=["sigmoid", "exp", "softplus"])
    ap.add_argument("--sigmoid_shift", type=float, default=2.0)
    ap.add_argument("--deq_coeff", type=float, default=0.9)
    ap.add_argument("--is_bn", action="store_true")
    ap.add_argument("--learn_prior", action="store_true")
    ap.add_argument("--no_divide_loss", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--sample_n", type=int, default=100)
    ap.add_argument("--val_max", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--min_atoms", type=int, default=5)
    ap.add_argument("--max_atoms", type=int, default=48)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--exclude_idx", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=2019)
    ap.add_argument("--no_final_eval", action="store_true")
    a = ap.parse_args()
    a.out = a.out or os.path.join("logs", a.dataset, "base")
    return a


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for types, bonds in tqdm(loader, desc="val", leave=False):
        x, adj = featurize(types.to(device), bonds.to(device), model.node_dim)
        tot += model.per_sample_nll(x, adj, per_dim=True).sum().item()
        cnt += x.size(0)
    model.train()
    return tot / max(cnt, 1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
    set_seed(a.seed)
    setup_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    excl = np.load(a.exclude_idx) if a.exclude_idx else None
    meta, train_ds, val_ds = build_datasets(a.dataset, a.root, n_jobs=a.num_workers or 4, exclude_idx=excl,
                                            val_max=a.val_max, seed=a.seed)
    micro_bs = a.batch_size // a.grad_accum
    train_loader = make_loader(train_ds, a.batch_size, True, a.num_workers, drop_last=True, seed=a.seed)
    val_loader = make_loader(val_ds, 64, False, min(a.num_workers, 4))
    train_smiles = load_smiles(a.dataset, a.root, "train")
    print(f"[train] {a.dataset}: {len(train_ds)} train | {len(val_ds)} val | max_size={meta['max_size']} "
          f"| unroll={meta['edge_unroll']}")

    # model and optimizer
    model = GraphAF(meta["max_size"], meta["node_dim"], meta["bond_dim"], meta["edge_unroll"], a.num_flow_layer,
                    a.gcn_layer, a.nhid, a.nout, a.st_type, a.sigmoid_shift, a.deq_coeff, a.is_bn,
                    a.learn_prior, meta["atom_decoder"]).to(device)
    print(f"[train] {model.num_parameters() / 1e6:.2f}M parameters | {model.node_mask.size(0)} steps per graph")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                           weight_decay=a.weight_decay)
    total_len = model.len_node + model.len_edge

    epoch0, best_val = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
        opt.load_state_dict(ck["optimizer"])
        epoch0, best_val = ck["step"], ck.get("best_val", best_val)
        print(f"[train] resumed at epoch {epoch0}")

    log_f = open(os.path.join(a.out, "log.jsonl"), "a")

    def log(d):
        tqdm.write(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
        log_f.write(json.dumps(d) + "\n")
        log_f.flush()

    # training loop
    model.train()
    step = epoch0 * len(train_loader)
    for epoch in range(epoch0, a.epochs):
        bar = tqdm(train_loader, desc=f"epoch {epoch}")
        run_loss, run_n, t0 = 0.0, 0, time.time()
        for types, bonds in bar:
            x, adj = featurize(types.to(device, non_blocking=True), bonds.to(device, non_blocking=True),
                               model.node_dim)
            B = x.size(0)
            for xc, ac in zip(x.split(micro_bs), adj.split(micro_bs)):
                nll = model.nll(xc, ac).sum() / B
                loss = nll if a.no_divide_loss else nll / total_len
                loss.backward()
                run_loss += nll.item() / total_len
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_n += 1
            if step % a.log_every == 0:
                bar.set_postfix(nll_dim=f"{run_loss / run_n:.4f}")
                log(dict(step=step, epoch=epoch, nll_per_dim=run_loss / run_n,
                         it_per_s=run_n / (time.time() - t0)))
                run_loss, run_n, t0 = 0.0, 0, time.time()

        vl = validate(model, val_loader, device)
        rec = dict(epoch=epoch, val_nll_per_dim=vl)
        if a.sample_every and ((epoch + 1) % a.sample_every == 0 or epoch + 1 == a.epochs):
            samples, _ = generate_molecules(model, a.sample_n, 128, a.temperature, a.max_atoms, a.min_atoms,
                                            verbose=False)
            rec.update({k: float(v) for k, v in evaluate_samples(samples, train_smiles).items()
                        if k in ("validity", "validity_wo_check", "uniqueness", "novelty")})
            model.train()
        log(rec)
        if vl < best_val:
            best_val = vl
            save_checkpoint(os.path.join(a.out, "best.pt"), model, meta, epoch + 1, best_val=best_val)
        save_checkpoint(os.path.join(a.out, "last.pt"), model, meta, epoch + 1, opt, best_val=best_val)

    # final evaluation with the best checkpoint (lowest val loss)
    if not a.no_final_eval:
        best, _, _ = load_checkpoint(os.path.join(a.out, "best.pt"), device)
        samples, dt = generate_molecules(best, a.num_samples, 512, a.temperature, a.max_atoms, a.min_atoms)
        res = evaluate_samples(samples, train_smiles)
        res["reconstruction"] = reconstruction_rate(best, val_loader)
        print("[final] " + summarize(res))
        json.dump(res, open(os.path.join(a.out, "final_metrics.json"), "w"), indent=2, default=float)
        print(f"[final] more samples / property scores: python -m src.generate --ckpt {a.out}/best.pt")


if __name__ == "__main__":
    main()
