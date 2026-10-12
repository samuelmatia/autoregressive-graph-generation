# python train.py --dataset qm9
# python train.py --dataset zinc
# python train.py --dataset moses
# python train.py --dataset qm9 --lr 1e-3 --out logs/qm9_lr1e-3   # paper appendix lr, the repo script uses 1e-4
# python train.py --dataset qm9 --exclude_idx forget_idx.npy --out logs/qm9_retain   # retain-only retraining
# python train.py --dataset qm9 --resume logs/qm9/last.pt
# python train.py --dataset qm9 --save_every_check   # keep one checkpoint per sampling check (select by FCD afterwards)

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data import PAD, build_datasets, make_loader, load_smiles
from src.evaluate import evaluate_graphs, summarize, load_eval_data
from src.generate import generate_graphs
from src.models.model import GeelLSTM, load_checkpoint, save_checkpoint
from src.models.utils import set_seed, setup_gpu

# defaults follow the official scripts (script/qm9.sh, script/zinc.sh, script/moses.sh)
DEFAULTS = {
    "qm9":   dict(lr=1e-4, epochs=2000, check_every=200),
    "zinc":  dict(lr=5e-4, epochs=2000, check_every=200),
    "moses": dict(lr=5e-4, epochs=1000, check_every=50),
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emb_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_max", type=int, default=10000)
    ap.add_argument("--check_every", type=int, default=None)
    ap.add_argument("--check_samples", type=int, default=1000)
    ap.add_argument("--save_every_check", action="store_true")
    ap.add_argument("--final_samples", type=int, default=10000)
    ap.add_argument("--exclude_idx", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_final_eval", action="store_true")
    a = ap.parse_args()
    d = DEFAULTS[a.dataset]
    a.lr = a.lr or d["lr"]
    a.epochs = a.epochs or d["epochs"]
    a.check_every = a.check_every or d["check_every"]
    a.out = a.out or os.path.join("logs", a.dataset)
    return a


@torch.no_grad()
def validate(model, loader, device, amp) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch.to(device, non_blocking=True)
        with amp:
            logits = model(batch[:, :-1])[0]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum")
        tot += nll.item()
        cnt += (batch[:, 1:] != PAD).sum().item()
    model.train()
    return tot / max(cnt, 1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
    set_seed(a.seed)
    setup_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda")

    # data
    excl = np.load(a.exclude_idx) if a.exclude_idx else None
    meta, tok, train_ds, val_ds = build_datasets(a.dataset, a.root, n_jobs=a.num_workers or 4,
                                                 exclude_idx=excl, val_max=a.val_max, seed=a.seed)
    name = meta["name"]
    train_loader = make_loader(train_ds, a.batch_size, True, a.num_workers, drop_last=False, seed=a.seed)
    val_loader = make_loader(val_ds, a.batch_size, False, min(a.num_workers, 4))
    train_smiles = load_smiles(name, a.root, "train")
    print(f"[train] {name}: {len(train_ds)} train | {len(val_ds)} val | vocab={len(tok)} | bw={tok.bw} | "
          f"atom tokens={tok.num_atoms} | max_nodes={tok.max_nodes} | max_len={tok.max_len}")

    # model and optimizer
    model = GeelLSTM(tok, a.emb_size, a.num_layers, a.dropout).to(device)
    print(f"[train] {model.num_parameters() / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
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
    for epoch in tqdm(range(epoch0 + 1, a.epochs + 1), desc="epochs", initial=epoch0, total=a.epochs):
        t0 = time.time()
        run_loss = torch.zeros((), device=device)
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=True)
            with amp:
                logits = model(batch[:, :-1])[0]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            run_loss += loss.detach()
            n_batches += 1

        vl = validate(model, val_loader, device, amp)
        rec = dict(epoch=epoch, train_loss=(run_loss / n_batches).item(), val_loss=vl, sec=time.time() - t0)
        if epoch % a.check_every == 0 or epoch == a.epochs:
            graphs, _ = generate_graphs(model, tok, a.check_samples, 1000, verbose=False)
            res = evaluate_graphs(graphs, meta["atoms"], train_smiles, progress=False)
            rec.update({k: float(res[k]) for k in ("validity_wo_cor", "validity_corrected", "uniqueness", "novelty")})
            model.train()
            if a.save_every_check:
                save_checkpoint(os.path.join(a.out, f"epoch{epoch}.pt"), model, tok, name, epoch, best_val=best_val)
        log(rec)
        if vl < best_val:
            best_val = vl
            save_checkpoint(os.path.join(a.out, "best.pt"), model, tok, name, epoch, best_val=best_val)
        save_checkpoint(os.path.join(a.out, "last.pt"), model, tok, name, epoch, opt, best_val=best_val)

    # final evaluation with the best checkpoint (lowest val loss)
    if not a.no_final_eval:
        best, _, _ = load_checkpoint(os.path.join(a.out, "best.pt"), device)
        graphs, dt = generate_graphs(best, tok, a.final_samples, 1000)
        _, _, test_smiles = load_eval_data(name, a.root)
        res = evaluate_graphs(graphs, meta["atoms"], train_smiles, test_smiles)
        print("[final] " + summarize(res))
        res.pop("smiles")
        json.dump(res, open(os.path.join(a.out, "final_metrics.json"), "w"), indent=2, default=float)
        print(f"[final] FCD / Scaf / SNN / Frag / NSPDK: python -m src.generate --ckpt {a.out}/best.pt --official --nspdk")


if __name__ == "__main__":
    main()
