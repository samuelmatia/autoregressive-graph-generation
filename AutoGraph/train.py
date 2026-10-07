# python train.py --dataset qm9
# python train.py --dataset moses --compile
# python train.py --dataset guacamol --compile
# python train.py --dataset qm9 --exclude_idx forget_idx.npy --out logs/qm9_retain   # retain-only retraining
# python train.py --dataset qm9 --resume logs/qm9/s/last.pt
# python train.py --dataset qm9 --batch_size 256 --grad_accum 2   # if the batch does not fit in GPU memory

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data import PAD, build_datasets, make_loader, load_smiles
from src.evaluate import evaluate_graphs, summarize
from src.generate import generate_graphs
from src.models.model import AutoGraphLM, load_checkpoint, save_checkpoint
from src.models.utils import get_cosine_schedule_with_warmup, set_seed, setup_gpu

# defaults follow the official configs
DEFAULTS = {
    "qm9":      dict(batch_size=512, max_steps=100_000, num_samples=10_000, max_gen_len=500),
    "moses":    dict(batch_size=512, max_steps=400_000, num_samples=30_000, max_gen_len=512),
    "guacamol": dict(batch_size=256, max_steps=800_000, num_samples=60_000, max_gen_len=1024),
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model_size", default="s", choices=["xs", "s", "m"])
    ap.add_argument("--attention_dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_max", type=int, default=10000)
    ap.add_argument("--sample_every", type=int, default=10000)
    ap.add_argument("--sample_n", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--exclude_idx", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_final_eval", action="store_true")
    a = ap.parse_args()
    d = DEFAULTS[a.dataset]
    a.batch_size = a.batch_size or d["batch_size"]
    a.max_steps = a.max_steps or d["max_steps"]
    a.num_samples, a.max_gen_len = d["num_samples"], d["max_gen_len"]
    a.out = a.out or os.path.join("logs", a.dataset, a.model_size)
    return a


@torch.no_grad()
def validate(model, loader, device, amp) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch.to(device, non_blocking=True)
        with amp:
            logits = model(batch[:, :-1])
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
    use_amp = device.type == "cuda"
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    # data
    excl = np.load(a.exclude_idx) if a.exclude_idx else None
    meta, tok, train_ds, val_ds = build_datasets(a.dataset, a.root, n_jobs=a.num_workers or 4,
                                                 exclude_idx=excl, val_max=a.val_max, seed=a.seed)
    micro_bs = a.batch_size // a.grad_accum
    train_loader = make_loader(train_ds, micro_bs, True, a.num_workers, drop_last=True, seed=a.seed)
    val_loader = make_loader(val_ds, 512, False, min(a.num_workers, 4))
    train_smiles = load_smiles(a.dataset, a.root, "train")
    print(f"[train] {a.dataset}: {len(train_ds)} train | {len(val_ds)} val | vocab={len(tok)} | max_nodes={tok.max_num_nodes}")

    # model and optimizer, no weight decay on biases and norms
    model = AutoGraphLM(len(tok), a.model_size, a.attention_dropout, a.attn).to(device)
    print(f"[train] {model.num_parameters() / 1e6:.1f}M parameters")
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), fused=use_amp)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup_frac * a.max_steps, 0.9 * a.max_steps, min_factor=0.1)

    step, best_val = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, best_val = ck["step"], ck.get("best_val", best_val)
        print(f"[train] resumed at step {step}")
    fwd = torch.compile(model, dynamic=True) if a.compile else model

    log_f = open(os.path.join(a.out, "log.jsonl"), "a")

    def log(d):
        tqdm.write(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
        log_f.write(json.dumps(d) + "\n")
        log_f.flush()

    # training loop
    model.train()
    run_loss, run_n = torch.zeros((), device=device), 0
    t0 = time.time()
    data_iter = iter(train_loader)
    bar = tqdm(total=a.max_steps, initial=step, desc="train")
    while step < a.max_steps:
        for _ in range(a.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            with amp:
                logits = fwd(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1), ignore_index=PAD)
            (loss / a.grad_accum).backward()
            run_loss += loss.detach()
            run_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        bar.update(1)

        if step % a.log_every == 0:
            avg = (run_loss / run_n).item()
            bar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            log(dict(step=step, loss=avg, lr=sched.get_last_lr()[0], it_per_s=a.log_every / (time.time() - t0)))
            t0, run_loss, run_n = time.time(), torch.zeros((), device=device), 0

        if step % a.val_every == 0 or step == a.max_steps:
            vl = validate(model, val_loader, device, amp)
            rec = dict(step=step, val_loss=vl)
            if a.sample_every and (step % a.sample_every == 0 or step == a.max_steps):
                graphs, _ = generate_graphs(model, tok, a.sample_n, 512, a.top_k, 1.0, a.max_gen_len, verbose=False)
                res = evaluate_graphs(graphs, meta["atom_decoder"], train_smiles,
                                      check_stability=(a.dataset == "qm9"), progress=False)
                rec.update({k: float(v) for k, v in res.items() if k in
                            ("validity", "relaxed_validity", "uniqueness", "novelty", "mol_stable")})
                model.train()
            log(rec)
            if vl < best_val:
                best_val = vl
                save_checkpoint(os.path.join(a.out, "best.pt"), model, tok, a.dataset, step, best_val=best_val)
            save_checkpoint(os.path.join(a.out, "last.pt"), model, tok, a.dataset, step, opt, sched, best_val=best_val)
            t0 = time.time()
    bar.close()

    # final evaluation with the best checkpoint (lowest val loss)
    if not a.no_final_eval:
        best, _, _ = load_checkpoint(os.path.join(a.out, "best.pt"), device)
        graphs, dt = generate_graphs(best, tok, a.num_samples, 512, a.top_k, 1.0, a.max_gen_len)
        res = evaluate_graphs(graphs, meta["atom_decoder"], train_smiles, check_stability=(a.dataset == "qm9"))
        print("[final] " + summarize(res))
        json.dump({k: v for k, v in res.items() if k not in ("smiles", "relaxed_smiles")},
                  open(os.path.join(a.out, "final_metrics.json"), "w"), indent=2, default=float)
        print(f"[final] official MOSES/GuacaMol metrics: python -m src.generate --ckpt {a.out}/best.pt --official")


if __name__ == "__main__":
    main()