# python train.py --dataset zinc250k --epochs 300 --out logs/zinc250k/pretrain300      # pretrain first
# python finetune.py --ckpt logs/zinc250k/pretrain300/best.pt --property plogp --out logs/rl/plogp
# python finetune.py --ckpt logs/zinc250k/pretrain300/best.pt --property qed --out logs/rl/qed
# python finetune.py --ckpt logs/zinc250k/pretrain300/best.pt --task constrained --out logs/rl/constrained
# python finetune.py --ckpt ... --task constrained --repeat 50 --n_eval_mols 100      # quick constrained check
# python finetune.py --ckpt ... --property plogp --batch_size 32 --chunk 1024         # if memory is tight

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data import build_datasets
from src.evaluate import property_scores
from src.generate import generate_molecules, sample_batch
from src.models.model import load_checkpoint, save_checkpoint
from src.models.utils import set_seed, setup_gpu
from src.rl import Pool, StepMasks, build_co_set, evaluate_constrained, make_init, reinforce_update, score_molecule


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="property", choices=["property", "constrained"])
    ap.add_argument("--property", default="plogp", choices=["plogp", "qed"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=24)
    ap.add_argument("--reward_decay", type=float, default=None)
    ap.add_argument("--reward_type", default="exp", choices=["exp", "linear"])
    ap.add_argument("--exp_temperature", type=float, default=4.0)
    ap.add_argument("--exp_bias", type=float, default=4.0)
    ap.add_argument("--qed_coeff", type=float, default=3.0)
    ap.add_argument("--plogp_coeff", type=float, default=1 / 3)
    ap.add_argument("--moving_coeff", type=float, default=0.95)
    ap.add_argument("--no_penalty", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--max_atoms", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--n_mols", type=int, default=800)
    ap.add_argument("--modify_size", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--n_eval_mols", type=int, default=800)
    ap.add_argument("--seed", type=int, default=2019)
    a = ap.parse_args()
    if a.reward_decay is None:
        a.reward_decay = 0.97 if (a.task == "property" and a.property == "qed") else 0.9
    a.out = a.out or os.path.join("logs", "rl", a.task if a.task == "constrained" else a.property)
    return a


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
    set_seed(a.seed)
    setup_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = Pool(a.n_jobs)
    model, meta, _ = load_checkpoint(a.ckpt, device)
    masks = StepMasks(a.max_atoms, model.edge_unroll, device)

    cfg = dict(property="plogp" if a.task == "constrained" else a.property, reward_type="imp" if a.task == "constrained"
               else a.reward_type, exp_temperature=a.exp_temperature, exp_bias=a.exp_bias, qed_coeff=a.qed_coeff,
               plogp_coeff=a.plogp_coeff)
    co = ds = None
    if a.task == "constrained":
        co = build_co_set(a.root, pool, a.n_mols)
        _, ds, _ = build_datasets("zinc250k", a.root)
        ds.indices = np.array(co["indices"])
        print(f"[rl] constrained set: {len(co['smiles'])} molecules")
    rng = np.random.default_rng(a.seed)

    params = [p for n, p in model.named_parameters() if p.requires_grad and n != "prior_ln_var"]
    opt = torch.optim.Adam(params, lr=a.lr)
    log_f = open(os.path.join(a.out, "log.jsonl"), "a")
    top, best_reward, baseline = {}, -1e9, None

    bar = tqdm(range(a.iters), desc="rl")
    for it in bar:
        t0 = time.time()
        lr = a.lr * min(1.0, (it + 1) / a.warmup) if a.warmup else a.lr
        for g in opt.param_groups:
            g["lr"] = lr

        # rollouts
        if co is not None:
            ks = rng.integers(0, len(co["smiles"]), a.batch_size)
            inits = [make_init(ds, int(k), rng, a.modify_size) for k in ks]
            orgs = [co["plogp"][int(k)] for k in ks]
            samples, traj = sample_batch(model, a.batch_size, a.temperature, a.max_atoms, True, init=inits,
                                         record=True)
        else:
            orgs = [None] * a.batch_size
            samples, traj = sample_batch(model, a.batch_size, a.temperature, a.max_atoms, True, record=True)
        res = pool.map(score_molecule, [(s.smiles, cfg, o) for s, o in zip(samples, orgs)], disable=True)
        rewards = [r[0] for r in res]
        scores = [r[2] for r in res if r[2] is not None]
        for s, r in zip(samples, res):
            if s.smiles and r[2] is not None and a.task == "property":
                top[s.smiles] = r[2]

        # policy gradient step
        opt.zero_grad()
        loss, baseline = reinforce_update(model, traj, rewards, baseline, masks, a.reward_decay, a.moving_coeff,
                                          not a.no_penalty, a.chunk)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        rec = dict(iter=it, loss=loss, lr=lr, reward=float(np.mean(rewards)),
                   score=float(np.mean(scores)) if scores else 0.0, max_score=float(max(scores)) if scores else 0.0,
                   sec=time.time() - t0)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        bar.set_postfix(reward=f"{rec['reward']:.3f}", score=f"{rec['score']:.3f}", max=f"{rec['max_score']:.3f}")
        if rec["reward"] > best_reward:
            best_reward = rec["reward"]
            save_checkpoint(os.path.join(a.out, "best.pt"), model, meta, it, reward=best_reward)
        save_checkpoint(os.path.join(a.out, "last.pt"), model, meta, it, opt)

    # final report
    if a.task == "property":
        ranked = sorted(top.items(), key=lambda kv: -kv[1])
        with open(os.path.join(a.out, "top_mols.txt"), "w") as f:
            f.write("\n".join(f"{s}\t{v:.4f}" for s, v in ranked[:100]))
        print(f"[final] top-3 {a.property} seen during RL: {[round(v, 3) for _, v in ranked[:3]]}")
        samples, _ = generate_molecules(model, a.eval_n, 512, a.temperature, a.max_atoms, 10)
        print("[final] fresh samples: " + str(property_scores([s.smiles for s in samples])))
    else:
        out = evaluate_constrained(model, ds, co, pool, a.repeat, a.n_eval_mols, a.modify_size, a.max_atoms,
                                   a.temperature, seed=a.seed)
        json.dump(out, open(os.path.join(a.out, "constrained_metrics.json"), "w"), indent=2)
        for d, v in out.items():
            print(f"[final] delta={d}: " + " | ".join(f"{k}={x:.3f}" for k, x in v.items()))
    pool.close()


if __name__ == "__main__":
    main()
