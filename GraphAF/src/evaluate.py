# python -m src.evaluate --dataset zinc250k --smiles logs/zinc250k/base/gen/generated_smiles.txt
# python -m src.evaluate --dataset zinc250k --smiles logs/zinc250k/base/gen/generated_smiles.txt --props
# python -m src.evaluate --dataset zinc250k --ckpt logs/zinc250k/base/best.pt --recon

import argparse
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from .chem import min_penalized_logp, qed


# validity, uniqueness (among valid), novelty (among unique valid), valid rate without any resampling
def evaluate_samples(samples: Sequence, train_smiles: Optional[Sequence[str]] = None) -> dict:
    n = max(len(samples), 1)
    valid = [s.smiles for s in samples if s.smiles is not None]
    uniq = set(valid)
    out = {
        "num_samples": len(samples),
        "validity": len(valid) / n,
        "validity_wo_check": sum(s.smiles is not None and s.n_resample == 0 for s in samples) / n,
        "uniqueness": len(uniq) / len(valid) if valid else 0.0,
        "novelty": -1.0,
        "mean_atoms": float(np.mean([s.num_atoms for s in samples])) if samples else 0.0,
    }
    if train_smiles is not None and uniq:
        train = set(train_smiles)
        out["novelty"] = sum(s not in train for s in uniq) / len(uniq)
    return out


# penalized logP and QED of valid unique molecules, top-3 as in the property optimization tables
def property_scores(smiles: Sequence[Optional[str]], progress: bool = True) -> dict:
    rows = []
    for s in tqdm(sorted({s for s in smiles if s}), desc="properties", disable=not progress):
        try:
            m = Chem.MolFromSmiles(s)
            rows.append((s, min_penalized_logp(m), qed(m)))
        except Exception:
            continue
    out = {"num_scored": len(rows)}
    for k, name in ((1, "plogp"), (2, "qed")):
        v = sorted((r[k] for r in rows), reverse=True)
        out[f"{name}_top3"] = v[:3]
        out[f"{name}_mean"] = float(np.mean(v)) if v else 0.0
    return out


def summarize(res: dict) -> str:
    keys = ["validity", "validity_wo_check", "uniqueness", "novelty"]
    s = " | ".join(f"{k}={100 * res[k]:.2f}" for k in keys if k in res and res[k] != -1.0)
    if "reconstruction" in res:
        s += f" | reconstruction={100 * res['reconstruction']:.2f}"
    return s


def main():
    import torch
    from .data import build_datasets, load_smiles, make_loader
    from .generate import reconstruction_rate
    from .models.model import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--root", default="data")
    ap.add_argument("--props", action="store_true")
    ap.add_argument("--recon", action="store_true")
    a = ap.parse_args()

    if a.smiles:
        from .generate import Sample
        raw = [s.strip() for s in open(a.smiles) if s.strip()]
        samples = [Sample(None if s == "None" else s, 0, 0) for s in raw]
        res = evaluate_samples(samples, load_smiles(a.dataset, a.root, "train"))
        print(summarize(res))
        if a.props:
            print(property_scores([s.smiles for s in samples]))
    if a.recon:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta, _ = load_checkpoint(a.ckpt, device)
        _, _, val_ds = build_datasets(a.dataset, a.root)
        rate = reconstruction_rate(model, make_loader(val_ds, 64, False, 2))
        print(summarize({"reconstruction": rate}))


if __name__ == "__main__":
    main()
