# GEEL on molecules (QM9 / ZINC250k / MOSES)

    pip install -r requirements.txt
    python train.py --dataset qm9
    python -m src.generate --ckpt logs/qm9/best.pt
    python train.py --dataset zinc
    python -m src.generate --ckpt logs/zinc/best.pt --official --nspdk

Unlearning helpers: `load_checkpoint`, `model.per_sequence_nll`, `build_datasets(..., exclude_idx=forget_idx)`,
`generate_graphs`, `evaluate_graphs`.
