"""Pre-launch smoke test for a fine-tune config.

Exercises the full forward + backward + optimizer + val + checkpoint
round-trip on a handful of real batches, so crashes (NaN loss, OOM,
ckpt-load key mismatch, val pipeline break, etc.) surface in ~30 s
instead of after 15 min into a real epoch.

Usage:

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_finetune.py \
        --config configs/ft_a_softargmax_mild.yaml \
        --resume runs/deeptfus/ckpt_best.pt \
        --n-train 3 --n-val 1

Prints the per-term loss magnitudes to verify the added term's
contribution is in the expected range before launching a multi-hour run.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
from torch.utils.data import DataLoader

from data import TFUScapesDataset, discover_splits
from losses import DeepTFUSLoss
from models import build_deeptfus
from train import apply_runtime, load_config, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True, help="ckpt to load model weights from")
    ap.add_argument("--n-train", type=int, default=3, help="train batches")
    ap.add_argument("--n-val", type=int, default=1, help="val batches")
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}  config={args.config}")

    splits = discover_splits()
    print(f"[smoke] splits: train={len(splits['train'])} val={len(splits['val'])}")
    train_files = splits["train"][: args.n_train * int(cfg["train"]["batch_size"])]
    val_files = splits["val"][: args.n_val * int(cfg["train"]["batch_size"])]
    res = int(cfg["data"]["resolution"])
    n_pts = int(cfg["data"]["n_transducer_points"])
    train_ds = TFUScapesDataset(train_files, resolution=res, n_transducer_points=n_pts, train=True)
    val_ds = TFUScapesDataset(val_files, resolution=res, n_transducer_points=n_pts, train=False)
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    model = build_deeptfus(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model, dtype, _ = apply_runtime(
        model,
        precision=str(cfg["train"].get("precision", "fp32")),
        grad_ckpt_encoder=bool(cfg["train"].get("grad_checkpoint_encoder", False)),
        channels_last=False,
        compile=False,
    )
    print(f"[smoke] model params={n_params:,}  dtype={dtype}")

    print(f"[smoke] loading ckpt: {args.resume}")
    full_state = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(full_state["model"])
    print(f"[smoke]   loaded weights from epoch={full_state.get('epoch')}")

    loss_fn = DeepTFUSLoss(
        alpha=cfg["loss"]["alpha"],
        grad_weight=cfg["loss"]["grad_weight"],
        focal_weight=float(cfg["loss"].get("focal_weight", 0.0)),
        focal_temperature=float(cfg["loss"].get("focal_temperature", 0.1)),
        focal_warmup_off=int(cfg["loss"].get("focal_warmup_off", 3)),
        focal_warmup_epochs=int(cfg["loss"].get("focal_warmup_epochs", 10)),
        dice_weight=float(cfg["loss"].get("dice_weight", 0.0)),
        dice_threshold=float(cfg["loss"].get("dice_threshold", 0.5)),
        dice_temperature=float(cfg["loss"].get("dice_temperature", 0.05)),
    ).to(device)
    loss_fn.set_epoch(0)
    print(
        f"[smoke] loss: alpha={loss_fn.alpha} grad_w={loss_fn.grad_weight} "
        f"focal_w={loss_fn.focal_weight} focal_tau={loss_fn.focal_temperature} "
        f"dice_w={loss_fn.dice_weight} "
        f"lambda_focal_now={loss_fn.lambda_focal_now()}"
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    print("[smoke] === train steps ===")
    model.train()
    t_total = time.perf_counter()
    for i, batch in enumerate(train_loader):
        if i >= args.n_train:
            break
        t0 = time.perf_counter()
        ct = batch["ct"].to(device).to(dtype)
        tr = batch["transducer"].to(device).to(dtype)
        tgt = batch["pressure"].to(device).to(dtype)
        pred = model(ct, tr)
        total, parts = loss_fn(pred, tgt)
        if not torch.isfinite(total):
            print(f"[smoke] NON-FINITE LOSS at step {i}: {parts}")
            return 1
        optim.zero_grad(set_to_none=True)
        total.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"])))
        if not (gnorm == gnorm):
            print(f"[smoke] NaN grad norm at step {i}")
            return 1
        optim.step()
        dt = time.perf_counter() - t0
        wmse = parts["weighted_mse"]
        gl1 = parts["gradient_l1"]
        wmse_c = wmse
        gl1_c = loss_fn.grad_weight * gl1
        focal_c = parts.get("lambda_focal", 0.0) * parts.get("focal_l1_vox", 0.0)
        dice_c = parts.get("lambda_dice", 0.0) * (1.0 - parts.get("soft_dice_fwhm", 0.0))
        print(
            f"[smoke]   step {i}: total={float(total):.6f}  "
            f"wmse={wmse:.4e} (contrib {wmse_c:.4e})  "
            f"gl1={gl1:.4e} (contrib {gl1_c:.4e})  "
            f"focal_l1={parts.get('focal_l1_vox', 0.0):.3f} (contrib {focal_c:.4e})  "
            f"soft_dice={parts.get('soft_dice_fwhm', 0.0):.3f} (contrib {dice_c:.4e})  "
            f"gnorm={gnorm:.3f}  dt={dt:.2f}s"
        )

    print("[smoke] === val pass ===")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_val:
                break
            ct = batch["ct"].to(device).to(dtype)
            tr = batch["transducer"].to(device).to(dtype)
            tgt = batch["pressure"].to(device).to(dtype)
            pred = model(ct, tr)
            total, parts = loss_fn(pred, tgt)
            print(
                f"[smoke]   val {i}: total={float(total):.6f}  "
                f"wmse={parts['weighted_mse']:.4e}  "
                f"soft_dice={parts.get('soft_dice_fwhm', 0.0):.3f}"
            )

    print("[smoke] === checkpoint round-trip ===")
    with tempfile.TemporaryDirectory() as td:
        ck_path = Path(td) / "smoke_ck.pt"
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": 0}, ck_path)
        size = ck_path.stat().st_size
        loaded = torch.load(ck_path, map_location="cpu", weights_only=False)
        n_keys = len(loaded["model"])
        print(f"[smoke]   wrote {size/1e6:.1f} MB ({n_keys} param tensors), loaded back OK")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[smoke] peak GPU mem = {gpu_mem:.2f} GiB")
    print(f"[smoke] === PASS in {time.perf_counter() - t_total:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
