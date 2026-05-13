"""Local Mac smoke test for the entire reproduction pipeline.

Confirms (with model quality intentionally meaningless):

  1. data download script can fetch a small subset end-to-end
  2. model instantiates at config defaults
  3. forward + backward pass on a real downloaded sample is finite
  4. one-epoch tiny training run completes and saves a checkpoint
  5. checkpoint loads back
  6. eval harness runs on the checkpoint and writes metrics.json,
     per_voxel_mse.npy, and per-sample prediction npz files

Exits 0 on success. Prints a clear failure message and exits non-zero
otherwise. Runtime on M1 Pro / MPS: roughly 3-5 minutes.

Heavy fields (resolution, epochs, batch size, model width) are
overridden in-process to small values appropriate for CPU/MPS. This
script does NOT change configs/base.yaml; it builds a local config dict
from it.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import TFUScapesDataset, discover_splits
from eval import evaluate
from losses import DeepTFUSLoss
from models import build_deeptfus
from train import train as run_train


VERIFY_RUN_DIR = REPO_ROOT / "runs" / "_local_verify"
VERIFY_EVAL_DIR = VERIFY_RUN_DIR / "eval_test"


def _step(n: int, name: str) -> None:
    print(f"\n[{n}/7] {name}", flush=True)


def _fail(msg: str, exc: BaseException | None = None) -> None:
    print(f"\n[local_verify] FAIL: {msg}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(exc)
    sys.exit(1)


def _ensure_subset() -> dict:
    splits = discover_splits()
    if splits["train"] and splits["val"] and splits["test"]:
        return splits
    print("[local_verify] running scripts/download_data.py --mode subset")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "download_data.py"),
           "--mode", "subset", "--n-train", "4", "--n-val", "2", "--n-test", "2", "--per-subject", "2"]
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        _fail("download_data.py exited non-zero")
    splits = discover_splits()
    return splits


def _build_local_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["data"]["resolution"] = 64
    cfg["data"]["n_transducer_points"] = 128
    cfg["model"]["base_width"] = 8
    cfg["model"]["cond_dim"] = 32
    cfg["model"]["n_transducer_freqs"] = 4
    cfg["model"]["cross_attention_heads"] = 2
    cfg["model"]["cross_attention_levels"] = ["bottleneck"]
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 1
    cfg["train"]["num_workers"] = 0
    # Smoke test runs at 64^3 on CPU/MPS. Force fp32 + no grad-ckpt; bf16 on
    # MPS/CPU is either slow or unsupported, and the tiny model has no memory
    # pressure to need checkpointing.
    cfg["train"]["precision"] = "fp32"
    cfg["train"]["grad_checkpoint_encoder"] = False
    cfg["train"]["channels_last"] = False
    cfg["train"]["compile"] = False
    cfg["train"]["wandb_project"] = None  # smoke test never sends telemetry
    cfg["eval"]["save_predictions"] = True
    cfg["output"]["run_dir"] = str(VERIFY_RUN_DIR)
    return cfg


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    if VERIFY_RUN_DIR.exists():
        shutil.rmtree(VERIFY_RUN_DIR)

    _step(1, "download subset (or use cached files)")
    splits = _ensure_subset()
    if not splits["train"] or not splits["test"]:
        _fail(f"download produced empty split: {[(k, len(v)) for k, v in splits.items()]}")
    print(f"  found {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")

    cfg = _build_local_cfg()
    device = _pick_device()
    print(f"  device={device}")

    _step(2, "instantiate model and run one forward pass")
    try:
        model = build_deeptfus(**cfg["model"]).to(device)
        ds = TFUScapesDataset(
            splits["train"][:2],
            resolution=cfg["data"]["resolution"],
            n_transducer_points=cfg["data"]["n_transducer_points"],
            train=True,
        )
        b = ds[0]
        ct = b["ct"].unsqueeze(0).to(device)
        tr = b["transducer"].unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(ct, tr)
        assert y.shape == ct.shape, f"output shape {y.shape} != input shape {ct.shape}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ok, output shape {tuple(y.shape)}, params={n_params:,}")
    except Exception as e:
        _fail("forward pass failed", e)

    _step(3, "forward + backward on a real sample yields finite loss")
    try:
        loss_fn = DeepTFUSLoss(
            alpha=cfg["loss"]["alpha"],
            grad_weight=cfg["loss"]["grad_weight"],
        ).to(device)
        tgt = b["pressure"].unsqueeze(0).to(device)
        pred = model(ct, tr)
        total, parts = loss_fn(pred, tgt)
        assert torch.isfinite(total), f"non-finite loss: {parts}"
        total.backward()
        n_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert np.isfinite(n_grad), "non-finite gradient"
        print(f"  ok, total={total.item():.4f}, parts={parts}, sum|grad|={n_grad:.4f}")
    except Exception as e:
        _fail("forward+backward failed", e)

    _step(4, "one-epoch training run and checkpoint write")
    try:
        tiny_splits = {
            "train": splits["train"][:3],
            "val": splits["val"][:1] if splits["val"] else splits["train"][-1:],
            "test": splits["test"][:1] if splits["test"] else splits["train"][:1],
        }
        summary = run_train(cfg, device=device, splits=tiny_splits)
        ck = VERIFY_RUN_DIR / "ckpt_last.pt"
        assert ck.exists(), f"checkpoint not written: {ck}"
        print(f"  ok, wall={summary['wall_s']:.1f}s, ckpt={ck.name}")
    except Exception as e:
        _fail("training run failed", e)

    _step(5, "checkpoint loads back")
    try:
        state = torch.load(VERIFY_RUN_DIR / "ckpt_last.pt", map_location="cpu", weights_only=False)
        model2 = build_deeptfus(**state["config"]["model"]).to(device)
        model2.load_state_dict(state["model"])
        print("  ok, state_dict loaded into a fresh model")
    except Exception as e:
        _fail("checkpoint load failed", e)

    _step(6, "eval harness produces metrics.json + per_voxel_mse.npy + predictions/")
    try:
        test_ds = TFUScapesDataset(
            splits["test"][:2] if splits["test"] else splits["train"][:2],
            resolution=cfg["data"]["resolution"],
            n_transducer_points=cfg["data"]["n_transducer_points"],
            train=False,
        )
        VERIFY_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        res = evaluate(
            model2, test_ds, device,
            voxel_size_mm_at_native=cfg["eval"]["voxel_size_mm"],
            off_target_min_dist_mm=cfg["eval"]["off_target_min_dist_mm"],
            n_warmup=1,
            save_predictions_dir=VERIFY_EVAL_DIR / "predictions",
            per_voxel_error_path=VERIFY_EVAL_DIR / "per_voxel_mse.npy",
        )
        import json
        (VERIFY_EVAL_DIR / "metrics.json").write_text(json.dumps(res, indent=2))
        assert (VERIFY_EVAL_DIR / "per_voxel_mse.npy").exists()
        assert any((VERIFY_EVAL_DIR / "predictions").glob("*.npz"))
        print(f"  ok, n={res['n_samples']}, predictions={len(list((VERIFY_EVAL_DIR / 'predictions').glob('*.npz')))}")
    except Exception as e:
        _fail("eval failed", e)

    print("\n[local_verify] PASS  (all 6 steps green; pipeline is wired correctly)")
    print(f"  artifacts under: {VERIFY_RUN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
