"""DeepTFUS training entry point.

Reads `configs/base.yaml`, trains, writes:
  <run_dir>/ckpt_last.pt, ckpt_best.pt, [ckpt_epoch_<N>.pt if save_every]
  <run_dir>/train_log.csv, train_step_log.csv
  <run_dir>/config.snapshot.yaml
  <run_dir>/summary.json

Selects device automatically (CUDA > MPS > CPU). Single-config trainer; no
CLI argument soup. To override fields ad hoc, pass --overrides "key.path=value".

Resume from a saved checkpoint with `python train.py --resume <path>`. Resume
restores model + optimizer + scheduler state and continues the SAME WandB run
by run-id (so the live charts append rather than starting fresh). CSV files
detect their own headers and append rather than overwriting.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Required for the recommended H100 path (pure-bf16 + encoder grad-ckpt).
# Set before any torch CUDA init so the allocator picks it up. Harmless
# on CPU/MPS.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import TFUScapesDataset, discover_splits
from eval import per_sample_metrics
from losses import DeepTFUSLoss
from models import build_deeptfus

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


_DTYPE = {"fp32": torch.float32, "pure-bf16": torch.bfloat16}


def _init_wandb(
    cfg: dict,
    run_dir: Path,
    n_params: int,
    dtype: torch.dtype,
    resume_run_id: str | None = None,
):
    """Init a wandb run if configured. Returns the run handle or None.

    When `resume_run_id` is provided, the wandb run is reconnected to the
    exact prior run (same id + resume="must"), so subsequent step/epoch logs
    are appended to the existing curves rather than starting fresh charts.

    Disabled paths: `train.wandb_project` is null, OR `wandb` package missing,
    OR no `WANDB_API_KEY` and no prior `wandb login`. In any of those, the
    function returns None silently and training proceeds without telemetry.
    """
    project = cfg["train"].get("wandb_project")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("[train] wandb_project set but `wandb` not installed; skipping telemetry")
        return None
    if not (os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()):
        print("[train] wandb_project set but no WANDB_API_KEY and no ~/.netrc; skipping telemetry")
        return None
    try:
        init_kwargs = dict(
            project=str(project),
            entity=cfg["train"].get("wandb_entity") or None,
            dir=str(run_dir),
            config={**cfg, "_n_params": int(n_params), "_dtype": str(dtype)},
        )
        if resume_run_id:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
            print(f"[train] resuming wandb run id={resume_run_id}")
        else:
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        print(f"[train] wandb run: {run.url}")
        return run
    except Exception as e:
        print(f"[train] wandb.init failed (non-fatal): {e}")
        return None


def apply_runtime(
    model,
    precision: str,
    grad_ckpt_encoder: bool,
    channels_last: bool = False,
    compile: bool = False,
) -> tuple[torch.nn.Module, torch.dtype, bool]:
    """Apply the precision/memory-layout/compile knobs requested by config.

    Returns (model, dtype, channels_last_active). `channels_last_active` is
    True when the model lives in `channels_last_3d` memory format; callers
    use it to decide whether to convert spatial-input batches the same way.
    """
    if precision not in _DTYPE:
        raise ValueError(f"train.precision must be one of {sorted(_DTYPE)}: got {precision!r}")
    dtype = _DTYPE[precision]
    if dtype != torch.float32:
        model.to(dtype)
    if channels_last:
        model.to(memory_format=torch.channels_last_3d)
    if grad_ckpt_encoder:
        from torch.utils.checkpoint import checkpoint
        for stage in (model.down1, model.down2, model.down3, model.bottleneck):
            orig = stage.forward
            def make(o):
                return lambda x, z: checkpoint(o, x, z, use_reentrant=False)
            stage.forward = make(orig)
    if compile:
        # `default` mode is safer than `reduce-overhead`: the latter uses CUDA
        # graphs which our DynamicConv3d's data-dependent kernel shape would
        # likely break. `default` does graph capture + kernel fusion only.
        model = torch.compile(model, mode="default")
    return model, dtype, bool(channels_last)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_nested(cfg: dict, dotted: str, value):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    # parse value as yaml so "[bottleneck,level3]" or "1e-3" works
    cur[parts[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key.path=value: {ov!r}")
        k, v = ov.split("=", 1)
        _set_nested(cfg, k.strip(), v.strip())
    return cfg


def quick_val(model, loader, device, dtype, loss_fn=None,
              channels_last_active=False, max_n=None) -> dict:
    """Validation pass. Returns rel_l2 / focal_mm / max_p metrics (the paper's
    three) plus, if `loss_fn` is given, the per-sample-averaged validation loss
    in the same units as the training loss (`total`, `weighted_mse`,
    `gradient_l1`). The val loss is what makes train↔val gap directly readable
    (detects over- / under-fitting cleanly), so always pass `loss_fn` from the
    training loop.
    """
    model.eval()
    r, f, m = [], [], []
    val_total_sum = 0.0
    val_wmse_sum = 0.0
    val_grad_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if max_n is not None and i >= max_n:
                break
            ct = b["ct"].to(device).to(dtype)
            tr = b["transducer"].to(device).to(dtype)
            tgt = b["pressure"].to(device).to(dtype)
            if channels_last_active:
                ct = ct.to(memory_format=torch.channels_last_3d)
                tgt = tgt.to(memory_format=torch.channels_last_3d)
            pred = model(ct, tr)
            if loss_fn is not None:
                total, parts = loss_fn(pred, tgt)
                val_total_sum += float(total.detach())
                val_wmse_sum += parts["weighted_mse"]
                val_grad_sum += parts["gradient_l1"]
                n_batches += 1
            pred_f = pred.squeeze(1).float()
            tgt_f = tgt.squeeze(1).float()
            for bi in range(pred_f.shape[0]):
                pm = per_sample_metrics(
                    pred_f[bi].cpu().numpy(), tgt_f[bi].cpu().numpy(),
                    voxel_size_mm=0.5 * (256.0 / pred_f.shape[-1]),
                )
                r.append(pm["relative_l2"])
                f.append(pm["focal_position_error_mm"])
                m.append(pm["max_pressure_error"])
    model.train()
    out = {
        "rel_l2_mean": float(np.mean(r)) if r else float("nan"),
        "rel_l2_median": float(np.median(r)) if r else float("nan"),
        "focal_mm_mean": float(np.mean(f)) if f else float("nan"),
        "max_p_mean": float(np.mean(m)) if m else float("nan"),
        "n": len(r),
    }
    if n_batches > 0:
        out["val_total"] = val_total_sum / n_batches
        out["val_weighted_mse"] = val_wmse_sum / n_batches
        out["val_gradient_l1"] = val_grad_sum / n_batches
    else:
        out["val_total"] = float("nan")
        out["val_weighted_mse"] = float("nan")
        out["val_gradient_l1"] = float("nan")
    return out


EPOCH_CSV_HEADER = (
    "epoch,train_total,weighted_mse,gradient_l1,grad_norm,"
    "val_rel_l2_mean,val_rel_l2_median,val_focal_mm_mean,val_max_p_mean,"
    "val_total,val_weighted_mse,val_gradient_l1,"
    "lr,wall_s\n"
)
STEP_CSV_HEADER = (
    "global_step,epoch,total,weighted_mse,gradient_l1,focal_l1_vox,lambda_focal,"
    "soft_dice_fwhm,lambda_dice,grad_norm,"
    "lr,gpu_mem_GiB,step_wall_s\n"
)


def _open_csv_for_run(path: Path, header: str, resume: bool):
    """Open a CSV in append mode if resuming AND the file exists with the
    expected header, otherwise open fresh and write the header.
    """
    if resume and path.exists():
        existing_header = path.read_text().splitlines()[0] + "\n" if path.stat().st_size > 0 else ""
        if existing_header == header:
            return open(path, "a")
        # header mismatch: archive old file and start fresh so we don't lie
        path.rename(path.with_suffix(path.suffix + ".pre_resume.bak"))
    f = open(path, "w")
    f.write(header)
    f.flush()
    return f


def train(cfg: dict, device: torch.device | None = None,
          splits: dict[str, list[str]] | None = None,
          resume_from: str | None = None,
          finetune: bool = False) -> dict:
    """Train DeepTFUS.

    Modes:
      - fresh run (resume_from=None): standard training from scratch.
      - resume (resume_from=ckpt, finetune=False): restore model + optimizer +
        scheduler + epoch counter + WandB run id; pick up exactly where the
        prior run left off. Cfg model/loss sections are overridden from the
        ckpt to keep architecture consistent.
      - fine-tune (resume_from=ckpt, finetune=True): load model weights ONLY;
        start fresh epoch counter, optimizer, scheduler, WandB run, CSV files,
        and use the LIVE cfg's model/loss (do NOT inherit from ckpt). Used
        to continue training a converged base ckpt with a modified loss
        (e.g. the variant A-E configs that add a soft-argmax or Dice term).
    """
    set_seed(int(cfg["train"]["seed"]))
    device = device or pick_device()
    print(f"[train] device={device}")

    # ---- Load resume checkpoint early so we can use its config + state. ----
    resume_state = None
    if resume_from is not None:
        mode = "fine-tuning from" if finetune else "resuming from"
        print(f"[train] {mode} checkpoint: {resume_from}")
        full_state = torch.load(resume_from, map_location="cpu", weights_only=False)
        if finetune:
            # Fine-tune mode: keep only the model weights. Use the LIVE cfg's
            # model/loss as authoritative (we want the new architecture or new
            # loss term, not the saved one).
            resume_state = {"model": full_state["model"]}
            print(f"[train]   fine-tune: loading model weights from ckpt; "
                  f"starting fresh optim/scheduler/epoch/wandb run")
        else:
            resume_state = full_state
            # Resume mode: prefer the resumed run's config to keep architecture
            # consistent. Override the input cfg's model + loss sections so the
            # rebuilt model matches; keep the live cfg's train/output for
            # runtime knobs.
            for k in ("model", "loss"):
                if k in resume_state.get("config", {}):
                    cfg[k] = resume_state["config"][k]
            print(f"[train]   resuming from epoch {resume_state['epoch'] + 1} (saved through epoch {resume_state['epoch']})")

    if splits is None:
        splits = discover_splits()
    print(f"[train] splits: " + str({k: len(v) for k, v in splits.items()}))
    if not splits["train"]:
        raise RuntimeError("No train files. Run `python scripts/download_data.py` first.")
    val_files = splits["val"] or splits["train"][-2:]

    res = int(cfg["data"]["resolution"])
    n_pts = int(cfg["data"]["n_transducer_points"])
    train_ds = TFUScapesDataset(splits["train"], resolution=res, n_transducer_points=n_pts, train=True)
    val_ds = TFUScapesDataset(val_files, resolution=res, n_transducer_points=n_pts, train=False)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    if device.type == "mps":
        nw = 0
    # persistent_workers keeps dataloader workers alive across epochs
    # (avoids forking a 2+ GB process every epoch boundary, which is the
    # dominant source of epoch-time variance otherwise). Requires nw > 0.
    persist = nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        drop_last=False, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        persistent_workers=persist,
    )

    model = build_deeptfus(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    precision = str(cfg["train"].get("precision", "fp32"))
    grad_ckpt_encoder = bool(cfg["train"].get("grad_checkpoint_encoder", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    compile_model = bool(cfg["train"].get("compile", False))
    model, dtype, channels_last_active = apply_runtime(
        model, precision=precision, grad_ckpt_encoder=grad_ckpt_encoder,
        channels_last=channels_last, compile=compile_model,
    )
    print(f"[train] DeepTFUS params={n_params:,}  precision={precision}  grad_ckpt_encoder={grad_ckpt_encoder}  channels_last={channels_last_active}  compile={compile_model}")

    # ---- Restore model weights from checkpoint, if resuming. ----
    if resume_state is not None:
        # state_dict was saved before potential torch.compile wrapping or
        # channels_last conversion, but those don't change parameter names.
        # If torch.compile is active, the wrapped module exposes the original
        # nn.Module via `_orig_mod`; load into that to avoid a key prefix.
        target = getattr(model, "_orig_mod", model)
        target.load_state_dict(resume_state["model"])

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

    epochs = int(cfg["train"]["epochs"])
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    if resume_state is not None and not finetune:
        optim.load_state_dict(resume_state["optim"])
        if "scheduler" in resume_state:
            sched.load_state_dict(resume_state["scheduler"])
        else:
            # Backward compat: older ckpts don't carry scheduler state. Advance
            # it by the resumed-epoch count to recreate the right LR.
            for _ in range(resume_state["epoch"] + 1):
                sched.step()
    grad_clip = float(cfg["train"]["grad_clip"])

    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    csv_path = run_dir / "train_log.csv"
    step_csv_path = run_dir / "train_step_log.csv"
    # Fine-tune mode writes fresh CSV files and a fresh WandB run; full-resume
    # mode appends to existing CSVs and reconnects the prior WandB run.
    csv_append = (resume_state is not None) and not finetune
    csv_f = _open_csv_for_run(csv_path, EPOCH_CSV_HEADER, resume=csv_append)
    csv_f.close()
    step_csv_f = _open_csv_for_run(step_csv_path, STEP_CSV_HEADER, resume=csv_append)
    step_csv_f.close()

    resume_run_id = resume_state.get("wandb_run_id") if (resume_state and not finetune) else None
    wandb_run = _init_wandb(cfg, run_dir, n_params, dtype, resume_run_id=resume_run_id)
    wandb_run_id = wandb_run.id if wandb_run is not None else None

    save_every = int(cfg["train"].get("save_every_epochs", 0))

    best_val = resume_state.get("best_val", float("inf")) if (resume_state and not finetune) else float("inf")
    val_every = int(cfg["train"].get("val_every", 1))
    t_start = time.time()
    # When resuming, account for the time already spent so `wall_s` keeps
    # growing across the join point rather than restarting at 0. Fine-tune
    # mode starts wall_s fresh at 0 (we're treating this as a new run).
    wall_offset = resume_state["wall_s"] if (resume_state and not finetune and "wall_s" in resume_state) else 0.0
    if resume_state is not None and not finetune:
        print(
            f"[train] resume wall_offset={wall_offset:.2f}s, "
            f"starting at epoch={resume_state['epoch']+1}/{epochs}, "
            f"global_step={resume_state['global_step']}, "
            f"sched_last_epoch={sched.last_epoch}, "
            f"best_val_so_far={best_val:.6f}"
        )
    global_step = resume_state["global_step"] if (resume_state and not finetune) else 0
    start_epoch = (resume_state["epoch"] + 1) if (resume_state and not finetune) else 0
    for ep in range(start_epoch, epochs):
        loss_fn.set_epoch(ep)
        model.train()
        ep_total = 0.0
        ep_wmse = 0.0
        ep_grad = 0.0
        ep_gnorm = 0.0
        for batch in train_loader:
            t_step = time.perf_counter()
            ct = batch["ct"].to(device).to(dtype)
            tr = batch["transducer"].to(device).to(dtype)
            tgt = batch["pressure"].to(device).to(dtype)
            if channels_last_active:
                ct = ct.to(memory_format=torch.channels_last_3d)
                tgt = tgt.to(memory_format=torch.channels_last_3d)
            pred = model(ct, tr)
            total, parts = loss_fn(pred, tgt)
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {ep}: {parts}. Aborting.")
            optim.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            else:
                gnorm = float(sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None) ** 0.5)
            optim.step()
            step_total = float(total.detach())
            ep_total += step_total
            ep_wmse += parts["weighted_mse"]
            ep_grad += parts["gradient_l1"]
            ep_gnorm += gnorm
            step_wall = time.perf_counter() - t_step
            gpu_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

            with open(step_csv_path, "a") as f:
                f.write(
                    f"{global_step},{ep},{step_total:.6f},{parts['weighted_mse']:.6f},"
                    f"{parts['gradient_l1']:.6f},{parts.get('focal_l1_vox', 0.0):.6f},"
                    f"{parts.get('lambda_focal', 0.0):.6e},"
                    f"{parts.get('soft_dice_fwhm', 0.0):.6f},"
                    f"{parts.get('lambda_dice', 0.0):.6e},"
                    f"{gnorm:.6f},"
                    f"{optim.param_groups[0]['lr']:.6f},{gpu_mem:.3f},{step_wall:.4f}\n"
                )
            if wandb_run is not None:
                wandb_run.log({
                    "step/total": step_total,
                    "step/weighted_mse": parts["weighted_mse"],
                    "step/gradient_l1": parts["gradient_l1"],
                    "step/focal_l1_vox": parts.get("focal_l1_vox", 0.0),
                    "step/lambda_focal": parts.get("lambda_focal", 0.0),
                    "step/soft_dice_fwhm": parts.get("soft_dice_fwhm", 0.0),
                    "step/lambda_dice": parts.get("lambda_dice", 0.0),
                    "step/grad_norm": gnorm,
                    "step/lr": optim.param_groups[0]["lr"],
                    "step/gpu_mem_GiB": gpu_mem,
                    "step/wall_s": step_wall,
                    "epoch": ep,
                }, step=global_step)
            global_step += 1
        sched.step()

        n = max(1, len(train_loader))
        avg_total = ep_total / n
        avg_wmse = ep_wmse / n
        avg_grad = ep_grad / n
        avg_gnorm = ep_gnorm / n

        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            vm = quick_val(model, val_loader, device, dtype, loss_fn=loss_fn,
                           channels_last_active=channels_last_active)
        else:
            vm = {"rel_l2_mean": float("nan"), "rel_l2_median": float("nan"),
                  "focal_mm_mean": float("nan"), "max_p_mean": float("nan"),
                  "val_total": float("nan"), "val_weighted_mse": float("nan"),
                  "val_gradient_l1": float("nan"), "n": 0}

        wall = (time.time() - t_start) + wall_offset
        with open(csv_path, "a") as f:
            f.write(
                f"{ep},{avg_total:.6f},{avg_wmse:.6f},{avg_grad:.6f},{avg_gnorm:.6f},"
                f"{vm['rel_l2_mean']:.6f},{vm['rel_l2_median']:.6f},"
                f"{vm['focal_mm_mean']:.4f},{vm['max_p_mean']:.4f},"
                f"{vm['val_total']:.6f},{vm['val_weighted_mse']:.6f},{vm['val_gradient_l1']:.6f},"
                f"{optim.param_groups[0]['lr']:.6f},{wall:.1f}\n"
            )
        print(
            f"[train] ep {ep+1:02d}/{epochs} "
            f"loss={avg_total:.4f} (wmse={avg_wmse:.3f}, grad={avg_grad:.3f}, gn={avg_gnorm:.3f}) "
            f"val rel_l2={vm['rel_l2_mean']:.3f} val_total={vm['val_total']:.4f} "
            f"focal_mm={vm['focal_mm_mean']:.2f} max_p={vm['max_p_mean']:.3f} ({wall:.0f}s)"
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch/train_total": avg_total,
                "epoch/train_weighted_mse": avg_wmse,
                "epoch/train_gradient_l1": avg_grad,
                "epoch/train_grad_norm": avg_gnorm,
                "epoch/val_rel_l2_mean": vm["rel_l2_mean"],
                "epoch/val_rel_l2_median": vm["rel_l2_median"],
                "epoch/val_focal_mm_mean": vm["focal_mm_mean"],
                "epoch/val_max_p_mean": vm["max_p_mean"],
                "epoch/val_total": vm["val_total"],
                "epoch/val_weighted_mse": vm["val_weighted_mse"],
                "epoch/val_gradient_l1": vm["val_gradient_l1"],
                "epoch/lr": optim.param_groups[0]["lr"],
                "epoch/wall_s": wall,
                "epoch": ep,
            }, step=global_step)

        target = getattr(model, "_orig_mod", model)
        ckpt_payload = {
            "model": target.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "config": cfg,
            "epoch": ep,
            "val": vm,
            "global_step": global_step,
            "wall_s": wall,
            "best_val": best_val if (np.isnan(vm["rel_l2_mean"]) or vm["rel_l2_mean"] >= best_val) else vm["rel_l2_mean"],
            "wandb_run_id": wandb_run_id,
        }
        torch.save(ckpt_payload, run_dir / "ckpt_last.pt")
        if not np.isnan(vm["rel_l2_mean"]) and vm["rel_l2_mean"] < best_val:
            best_val = vm["rel_l2_mean"]
            ckpt_payload["best_val"] = best_val
            torch.save(ckpt_payload, run_dir / "ckpt_best.pt")
        if save_every > 0 and ((ep + 1) % save_every == 0 or ep == epochs - 1):
            torch.save(ckpt_payload, run_dir / f"ckpt_epoch_{ep:03d}.pt")

    summary = {
        "best_val_rel_l2": best_val,
        "epochs_trained": epochs,
        "n_train": len(splits["train"]),
        "n_val": len(val_files),
        "n_params": n_params,
        "precision": precision,
        "grad_checkpoint_encoder": grad_ckpt_encoder,
        "wall_s": (time.time() - t_start) + wall_offset,
        "resumed_from": resume_from,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] done in {summary['wall_s']:.0f}s  best_val_rel_l2={best_val:.4f}")
    if wandb_run is not None:
        wandb_run.summary.update({
            "best_val_rel_l2": best_val,
            "wall_s": summary["wall_s"],
            "n_train": summary["n_train"],
            "n_val": summary["n_val"],
        })
        ckpt_best = run_dir / "ckpt_best.pt"
        if ckpt_best.exists():
            try:
                import wandb
                art = wandb.Artifact(f"{wandb_run.name}-ckpt-best", type="model")
                art.add_file(str(ckpt_best))
                wandb_run.log_artifact(art)
            except Exception as e:
                print(f"[train] wandb artifact upload failed (non-fatal): {e}")
        wandb_run.finish()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help='dotted key=yaml-value overrides, e.g. train.epochs=2')
    ap.add_argument("--resume", default=None,
                    help="path to a saved ckpt to resume from; reuses model/optim/scheduler state, continues WandB run, appends CSVs")
    ap.add_argument("--finetune", action="store_true",
                    help="when combined with --resume, load only the model weights; "
                         "start fresh optim/scheduler/epoch/wandb run/CSVs. Use the "
                         "live --config's model/loss instead of the ckpt's.")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg, resume_from=args.resume, finetune=args.finetune)


if __name__ == "__main__":
    main()
