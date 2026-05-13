"""Evaluation harness producing the blog-post metric suite.

Most metrics are computed in the paper's normalized log space
(`p_norm = log(1 + p / p.max())`, peak = log(2)). The TUSNet-spec metrics
(focal_pressure_error, focal_volume_fwhm, focal_iou_fwhm) are computed in
the un-log-transformed normalized pressure space (apply `np.expm1` first)
since FWHM is defined on raw pressure, not on log pressure.

Metrics per sample (all kept, none replaced):

  Paper-matched (for Table 1 comparison, log-normalized space):
    relative_l2               ||pred - target|| / ||target||
    focal_position_error_mm   euclid distance between argmax of pred and
                              target, in mm. voxel_size_mm comes from
                              cfg.eval.voxel_size_mm scaled by 256/D.
    max_pressure_error        |pred.max() - target.max()| / target.max()

  TUSNet-spec (operate on expm1(p_norm) ∈ [0, 1]):
    focal_pressure_error      |P̂(r_gt) - P(r_gt)| / P(r_gt), where r_gt is
                              the ground-truth argmax. Tests how well the
                              prediction matches the GT pressure value AT
                              the GT focal location (a different question
                              from max_pressure_error and from focal_position_error).
    focal_volume_pred_mm3     Volume in mm^3 of the connected component
                              around the predicted peak where p_norm >= 0.5*peak.
    focal_volume_gt_mm3       Same for the GT.
    focal_volume_error        |V_pred - V_gt| / V_gt
    focal_iou_fwhm            3D intersection-over-union of the predicted and
                              GT FWHM masks (connected-component around peak).
    off_target_volume_mm3     Volume of secondary lobes above the -6 dB
                              threshold and outside `off_target_min_dist_mm`
                              of the predicted focal point.
    off_target_lobe_count     Connected-component count of the same mask.
    focal_dice                Dice on the -6 dB iso-surface (p > 0.5*peak
                              in raw amplitude = p_norm > log(1.5)); legacy,
                              kept for backwards compat with earlier eval runs.

  Performance:
    inference_latency_s       wall-clock per single-volume forward (warmed up)

Aggregate (across the test set):
  - all per-sample metrics: mean / std / median / min / max / p05 / p95
  - per_voxel_mean_squared_error.npy  (D, H, W) array on disk
  - per_sample.csv with the full schema for downstream analysis

The schema for per_sample.csv (one row per test sample):
  sample_id, focal_position_error_mm, focal_pressure_error,
  focal_volume_pred_mm3, focal_volume_gt_mm3, focal_volume_error,
  focal_iou_fwhm, focal_dice, max_pressure_error, relative_l2,
  off_target_volume_mm3, off_target_lobe_count, inference_latency_s
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import ndimage

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

PEAK_NORM = math.log(2.0)
MINUS_6DB_NORM = math.log(1.5)


def _argmax_idx(p: np.ndarray) -> np.ndarray:
    return np.array(np.unravel_index(int(np.argmax(p)), p.shape), dtype=np.float32)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)


def _fwhm_mask_around_peak(p_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (focal_mask, peak_idx) where focal_mask is the connected component
    of `p_norm >= 0.5 * p_norm.max()` that contains the global peak voxel.

    p_norm is the un-log-transformed normalized pressure field (in [0, 1]
    when using the paper's per-sample normalization). Using the connected
    component containing the peak excludes side lobes that happen to cross
    half-max, which would otherwise inflate the focal volume.
    """
    peak = float(p_norm.max())
    peak_idx = np.array(np.unravel_index(int(np.argmax(p_norm)), p_norm.shape), dtype=np.int64)
    if peak <= 0.0:
        return np.zeros_like(p_norm, dtype=bool), peak_idx
    mask = p_norm >= 0.5 * peak
    labels, _ = ndimage.label(mask)  # default 6-connectivity in 3D
    peak_label = int(labels[peak_idx[0], peak_idx[1], peak_idx[2]])
    return labels == peak_label, peak_idx


def per_sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_size_mm: float = 0.5,
    off_target_min_dist_mm: float = 10.0,
) -> dict:
    """Compute the per-sample metric dict. pred/target are 3D arrays in the
    paper's normalized log space (peak ~ log(2) = 0.6931).

    Returns a dict containing paper-matched metrics (computed in log space) AND
    TUSNet-spec metrics (computed in expm1-decoded normalized pressure space).
    The two spaces share argmax (log is monotone), so focal_position_error_mm
    is identical regardless of which space you compute it in.
    """
    pred64 = pred.astype(np.float64)
    target64 = target.astype(np.float64)

    # ---- paper-matched (log-normalized space) ----
    diff = pred64 - target64
    rel_l2 = float(np.sqrt((diff ** 2).sum()) / (np.sqrt((target64 ** 2).sum()) + 1e-12))

    pos_pred = _argmax_idx(pred)
    pos_tgt = _argmax_idx(target)
    focal_err_mm = float(np.linalg.norm(pos_pred - pos_tgt) * voxel_size_mm)

    pred_peak = float(pred.max())
    tgt_peak = float(target.max())
    max_p_err = abs(pred_peak - tgt_peak) / (abs(tgt_peak) + 1e-12)

    # ---- TUSNet-spec (expm1-decoded normalized pressure in [0, 1]) ----
    pred_norm = np.expm1(pred64)  # = pred/p_max in [0, 1]
    target_norm = np.expm1(target64)

    # 1) focal_pressure_error: relative error of pred at GT-argmax voxel.
    gt_idx = pos_tgt.astype(np.int64)
    p_pred_at_gt = float(pred_norm[gt_idx[0], gt_idx[1], gt_idx[2]])
    p_gt_at_gt = float(target_norm[gt_idx[0], gt_idx[1], gt_idx[2]])
    focal_pressure_error = abs(p_pred_at_gt - p_gt_at_gt) / (p_gt_at_gt + 1e-8)

    # 2) focal_volume_fwhm: connected component around peak where p_norm >= 0.5 * peak.
    pred_fwhm_mask, _ = _fwhm_mask_around_peak(pred_norm)
    target_fwhm_mask, _ = _fwhm_mask_around_peak(target_norm)
    vox_mm3 = voxel_size_mm ** 3
    focal_volume_pred_mm3 = float(pred_fwhm_mask.sum() * vox_mm3)
    focal_volume_gt_mm3 = float(target_fwhm_mask.sum() * vox_mm3)
    focal_volume_error = abs(focal_volume_pred_mm3 - focal_volume_gt_mm3) / (focal_volume_gt_mm3 + 1e-8)

    # 3) focal_iou_fwhm: IoU of FWHM masks.
    inter = int(np.logical_and(pred_fwhm_mask, target_fwhm_mask).sum())
    union = int(np.logical_or(pred_fwhm_mask, target_fwhm_mask).sum())
    focal_iou_fwhm = float(inter / (union + 1e-8))

    # ---- legacy / supporting metrics (kept for backward compat with earlier eval runs) ----
    mask_pred = pred > MINUS_6DB_NORM
    mask_tgt = target > MINUS_6DB_NORM
    focal_dice = _dice(mask_pred, mask_tgt)

    # off-target hot-spots: regions in mask_pred farther than the radius from
    # the predicted focal point.
    if mask_pred.any():
        zz, yy, xx = np.indices(mask_pred.shape, dtype=np.float32)
        dist_vox = np.sqrt((zz - pos_pred[0]) ** 2 + (yy - pos_pred[1]) ** 2 + (xx - pos_pred[2]) ** 2)
        off_target_mask = mask_pred & (dist_vox * voxel_size_mm > off_target_min_dist_mm)
        off_target_vol_mm3 = float(off_target_mask.sum() * vox_mm3)
        _, n_lobes = ndimage.label(off_target_mask)
    else:
        off_target_vol_mm3 = 0.0
        n_lobes = 0

    return {
        "relative_l2": rel_l2,
        "focal_position_error_mm": focal_err_mm,
        "max_pressure_error": float(max_p_err),
        "focal_pressure_error": float(focal_pressure_error),
        "focal_volume_pred_mm3": focal_volume_pred_mm3,
        "focal_volume_gt_mm3": focal_volume_gt_mm3,
        "focal_volume_error": float(focal_volume_error),
        "focal_iou_fwhm": focal_iou_fwhm,
        "focal_dice": focal_dice,
        "off_target_volume_mm3": off_target_vol_mm3,
        "off_target_lobe_count": int(n_lobes),
        "pred_focal_idx": pos_pred.tolist(),
        "target_focal_idx": pos_tgt.tolist(),
        "pred_peak_norm": pred_peak,
        "target_peak_norm": tgt_peak,
        "voxel_size_mm": float(voxel_size_mm),
    }


def _time_inference(model, ct: torch.Tensor, tr: torch.Tensor, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(ct, tr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    voxel_size_mm_at_native: float = 0.5,
    off_target_min_dist_mm: float = 10.0,
    n_warmup: int = 3,
    save_predictions_dir: Path | None = None,
    per_voxel_error_path: Path | None = None,
) -> dict:
    model.eval()
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = next(model.parameters()).dtype
    # Warm-up
    for i in range(min(n_warmup, len(dataset))):
        b = dataset[i]
        ct = b["ct"].unsqueeze(0).to(device).to(model_dtype)
        tr = b["transducer"].unsqueeze(0).to(device).to(model_dtype)
        _time_inference(model, ct, tr, device)

    per: list[dict] = []
    latencies: list[float] = []
    err_sq_sum: np.ndarray | None = None
    n_for_map = 0
    t_loop_start = time.time()

    for i, item in enumerate(dataset):
        ct = item["ct"].unsqueeze(0).to(device).to(model_dtype)
        tr = item["transducer"].unsqueeze(0).to(device).to(model_dtype)
        target = item["pressure"].squeeze(0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        pred_t = model(ct, tr)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        latency = time.perf_counter() - t0
        pred = pred_t.squeeze(0).squeeze(0).float().cpu().numpy()
        D = pred.shape[-1]
        vmm = voxel_size_mm_at_native * (256.0 / D)
        m = per_sample_metrics(pred, target, voxel_size_mm=vmm,
                               off_target_min_dist_mm=off_target_min_dist_mm)
        m["sample_id"] = item["sample_id"]
        m["inference_latency_s"] = float(latency)
        per.append(m)
        latencies.append(float(latency))

        e2 = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
        if err_sq_sum is None:
            err_sq_sum = e2
        else:
            err_sq_sum = err_sq_sum + e2
        n_for_map += 1

        if save_predictions_dir is not None:
            np.savez_compressed(
                save_predictions_dir / f"{item['sample_id']}.npz",
                pred=pred.astype(np.float32),
                target=target.astype(np.float32),
                ct=item["ct"].squeeze(0).cpu().numpy().astype(np.float32),
                transducer=item["transducer"].cpu().numpy().astype(np.float32),
                p_max_raw=float(item.get("p_max_raw", float("nan"))),
                p_max_resized=float(item.get("p_max_resized", float("nan"))),
            )

        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(dataset):
            elapsed = time.time() - t_loop_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(dataset) - (i + 1)) / max(rate, 1e-6)
            print(f"[eval] {i + 1}/{len(dataset)} samples  "
                  f"rate={rate:.2f}/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                  f"latest: rel_l2={m['relative_l2']:.3f}  "
                  f"focal_mm={m['focal_position_error_mm']:.2f}  "
                  f"IoU={m['focal_iou_fwhm']:.3f}", flush=True)

    if err_sq_sum is not None and per_voxel_error_path is not None:
        per_voxel_error_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(per_voxel_error_path, (err_sq_sum / max(1, n_for_map)).astype(np.float32))

    keys = [
        "relative_l2", "focal_position_error_mm", "max_pressure_error",
        "focal_pressure_error", "focal_volume_pred_mm3", "focal_volume_gt_mm3",
        "focal_volume_error", "focal_iou_fwhm", "focal_dice",
        "off_target_volume_mm3", "off_target_lobe_count",
        "inference_latency_s",
    ]
    agg = {}
    for k in keys:
        v = np.array([d[k] for d in per], dtype=np.float64)
        agg[k] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()),
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
        }
    return {"per_sample": per, "aggregate": agg, "n_samples": len(per)}


def write_per_sample_csv(per: list[dict], path: Path) -> None:
    """Write per-sample metrics to a CSV so downstream analysis
    (histograms, per-placement breakdowns) has a tidy table to load.
    """
    import csv
    columns = [
        "sample_id",
        "focal_position_error_mm", "focal_pressure_error",
        "focal_volume_pred_mm3", "focal_volume_gt_mm3", "focal_volume_error",
        "focal_iou_fwhm", "focal_dice",
        "max_pressure_error", "relative_l2",
        "off_target_volume_mm3", "off_target_lobe_count",
        "inference_latency_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in per:
            w.writerow([r.get(c, "") for c in columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="dir to write metrics.json, per_voxel_mse.npy, and predictions/. "
                         "Defaults to <ckpt-parent>/eval_<split>/")
    ap.add_argument("--no-save-predictions", action="store_true",
                    help="skip saving per-sample npz predictions; ~10x faster eval, "
                         "no 50-100 GB of disk usage. Aggregate metrics still computed.")
    args = ap.parse_args()

    from data import TFUScapesDataset, discover_splits

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    splits = discover_splits()
    files = splits[args.split]
    if not files:
        raise RuntimeError(f"no files in split '{args.split}'. Run download_data.py.")

    ds = TFUScapesDataset(
        files,
        resolution=int(cfg["data"]["resolution"]),
        n_transducer_points=int(cfg["data"]["n_transducer_points"]),
        train=False,
    )

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    from models import build_deeptfus
    model = build_deeptfus(**cfg["model"]).to(device)
    # Match training precision so the saved bf16 weights load cleanly. fp32
    # is the safe default if the older checkpoint pre-dates the precision key.
    precision = str(cfg.get("train", {}).get("precision", "fp32"))
    if precision == "pure-bf16":
        model.to(torch.bfloat16)
    elif precision != "fp32":
        raise ValueError(f"unknown train.precision in checkpoint: {precision!r}")
    model.load_state_dict(ck["model"])

    ck_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ck_path.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_preds = cfg.get("eval", {}).get("save_predictions", True) and not args.no_save_predictions
    pred_dir = out_dir / "predictions" if save_preds else None

    res = evaluate(
        model, ds, device,
        voxel_size_mm_at_native=float(cfg["eval"]["voxel_size_mm"]),
        off_target_min_dist_mm=float(cfg["eval"]["off_target_min_dist_mm"]),
        n_warmup=int(cfg["eval"].get("n_warmup_inferences", 3)),
        save_predictions_dir=pred_dir,
        per_voxel_error_path=out_dir / "per_voxel_mse.npy",
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    write_per_sample_csv(res["per_sample"], out_dir / "per_sample.csv")
    print(f"[eval] n={res['n_samples']}, wrote {out_dir / 'metrics.json'} + per_sample.csv")
    for k, v in res["aggregate"].items():
        print(f"  {k:30s} mean={v['mean']:.4f}  median={v['median']:.4f}  std={v['std']:.4f}")


if __name__ == "__main__":
    main()
