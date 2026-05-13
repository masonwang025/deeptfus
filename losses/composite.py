"""DeepTFUS composite loss.

Implements the paper's two-term composite (spatially-weighted MSE +
gradient-L1) and two optional fine-tune additions used by the variants
A through E in the configs:

  - soft-argmax focal-position L1 term (variants A, B, C, D, E)
  - soft 3D Dice on the half-max iso-volume    (variant E)

Paper §3.3 / Eq 4-5:

    L_weighted = (1/|Omega|) Sum_v  w(v) * (pred(v) - target(v))^2
    w(v)       = exp(alpha (P(v) - max_v' P(v'))) / E_v[exp(alpha (P(v) - max_v' P(v')))]

A continuous exponential weight on the target pressure, normalized
per-sample so that mean(w) = 1. With alpha = 5 and per-sample
log-normalization (peak ~ log(2) ~ 0.69), the weight runs from roughly
0.3 in background voxels to ~10x at the focal peak. Without it, the
model would learn to drive whole-volume MSE to zero by predicting
near-zero pressure everywhere (the focal spot is <1% of the voxels).

    L_grad = (1/3) Sum_i  || grad_i pred - grad_i target ||_2^2          (Eq 6)

For numerical stability we use mean-of-absolute-differences over the
spatial finite-difference axes (equivalent up to a constant scaling).
Paper's lambda = 0.1 weighting is set in `configs/base.yaml`.

Soft-argmax focal-position L1 term (disabled by default):

    soft_argmax(P_hat_norm, tau) = Sum_v softmax(P_hat_norm(v) / tau) * v
    L_focal = || soft_argmax(P_hat_norm, tau) - argmax(P_gt_norm) ||_1

`P_hat_norm = expm1(P_hat_log)` is the un-log-transformed normalized
pressure in [0, 1]. The temperature tau is on this scale; smaller tau
gives a sharper softmax (closer to true argmax) but vanishing gradient
away from the peak. The model receives a gradient saying "move mass
from wherever the soft-argmax currently is toward argmax(P_gt)" which
the weighted-MSE term does not provide.

Schedule: lambda_focal warms up linearly from 0 to `focal_weight` over
`focal_warmup_epochs`, after an optional `focal_warmup_off`-epoch hold
at zero. Cold-starting the focal term on a fresh model can collapse
predictions to a sharp spike in a random location; the warmup avoids
that.

Soft 3D Dice on the half-max iso-volume (disabled by default):

    m(p) = sigmoid((p - threshold) / tau_dice)
    L_dice = 1 - Dice(m(P_hat_norm), m(P_gt_norm))

A differentiable approximation of the focal-volume IoU/Dice eval
metrics. Pulls the predicted half-max region toward the ground-truth
half-max region in shape, not just position. Independent mechanism
from soft-argmax: soft-argmax moves the centroid; Dice constrains the
lobe outline.

The whole loss is computed in fp32 regardless of the dtype the model
trains in. The exponential weight + per-sample 16.8M-element reduction
is sensitive to bf16 precision loss, and the fp32 cast is cheap (one
per step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gd = x[..., 1:, :, :] - x[..., :-1, :, :]
    gh = x[..., :, 1:, :] - x[..., :, :-1, :]
    gw = x[..., :, :, 1:] - x[..., :, :, :-1]
    return gd, gh, gw


def soft_argmax_3d(p_norm: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable 3D argmax via temperature-controlled softmax expectation.

    Args:
        p_norm: (B, D, H, W) un-log-transformed normalized pressure in [0, 1].
        temperature: scalar tau. Smaller tau gives a sharper peak in the
                     softmax, closer to true argmax; larger tau gives a
                     softer support, biased toward the volume centroid.

    Returns:
        (B, 3) tensor of (z, y, x) voxel coordinates.
    """
    B, D, H, W = p_norm.shape
    p_flat = p_norm.reshape(B, -1)
    weights = F.softmax(p_flat / temperature, dim=-1)
    dd, hh, ww = torch.meshgrid(
        torch.arange(D, device=p_norm.device, dtype=p_norm.dtype),
        torch.arange(H, device=p_norm.device, dtype=p_norm.dtype),
        torch.arange(W, device=p_norm.device, dtype=p_norm.dtype),
        indexing="ij",
    )
    coords = torch.stack([dd, hh, ww], dim=-1).reshape(-1, 3)
    return weights @ coords


def _hard_argmax_3d(p: torch.Tensor) -> torch.Tensor:
    """Non-differentiable argmax over the spatial dims of a (B, D, H, W) tensor.

    Returns (B, 3) coordinates. argmax(log(1+x)) = argmax(x) since log1p
    is monotone, so this can be called on either the log-space target or
    the expm1-decoded one with the same answer.
    """
    B, D, H, W = p.shape
    flat = p.reshape(B, -1)
    idx = flat.argmax(dim=-1)
    z = idx // (H * W)
    rem = idx % (H * W)
    y = rem // W
    x = rem % W
    return torch.stack([z, y, x], dim=-1).to(p.dtype)


class DeepTFUSLoss(nn.Module):
    """Paper §3.3 composite loss + optional soft-argmax + soft Dice terms.

    Computes everything in fp32 internally.

    Args:
        alpha: paper Eq 5 exponent. Default 5.0.
        grad_weight: paper lambda for gradient-consistency. Default 0.1.
        focal_weight: target lambda for the soft-argmax focal-position L1
                      term after warmup. Default 0.0 (term disabled).
        focal_temperature: softmax temperature on the expm1-decoded
                           pressure field. Smaller is sharper.
        focal_warmup_off: number of epochs at the start where lambda_focal
                          is exactly 0 (let the field shape stabilize).
        focal_warmup_epochs: number of epochs over which lambda_focal
                             ramps linearly from 0 to `focal_weight`.
        dice_weight: lambda for the soft Dice term. Default 0.0 (disabled).
        dice_threshold: threshold on the normalized pressure (0.5 = -6 dB).
        dice_temperature: sigmoid soft-mask sharpness for the Dice mask.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        grad_weight: float = 0.1,
        focal_weight: float = 0.0,
        focal_temperature: float = 0.1,
        focal_warmup_off: int = 3,
        focal_warmup_epochs: int = 10,
        dice_weight: float = 0.0,
        dice_threshold: float = 0.5,
        dice_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.grad_weight = float(grad_weight)
        self.focal_weight = float(focal_weight)
        self.focal_temperature = float(focal_temperature)
        self.focal_warmup_off = int(focal_warmup_off)
        self.focal_warmup_epochs = int(focal_warmup_epochs)
        self.dice_weight = float(dice_weight)
        self.dice_threshold = float(dice_threshold)
        self.dice_temperature = float(dice_temperature)
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Called from the training loop at the start of each epoch.
        Drives the linear warmup schedule of lambda_focal."""
        self._current_epoch = int(epoch)

    def lambda_focal_now(self) -> float:
        if self.focal_weight <= 0:
            return 0.0
        ep = self._current_epoch
        if ep < self.focal_warmup_off:
            return 0.0
        ramp_start = self.focal_warmup_off
        ramp_end = self.focal_warmup_off + max(1, self.focal_warmup_epochs)
        if ep >= ramp_end:
            return self.focal_weight
        return self.focal_weight * (ep - ramp_start) / max(1, self.focal_warmup_epochs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pred_f = pred.float()
        target_f = target.float()
        if pred_f.dim() == 5 and pred_f.size(1) == 1:
            pred_5d = pred_f
            target_5d = target_f
        else:
            pred_5d = pred_f.unsqueeze(1) if pred_f.dim() == 4 else pred_f
            target_5d = target_f.unsqueeze(1) if target_f.dim() == 4 else target_f
        B = pred_5d.size(0)

        flat_t = target_5d.reshape(B, -1)
        t_max = flat_t.max(dim=1).values.view(B, 1, 1, 1, 1)
        delta = self.alpha * (target_5d - t_max)
        w_unnorm = torch.exp(delta)
        Z = w_unnorm.reshape(B, -1).mean(dim=1).view(B, 1, 1, 1, 1).clamp_min(1e-8)
        w = w_unnorm / Z
        wmse = (w * (pred_5d - target_5d).pow(2)).mean()

        pd, ph, pw = _grad(pred_5d)
        td, th, tw = _grad(target_5d)
        gl1 = (pd - td).abs().mean() + (ph - th).abs().mean() + (pw - tw).abs().mean()

        total = wmse + self.grad_weight * gl1
        parts = {
            "weighted_mse": float(wmse.detach().item()),
            "gradient_l1": float(gl1.detach().item()),
            "focal_l1_vox": 0.0,
            "lambda_focal": 0.0,
            "soft_dice_fwhm": 0.0,
            "lambda_dice": 0.0,
        }

        lam = self.lambda_focal_now()
        if lam > 0.0:
            # expm1-decode to normalized pressure in [0, 1]. log1p+expm1
            # inverts cleanly; argmax of the decoded pressure equals argmax
            # of the log-space target (log1p is monotone), so the target
            # argmax can be taken on either tensor.
            pred_norm = torch.expm1(pred_5d.squeeze(1).clamp_max(50.0))
            sa = soft_argmax_3d(pred_norm, temperature=self.focal_temperature)  # (B, 3)
            tgt_idx = _hard_argmax_3d(target_5d.squeeze(1))                     # (B, 3)
            focal_l1 = (sa - tgt_idx).abs().sum(dim=-1).mean()                  # mean per-sample L1
            total = total + lam * focal_l1
            parts["focal_l1_vox"] = float(focal_l1.detach().item())
            parts["lambda_focal"] = float(lam)

        if self.dice_weight > 0.0:
            # Soft 3D Dice on the normalized-pressure half-max iso-volumes.
            # Soft binary mask via sigmoid for differentiability:
            #   m(p) = sigmoid((p - threshold) / tau)
            # Loss is (1 - Dice) so the optimizer pulls the predicted FWHM
            # region toward GT FWHM in shape, not just position.
            pred_norm = torch.expm1(pred_5d.squeeze(1).clamp_max(50.0))
            target_norm = torch.expm1(target_5d.squeeze(1).clamp_max(50.0))
            thr = self.dice_threshold
            tau = max(self.dice_temperature, 1e-6)
            pred_mask = torch.sigmoid((pred_norm - thr) / tau)
            target_mask = torch.sigmoid((target_norm - thr) / tau)
            inter = (pred_mask * target_mask).flatten(1).sum(dim=-1)
            denom = pred_mask.flatten(1).sum(dim=-1) + target_mask.flatten(1).sum(dim=-1)
            dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
            l_dice = (1.0 - dice).mean()
            total = total + self.dice_weight * l_dice
            parts["soft_dice_fwhm"] = float(dice.mean().detach().item())
            parts["lambda_dice"] = float(self.dice_weight)

        return total, parts
