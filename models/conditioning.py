"""Transducer-aware conditioning modules used by DeepTFUS.

Paper Section 3.2 specifies three conditioning mechanisms, all driven by an
embedding of the transducer point cloud:

  1. Fourier positional encoding -> per-point MLP -> attention pool -> z_T
     (paper Eq 3 + the paragraph after: "MLP layers, followed by layer
      normalization and ReLU activations to produce point-wise features z_i",
      then attention-weighted sum z_T = Σ_i α_i z_i).
  2. FiLM modulation in the **decoding path** (paper §3.2 second mechanism).
  3. Dynamic depthwise 3D convolutions on the encoder, kernel weights
     predicted by an MLP from z_T.
  4. Cross-attention at each encoder level, exchanging spatial U-Net tokens
     with the transducer embedding.

The paper does not pin down the dynamic-conv kernel size, the number of
attention heads, which encoder levels carry cross-attention, or the
Fourier-encoding frequency count. Values threaded through `configs/base.yaml`
are documented as TENTATIVE; everything else here follows the paper text.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_positional_encoding(pts: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Sinusoidal positional encoding. pts: (B, N, 3) -> (B, N, 3 + 6*n_freqs)."""
    B, N, D = pts.shape
    assert D == 3, "transducer points must be (B, N, 3)"
    freqs = (2.0 ** torch.arange(n_freqs, device=pts.device, dtype=pts.dtype)) * math.pi
    freqs = freqs.view(1, 1, 1, n_freqs)
    proj = pts.unsqueeze(-1) * freqs                              # (B, N, 3, n_freqs)
    sin = torch.sin(proj).reshape(B, N, 3 * n_freqs)
    cos = torch.cos(proj).reshape(B, N, 3 * n_freqs)
    return torch.cat([pts, sin, cos], dim=-1)


class TransducerEncoder(nn.Module):
    """Point cloud -> single global transducer vector z_T (paper §3.2).

    Fourier-encoded coords concatenated with the raw coords, then MLP layers
    each followed by LayerNorm + ReLU, producing per-point features z_i. A
    learned vector w defines softmax attention weights α_i = softmax(w · z_i),
    and z_T = Σ_i α_i z_i.

    Returns z_T only. Per-point features z_i are an internal computation
    detail; all three downstream conditioning paths (FiLM, dynamic conv,
    cross-attention) take z_T as their conditioning input per paper §3.2.
    """

    def __init__(self, n_freqs: int = 8, hidden: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        in_dim = 3 + 6 * n_freqs
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.attn_query = nn.Parameter(torch.randn(out_dim) / math.sqrt(out_dim))
        self.out_dim = out_dim

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        feat = fourier_positional_encoding(pts, self.n_freqs)     # (B, N, in_dim)
        z = self.mlp(feat)                                        # (B, N, d) per-point z_i
        scores = z @ self.attn_query                              # (B, N) = w · z_i
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        z_T = (z * weights).sum(dim=1)                            # (B, d)
        return z_T


class FiLM3d(nn.Module):
    """out = gamma(z) * x + beta(z) on a 5D feature map."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(z).view(z.size(0), -1, 1, 1, 1)
        beta = self.to_beta(z).view(z.size(0), -1, 1, 1, 1)
        return gamma * x + beta


class DynamicConv3d(nn.Module):
    """Depthwise 3D conv with a per-sample kernel predicted from z_T.

    The kernel weights for a (C, 1, k, k, k) depthwise filter are produced
    by an MLP over z_T. Implemented via grouped conv (groups = B * C) by
    folding the batch into the channel axis. Initialized so the layer is
    identity at the start of training (zero weight prediction; bias is a
    delta-function kernel).
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.k = int(kernel_size)
        n = self.channels * self.k ** 3
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, n),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        center = (self.k // 2) * (self.k * self.k) + (self.k // 2) * self.k + (self.k // 2)
        bias = torch.zeros(n)
        for c in range(self.channels):
            bias[c * self.k ** 3 + center] = 1.0
        with torch.no_grad():
            self.mlp[-1].bias.copy_(bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert C == self.channels, f"DynamicConv3d expected {self.channels} channels, got {C}"
        w = self.mlp(z).view(B * C, 1, self.k, self.k, self.k)
        pad = self.k // 2
        y = F.conv3d(x.reshape(1, B * C, D, H, W), w, groups=B * C, padding=pad)
        return y.reshape(B, C, D, H, W)


class CrossAttention3d(nn.Module):
    """Cross-attention + concat+1x1x1 fusion (paper §3.2 third conditioning).

    Two attention blocks, the second one gated by `bidirectional`:

      Block A (always on):  Q = z_T,        K, V = CT_flat   -> updated z_T
      Block B (optional):   Q = CT_flat,    K, V = z_T       -> updated CT_flat

    z_T is the single global transducer token (B, cond_dim), per the paper's
    "the transducer embedding z_T is used as a global token". With z_T as a
    single token, block B's attention is effectively a learned broadcast of a
    projection of z_T to every CT position: functionally a fancier FiLM that
    overlaps the encoder's `DynamicConv3d` conditioning. Our default run sets
    `bidirectional=False` and skips block B for that reason; the paper-faithful
    bi-directional design is recovered with `bidirectional=True`.

    Fusion: the CT half of the concat is either the block-B-updated CT
    (bi-directional) or the original input CT (direction-2 only). The other
    half is the broadcast of the updated z_T. The fusion conv is
    zero-initialized so the residual connection makes the layer identity at
    start of training.

    Memory budget: dominated by K_CT, V_CT at the level's resolution (each
    is one (B, C, T) tensor) plus a (B, 2C, T) concat tensor for fusion.
    Block B adds another (B, C, T) Q_CT projection and a (B, C, T) attention
    output, roughly doubling the per-level activation budget when enabled.
    Cheap at levels 1, 2, 3, bottleneck on an 80GB H100 at bw=16 b=4; level
    0 fits only with `bidirectional=False`.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        n_heads: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0, "channels must divide n_heads"
        self.n_heads = int(n_heads)
        self.channels = int(channels)
        self.bidirectional = bool(bidirectional)

        # Block A: z_T queries, CT keys/values.
        self.q_A = nn.Linear(cond_dim, channels)
        self.k_A = nn.Conv3d(channels, channels, 1)
        self.v_A = nn.Conv3d(channels, channels, 1)
        self.out_A = nn.Linear(channels, channels)

        # Block B: CT queries, z_T keys/values. Allocated only when enabled
        # so the unused-direction-1 path uses no extra parameters or memory.
        if self.bidirectional:
            self.q_B = nn.Conv3d(channels, channels, 1)
            self.k_B = nn.Linear(cond_dim, channels)
            self.v_B = nn.Linear(cond_dim, channels)
            self.out_B = nn.Conv3d(channels, channels, 1)
        else:
            self.q_B = None
            self.k_B = None
            self.v_B = None
            self.out_B = None

        self.fuse = nn.Conv3d(2 * channels, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor, z_T: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        T = D * H * W
        h = self.n_heads
        c_h = C // h

        # ---- Block A: z_T (Q) attends over CT (K, V). ----
        q_zt = self.q_A(z_T).view(B, 1, h, c_h).transpose(1, 2)            # (B, h, 1, c_h)
        k_ct = self.k_A(x).flatten(2).transpose(1, 2).reshape(B, T, h, c_h).transpose(1, 2)
        v_ct = self.v_A(x).flatten(2).transpose(1, 2).reshape(B, T, h, c_h).transpose(1, 2)
        zt_attn = F.scaled_dot_product_attention(q_zt, k_ct, v_ct)         # (B, h, 1, c_h)
        zt_attn = zt_attn.transpose(1, 2).reshape(B, C)
        updated_zT = self.out_A(zt_attn)                                   # (B, C)

        # ---- Block B (optional): CT (Q) attends over z_T (single K, V). ----
        if self.bidirectional:
            q_ct = self.q_B(x).flatten(2).transpose(1, 2).reshape(B, T, h, c_h).transpose(1, 2)
            k_zt = self.k_B(z_T).view(B, 1, h, c_h).transpose(1, 2)
            v_zt = self.v_B(z_T).view(B, 1, h, c_h).transpose(1, 2)
            ct_attn = F.scaled_dot_product_attention(q_ct, k_zt, v_zt)     # (B, h, T, c_h)
            ct_attn = ct_attn.transpose(1, 2).reshape(B, T, C).transpose(1, 2).reshape(B, C, D, H, W)
            ct_for_fusion = self.out_B(ct_attn)                            # (B, C, D, H, W)
        else:
            ct_for_fusion = x

        # ---- Fusion: concat(ct_for_fusion, broadcast(updated_z_T)) -> 1x1x1 conv. ----
        zT_broadcast = updated_zT.view(B, C, 1, 1, 1).expand(B, C, D, H, W)
        fused = torch.cat([ct_for_fusion, zT_broadcast], dim=1)            # (B, 2C, D, H, W)
        return x + self.fuse(fused)
