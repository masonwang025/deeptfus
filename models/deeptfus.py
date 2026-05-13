"""DeepTFUS: 3D U-Net + dynamic conv (encoder) + cross-attn + (optional) FiLM.

Architecture per paper §3.2 and Figure 2, with two ablation knobs exposed in
`configs/base.yaml`:

  Encoder per stage: (Conv → GroupNorm → GELU → DynamicConv) × 2.
    The diagram alternates regular Conv with Dynamic Conv; each Dynamic Conv
    re-filters the previous Conv's output through a transducer-aware kernel
    predicted from z_T. FiLM is NOT applied here; paper §3.2 puts FiLM in
    the decoder.

  Decoder per stage: upsample → concat skip → (Conv → GroupNorm → GELU) × 2,
    optionally followed by FiLM. `use_film_decoder` (default `False`)
    controls the FiLM modulation. Off by default because the paper's own
    Table 1 ablation ("No FiLM") shows FiLM does not help.

  Cross-attention: optional per encoder level (`cross_attention_levels`).
    The paper-default would be every encoder level + bi-directional + concat
    fusion. We ship level1..bottleneck (level0 is memory-prohibitive at 256^3
    on a single 80GB H100) and `cross_attention_bidirectional=False`
    (direction 1 collapses to a learned broadcast when z_T is a single
    token, overlapping the encoder's DynamicConv).

  Output: 1×1×1 conv with zero-init weights and biases.

Items the paper omits (base_width, depth, dynamic-conv kernel size,
attention head count, levels) are config-driven and flagged TENTATIVE
in `configs/base.yaml`. The two ablation knobs are documented inline there.
Flip both to `True` to recover the paper-faithful "all three
conditioning mechanisms" architecture.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning import CrossAttention3d, DynamicConv3d, FiLM3d, TransducerEncoder


def _gn_groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ConvDynBlock(nn.Module):
    """Conv → GroupNorm → GELU → DynamicConv. One 'Conv | Dynamic Conv' pair
    from the paper's Figure 2. The DynamicConv kernel is generated from z_T,
    so this is the encoder's transducer-conditioning point.
    """

    def __init__(self, in_c: int, out_c: int, cond_dim: int, dyn_k: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(_gn_groups(out_c), out_c)
        self.act = nn.GELU()
        self.dyn = DynamicConv3d(out_c, cond_dim, dyn_k)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.dyn(self.act(self.norm(self.conv(x))), z)


class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU. Plain conv unit used in the decoder, paired
    with a single FiLM modulation at the end of each decoder stage.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(_gn_groups(out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class EncoderStage(nn.Module):
    """Two (Conv → GN → GELU → DynamicConv) sub-blocks, then optional cross-attn.

    No FiLM in the encoder (paper §3.2 puts FiLM in the decoding path only).
    Cross-attention, when enabled, takes the global transducer token z (paper:
    "z_T is used as a global token"), not per-point features.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        cond_dim: int,
        dyn_conv_k: int,
        use_ca: bool,
        n_heads: int,
        ca_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.b1 = ConvDynBlock(in_c, out_c, cond_dim, dyn_conv_k)
        self.b2 = ConvDynBlock(out_c, out_c, cond_dim, dyn_conv_k)
        if use_ca:
            heads = n_heads
            while out_c % heads != 0 and heads > 1:
                heads //= 2
            self.ca = CrossAttention3d(out_c, cond_dim, heads, bidirectional=ca_bidirectional)
        else:
            self.ca = None

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.b1(x, z)
        h = self.b2(h, z)
        if self.ca is not None:
            h = self.ca(h, z)
        return h


class DecoderStage(nn.Module):
    """Two (Conv → GN → GELU) sub-blocks, optionally followed by FiLM.

    Paper §3.2 puts FiLM in the decoding path. We gate it on `use_film` and
    default to off because the paper's own Table 1 ablation ("No FiLM") shows
    FiLM slightly *worsens* every metric except max_pressure_error; the
    conditioning paths we keep (encoder DynamicConv + bottleneck CrossAttn)
    already broadcast transducer state into the spatial features.
    """

    def __init__(self, in_c: int, out_c: int, cond_dim: int, use_film: bool = False) -> None:
        super().__init__()
        self.b1 = ConvBlock(in_c, out_c)
        self.b2 = ConvBlock(out_c, out_c)
        self.film = FiLM3d(cond_dim, out_c) if use_film else None

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.b2(self.b1(x))
        if self.film is not None:
            h = self.film(h, z)
        return h


CA_LEVEL_NAMES = ("level1", "level2", "level3", "bottleneck")


class DeepTFUS(nn.Module):
    """Paper architecture (Section 3.2). 4-level U-Net, base_width configurable."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 32,
        cond_dim: int = 128,
        n_transducer_freqs: int = 8,
        dynamic_conv_kernel: int = 3,
        cross_attention_heads: int = 4,
        cross_attention_levels: Iterable[str] = ("bottleneck",),
        cross_attention_bidirectional: bool = False,
        use_film_decoder: bool = False,
    ) -> None:
        super().__init__()
        ca_set = set(cross_attention_levels)
        unknown = ca_set - set(CA_LEVEL_NAMES)
        if unknown:
            raise ValueError(f"cross_attention_levels unknown: {unknown}. Valid: {CA_LEVEL_NAMES}")

        self.encoder = TransducerEncoder(
            n_freqs=n_transducer_freqs, hidden=cond_dim, out_dim=cond_dim
        )

        c1, c2, c3, c4 = base_width, base_width * 2, base_width * 4, base_width * 8

        def _enc(in_c: int, out_c: int, level: str) -> EncoderStage:
            return EncoderStage(
                in_c, out_c, cond_dim,
                dyn_conv_k=dynamic_conv_kernel,
                use_ca=(level in ca_set),
                n_heads=cross_attention_heads,
                ca_bidirectional=cross_attention_bidirectional,
            )

        self.down1 = _enc(in_channels, c1, "level1")
        self.down2 = _enc(c1, c2, "level2")
        self.down3 = _enc(c2, c3, "level3")
        self.bottleneck = _enc(c3, c4, "bottleneck")
        self.up3 = DecoderStage(c4 + c3, c3, cond_dim, use_film=use_film_decoder)
        self.up2 = DecoderStage(c3 + c2, c2, cond_dim, use_film=use_film_decoder)
        self.up1 = DecoderStage(c2 + c1, c1, cond_dim, use_film=use_film_decoder)
        self.out = nn.Conv3d(c1, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.pool = nn.AvgPool3d(2)

    def forward(self, ct: torch.Tensor, transducer: torch.Tensor) -> torch.Tensor:
        z = self.encoder(transducer)
        x1 = self.down1(ct, z)
        x2 = self.down2(self.pool(x1), z)
        x3 = self.down3(self.pool(x2), z)
        x4 = self.bottleneck(self.pool(x3), z)
        u3 = F.interpolate(x4, scale_factor=2, mode="trilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, x3], dim=1), z)
        u2 = F.interpolate(u3, scale_factor=2, mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1), z)
        u1 = F.interpolate(u2, scale_factor=2, mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1), z)
        return self.out(u1)


def build_deeptfus(**kwargs) -> DeepTFUS:
    return DeepTFUS(**kwargs)
