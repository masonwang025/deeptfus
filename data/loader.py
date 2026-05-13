"""TFUScapes dataset loader.

Each .npz contains:
  ct:        (256, 256, 256) float, pseudo-CT in HU-like units
  pmap:      (256, 256, 256) float, k-Wave steady-state pressure (Pa)
  tr_coords: (~17k, 3) int, transducer-surface points in [0, 255] voxel indices

Normalization (paper Section 4):
  CT:       linearly mapped to [0, 1] from [0, 2000] HU.
  Pressure: per-sample p_norm = log(1 + p / p.max()), so peak is log(2) ~= 0.693.
  Coords:   rescaled from [0, 255] to [-1, 1].

The loader trilinearly resizes ct and pmap to `resolution`^3. The transducer
coords are in resolution-independent normalized coordinates, so the same
network conditioning works at any spatial resolution.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CT_MIN = 0.0
CT_MAX = 2000.0
EPS = 1e-12


def _trilinear_resize(vol: np.ndarray, out_size: int) -> np.ndarray:
    t = torch.from_numpy(vol).float()[None, None]
    t = F.interpolate(t, size=(out_size, out_size, out_size), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


class TFUScapesDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        resolution: int = 256,
        n_transducer_points: int = 512,
        train: bool = True,
    ) -> None:
        if len(files) == 0:
            raise ValueError("TFUScapesDataset: empty file list")
        self.files = list(files)
        self.resolution = int(resolution)
        self.n_pts = int(n_transducer_points)
        self.train = bool(train)
        self.ids = [self._derive_id(p) for p in self.files]

    @staticmethod
    def _derive_id(path: str) -> str:
        p = Path(path)
        return f"{p.parent.name}_{p.stem}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with np.load(path) as data:
            ct = np.asarray(data["ct"], dtype=np.float32)
            pmap = np.asarray(data["pmap"], dtype=np.float32)
            tr = np.asarray(data["tr_coords"], dtype=np.int64)

        D = self.resolution
        ct_r = _trilinear_resize(ct, D)
        pmap_r = _trilinear_resize(pmap, D)

        ct_r = np.clip((ct_r - CT_MIN) / (CT_MAX - CT_MIN), 0.0, 1.5).astype(np.float32)

        # Paper normalization: p_norm = log(1 + p / p.max()).
        # We resize BEFORE normalizing so the per-sample max is computed on the
        # downsampled volume that the network actually sees and predicts. The
        # raw max is also retained for any downstream absolute-Pa work.
        p_max_raw = float(max(pmap.max(), EPS))
        p_max_resized = float(max(pmap_r.max(), EPS))
        pmap_norm = np.log1p(np.clip(pmap_r, 0.0, None) / p_max_resized).astype(np.float32)

        # Transducer coords in [-1, 1] at the original 256 grid.
        N = tr.shape[0]
        if self.train:
            sel = np.random.choice(N, size=min(self.n_pts, N), replace=N < self.n_pts)
        else:
            stride = max(1, N // self.n_pts)
            sel = np.arange(0, N, stride)[: self.n_pts]
            if len(sel) < self.n_pts:
                sel = np.concatenate([sel, np.zeros(self.n_pts - len(sel), dtype=np.int64)])
        tr_norm = (tr[sel].astype(np.float32) / 255.0) * 2.0 - 1.0

        return {
            "ct": torch.from_numpy(ct_r)[None],
            "transducer": torch.from_numpy(tr_norm),
            "pressure": torch.from_numpy(pmap_norm)[None],
            "p_max_resized": torch.tensor(p_max_resized, dtype=torch.float32),
            "p_max_raw": torch.tensor(p_max_raw, dtype=torch.float32),
            "sample_id": self.ids[idx],
        }


# Constant: in the paper's normalization, -6 dB (factor 0.5 in amplitude) lands
# at log(1.5) in log-normalized space, while the peak is log(2). Useful for
# focal-volume Dice and off-target hot-spot detection.
PEAK_NORM = math.log(2.0)
MINUS_6DB_NORM = math.log(1.5)
