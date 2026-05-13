"""Train/val/test split derived from the upstream TFUScapes split manifests.

The HuggingFace dataset ships `splits/{train,val,test}.txt` with the paper's
85 / 10 / 30 subject partition. We honor that exactly. discover_splits()
returns the absolute paths of the npz files that actually exist locally
under external/hf_cache. Files referenced in the split manifests but not
yet downloaded are silently omitted.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = REPO_ROOT / "external" / "hf_cache"
SNAPSHOT_DIR = (
    HF_CACHE
    / "datasets--vinkle-srivastav--TFUScapes"
    / "snapshots"
    / "1c410548e40c491cedd779648257a1c9eaee3587"
)
# In-tree sample npz (a single train file, distributed with the upstream repo)
SAMPLE_FILE = REPO_ROOT / "external" / "TFUScapes" / "sample" / "TFUScapes" / "data" / "A00060925" / "exp_0.npz"


def _load_split_names(name: str) -> set[str]:
    p = SNAPSHOT_DIR / "splits" / f"{name}.txt"
    if not p.exists():
        return set()
    with open(p) as f:
        return {line.strip() for line in f if line.strip()}


def discover_splits() -> dict[str, list[str]]:
    """Return {train, val, test} -> list[abs npz path]. Subject-disjoint per paper."""
    train_names = _load_split_names("train")
    val_names = _load_split_names("val")
    test_names = _load_split_names("test")

    data_root = SNAPSHOT_DIR / "data"
    files: list[Path] = sorted(data_root.rglob("*.npz")) if data_root.exists() else []
    if SAMPLE_FILE.exists():
        files.append(SAMPLE_FILE)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for p in files:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        parts = p.parts
        if "data" not in parts:
            continue
        idx = parts.index("data")
        try:
            key = f"{parts[idx + 1]}/{parts[idx + 2]}"
        except IndexError:
            continue
        if key in train_names:
            splits["train"].append(s)
        elif key in val_names:
            splits["val"].append(s)
        elif key in test_names:
            splits["test"].append(s)
    return splits


if __name__ == "__main__":
    s = discover_splits()
    for k, v in s.items():
        print(f"{k}: {len(v)} files")
