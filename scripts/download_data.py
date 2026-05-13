"""Download TFUScapes data from HuggingFace.

Two modes:

  --mode full     : download the entire dataset (~200 GB across ~2,493 npz
                    files). Uses snapshot_download.

  --mode subset   : download a small stratified subset (N per split, at most
                    --per-subject files per subject). Useful for local Mac
                    smoke testing.

In both cases the files land under external/hf_cache/. After downloading,
`python -m data.splits` shows the counts your local file tree resolves to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "external" / "hf_cache"
REPO_ID = "vinkle-srivastav/TFUScapes"
SNAPSHOT_DIR = (
    CACHE_DIR
    / "datasets--vinkle-srivastav--TFUScapes"
    / "snapshots"
    / "1c410548e40c491cedd779648257a1c9eaee3587"
)


def _load_split(name: str) -> list[str]:
    p = SNAPSHOT_DIR / "splits" / f"{name}.txt"
    if not p.exists():
        # Trigger a one-file download to materialize the splits dir.
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO_ID, f"splits/{name}.txt", repo_type="dataset", cache_dir=str(CACHE_DIR))
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


def _stratify(items: Iterable[str], n: int, per_subject: int) -> list[str]:
    by_subj: dict[str, list[str]] = {}
    for it in items:
        by_subj.setdefault(it.split("/")[0], []).append(it)
    chosen: list[str] = []
    for subj in sorted(by_subj):
        for it in by_subj[subj][:per_subject]:
            chosen.append(it)
            if len(chosen) >= n:
                return chosen
    return chosen


def download_subset(n_train: int, n_val: int, n_test: int, per_subject: int) -> None:
    from huggingface_hub import hf_hub_download
    plan = {
        "train": _stratify(_load_split("train"), n_train, per_subject),
        "val": _stratify(_load_split("val"), n_val, per_subject),
        "test": _stratify(_load_split("test"), n_test, per_subject),
    }
    total = sum(len(v) for v in plan.values())
    print(f"[download] subset plan: {len(plan['train'])} train, {len(plan['val'])} val, {len(plan['test'])} test ({total} files)", flush=True)
    done = 0
    for split, items in plan.items():
        for rel in items:
            hf_hub_download(REPO_ID, f"data/{rel}", repo_type="dataset", cache_dir=str(CACHE_DIR))
            done += 1
            print(f"[download] {done}/{total} {rel}", flush=True)
    print("[download] subset done.")


def download_full() -> None:
    from huggingface_hub import snapshot_download
    print(f"[download] snapshot_download of {REPO_ID} (~200 GB). This will take a while.", flush=True)
    snapshot_download(REPO_ID, repo_type="dataset", cache_dir=str(CACHE_DIR))
    print("[download] full done.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "subset"], default="subset")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--per-subject", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "full":
        download_full()
    else:
        download_subset(args.n_train, args.n_val, args.n_test, args.per_subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
