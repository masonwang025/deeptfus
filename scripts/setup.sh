#!/usr/bin/env bash
# Bootstrap a fresh GPU instance. Idempotent; safe to re-run.
#
# Steps:
#   1. Install Python deps from requirements.txt
#   2. Confirm CUDA is visible to torch
#   3. Download the full TFUScapes dataset (~200 GB) into external/hf_cache
#   4. Run a CPU/MPS pipeline smoke test
#
# Many cloud GPU instances wipe disk on shutdown. The HuggingFace cache
# is content-addressed; re-running this on a fresh disk just resumes
# the download from where the prior instance left off.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] python deps"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "[setup] torch / cuda check"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device 0: {p.name}  mem={p.total_memory/1e9:.1f} GB  cc={p.major}.{p.minor}")
else:
    print("  WARN: torch.cuda.is_available() == False.")
    print("  Reinstall with a CUDA wheel if needed:")
    print("    pip uninstall -y torch")
    print("    pip install --index-url https://download.pytorch.org/whl/cu124 torch")
PY

echo "[setup] dataset (full, ~200 GB; resumes via HF cache if partial)"
python scripts/download_data.py --mode full

echo "[setup] pipeline smoke test"
python scripts/local_verify.py

echo "[setup] done. To train the base run:"
echo "    python train.py --config configs/base.yaml"
echo "  or fine-tune from the base ckpt:"
echo "    python train.py --config configs/ft_a_softargmax_mild.yaml \\"
echo "                    --resume runs/deeptfus/ckpt_best.pt --finetune"
