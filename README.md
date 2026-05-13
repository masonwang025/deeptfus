# deeptfus

An open reproduction of [DeepTFUS](https://arxiv.org/abs/2505.12998), a 3D
neural model that predicts the in-skull pressure field of a transcranial
focused-ultrasound transducer in <1s, replacing a ~30 min
k-Wave physics simulation.

Reproduction details (results, fine-tuning experiments, analysis) are shared in a
**[blog post](https://masonjwang.com/projects/reproducing-deeptfus)**.

Trained checkpoints, configs, and per-sample test eval for all six
models are on HuggingFace:
**[masonwang025/deeptfus-reproduction](https://huggingface.co/collections/masonwang025/deeptfus-reproduction-6a03e39286a09470b960511f)**.

## Contents

- `train.py`, `eval.py` &mdash; entry points
- `configs/base.yaml` &mdash; the paper recipe (the base reproduction run)
- `configs/ft_{a,b,c,d,e}_*.yaml` &mdash; five fine-tune variants from
  the blog post, each a single-line change to the loss
- `data/`, `losses/`, `models/` &mdash; core code
- `scripts/` &mdash; dataset download, env setup, smoke tests, trajectory plots

## Quick start

Built on Python 3.11, PyTorch 2.x, single H100 80GB. Should also fit on
A100 80GB.

```bash
pip install -r requirements.txt
hf auth login --token <your-hf-token>     # for the gated TFUScapes dataset
python scripts/download_data.py --mode full   # ~200 GB, ~10 min via hf-xet
python scripts/local_verify.py                # CPU/MPS pipeline smoke test
```

Or run `bash scripts/setup.sh` to do all of the above.

## Reproduce the base run

50 epochs from scratch on the paper recipe, ~9 hours on a single H100:

```bash
python train.py --config configs/base.yaml
python eval.py --checkpoint runs/deeptfus/ckpt_best.pt --split test
```

Test-set metrics (n = 597) land in `runs/deeptfus/eval_test/`.

## Fine-tune from the base ckpt

Each variant continues training the base ckpt for ~15 epochs with a
single new term added to the loss. Pick one:

```bash
python train.py --config configs/ft_a_softargmax_mild.yaml \
                --resume runs/deeptfus/ckpt_best.pt --finetune
python train.py --config configs/ft_b_softargmax_cranked.yaml \
                --resume runs/deeptfus/ckpt_best.pt --finetune
# ... c, d, e similarly
```

## Eval

The `eval.py` script writes a per-sample test CSV and a 12-metric
aggregate `metrics.json` (paper's three: `relative_l2`,
`focal_position_error_mm`, `max_pressure_error`; plus nine TUSNet shape
and safety metrics: `focal_pressure_error`, `focal_iou_fwhm`,
`focal_dice`, `focal_volume_pred_mm3`, `focal_volume_gt_mm3`,
`focal_volume_error`, `off_target_volume_mm3`, `off_target_lobe_count`,
`inference_latency_s`).

```bash
python eval.py --checkpoint runs/deeptfus/ckpt_best.pt --split test
```

## Plot training trajectories

`scripts/plot_trajectories.py` reads `train_log.csv` from each run
directory and renders a 3-panel chart of the val metrics over epochs:

```bash
python scripts/plot_trajectories.py runs/deeptfus runs/deeptfus_ft_a runs/deeptfus_ft_b \
    --labels base ft-A ft-B --out trajectories.png
```

## Credit

The paper proposes the architecture and releases the dataset; the
weights and training code are not released at the time of writing. See
[arXiv:2505.12998](https://arxiv.org/abs/2505.12998) and the upstream
dataset at
[CAMMA-public/TFUScapes](https://huggingface.co/datasets/vinkle-srivastav/TFUScapes).

```bibtex
@article{srivastav2025deeptfus,
  title  = {A Skull-Adaptive Framework for AI-Based 3D Transcranial
            Focused Ultrasound Simulation},
  author = {Srivastav, Vinkle and others},
  journal= {arXiv preprint arXiv:2505.12998},
  year   = {2025}
}
```

A few of the additional evaluation metrics are adapted from
[TUSNet (Naftchi-Ardebili et al., 2024)](https://arxiv.org/abs/2410.19995).

## License

Code: MIT (see `LICENSE`).

Trained weights distributed on HuggingFace are CC-BY-NC-ND-4.0,
matching the TFUScapes dataset license.
