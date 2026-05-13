"""Plot val metric trajectories from one or more training runs.

Reads `train_log.csv` from each run directory and renders a 3-panel
chart of the paper's three test metrics over epochs:

  - relative_l2 (val mean)
  - focal_position_error_mm (val mean)
  - max_pressure_error (val mean)

Usage:

    python scripts/plot_trajectories.py runs/deeptfus runs/deeptfus_ft_a \
        --labels base "ft-A" --out trajectories.png

If --labels is omitted, the basename of each run dir is used.

Saves both a PNG (2x DPI for HiDPI) and an SVG next to the --out path.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PANELS = [
    ("val_rel_l2_mean",   "rel L2",            "Relative L2 error"),
    ("val_focal_mm_mean", "focal err [mm]",    "Focal position error"),
    ("val_max_p_mean",    "peak pressure err", "Peak pressure error"),
]


def read_train_log(run_dir: Path) -> list[dict]:
    """Read `train_log.csv` from a run dir and return a list of per-epoch
    dicts. Skips rows where any required column is non-numeric (handles
    older CSVs gracefully)."""
    csv_path = run_dir / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "epoch":             int(r["epoch"]),
                    "val_rel_l2_mean":   float(r["val_rel_l2_mean"]),
                    "val_focal_mm_mean": float(r["val_focal_mm_mean"]),
                    "val_max_p_mean":    float(r["val_max_p_mean"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        raise RuntimeError(f"no usable rows in {csv_path}")
    return rows


def make_figure(runs: list[tuple[str, list[dict]]]) -> plt.Figure:
    plt.rcParams.update({
        "font.family":       ["Inter", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor":    "#444",
        "axes.linewidth":    0.9,
        "axes.grid":         False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlelocation": "left",
        "axes.titlepad":     8,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "xtick.color":       "#444",
        "ytick.color":       "#444",
        "legend.frameon":    False,
        "legend.fontsize":   11,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    handles, labels = [], []

    for ax, (key, ylabel, title) in zip(axes, PANELS):
        for i, (label, rows) in enumerate(runs):
            xs = [r["epoch"] for r in rows]
            ys = [r[key] for r in rows]
            line, = ax.plot(
                xs, ys, "o-",
                color=cmap(i % 10),
                markersize=4, linewidth=1.8,
                markeredgecolor="white", markeredgewidth=0.5,
                label=label,
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)

    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=min(len(runs), 6),
        frameon=False, fontsize=11,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="one or more run directories containing train_log.csv")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display labels (one per run dir; defaults to dir basename)")
    ap.add_argument("--out", type=Path, default=Path("trajectories.png"),
                    help="output path (PNG); SVG written alongside")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.run_dirs):
        raise SystemExit(f"--labels has {len(args.labels)} items but expected {len(args.run_dirs)}")
    labels = args.labels or [d.name for d in args.run_dirs]

    runs = [(label, read_train_log(d)) for label, d in zip(labels, args.run_dirs)]
    for label, rows in runs:
        print(f"  {label:24s} {len(rows)} epochs from {rows[0]['epoch']} to {rows[-1]['epoch']}")

    fig = make_figure(runs)

    out_png = args.out.with_suffix(".png")
    out_svg = args.out.with_suffix(".svg")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"saved {out_png} and {out_svg}")


if __name__ == "__main__":
    main()
