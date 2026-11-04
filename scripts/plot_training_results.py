"""Plot per-seed training curves and means over epochs shared by every seed.

Usage:
    python scripts/plot_training_results.py
    python scripts/plot_training_results.py --seeds 42 43 44 --output-dir outputs/plots
    python scripts/plot_training_results.py --csv path/seed42.csv path/seed43.csv path/seed44.csv

Individual curves include all recorded epochs. Means use only epochs present in
every input CSV, so early stopping never changes the number of averaged seeds.
Loss and mAP are saved side by side in training_results.png.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs/logs/baseline/csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "report/baseline/analysis"
METRICS = ("train_loss", "val_loss", "map")


def load_results(path: Path) -> dict[int, dict[str, float]]:
    """Read and validate metrics, indexed by epoch rather than CSV row order."""
    results = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"epoch", *METRICS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                epoch = int(row["epoch"])
                values = {metric: float(row[metric]) for metric in METRICS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid epoch or metric") from exc
            if epoch < 1 or epoch in results:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate epoch {epoch}")
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}:{line_number}: non-finite metric")
            results[epoch] = values
    if not results:
        raise ValueError(f"{path}: no training records")
    return results


def plot_curves(ax, runs, common_epochs, metrics, title, ylabel):
    """Draw thin transparent runs underneath opaque mean curves."""
    for metric, label, color in metrics:
        for run in runs.values():
            epochs = sorted(run)
            ax.plot(
                epochs, [run[epoch][metric] for epoch in epochs],
                color=color, alpha=0.25, linewidth=1.2, zorder=1,
            )
        averages = [
            mean(run[epoch][metric] for run in runs.values())
            for epoch in common_epochs
        ]
        ax.plot(
            common_epochs, averages, color=color, linewidth=2.6,
            marker="o" if len(common_epochs) == 1 else None,
            label=f"{label} (mean of {len(runs)} seeds)", zorder=2,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="gray", alpha=0.25, linewidth=1.2))
    labels.append("Individual seeds (same color as each mean)")
    ax.legend(handles, labels)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.set_ylim(bottom=0)
    if ylabel == "mAP":
        ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.text(
        0.5, -0.14,
        "Mean: epochs shared by all input CSVs only.",
        transform=ax.transAxes, ha="center", fontsize=9, color="dimgray",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing training_results_seed<N>.csv",
    )
    inputs.add_argument(
        "--csv", type=Path, nargs="+", metavar="PATH",
        help="Explicit CSV paths; overrides seed-based file selection",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for training_results.png (loss and mAP)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    try:
        paths = args.csv or [
            args.input_dir / f"training_results_seed{seed}.csv" for seed in args.seeds
        ]
        paths = [path.expanduser().resolve() for path in paths]
        if len(set(paths)) != len(paths):
            raise ValueError("Input CSV paths must not contain duplicates")
        runs = {path: load_results(path) for path in paths}
        common_epochs = sorted(set.intersection(*(set(run) for run in runs.values())))
        if not common_epochs:
            raise ValueError("No epochs are shared by all input CSVs")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    plot_curves(
        axes[0], runs, common_epochs,
        [("train_loss", "Train loss", "tab:blue"),
         ("val_loss", "Validation loss", "tab:orange")],
        "Training and validation loss", "Loss",
    )
    plot_curves(
        axes[1], runs, common_epochs, [("map", "mAP", "tab:green")],
        "Validation mAP", "mAP",
    )
    output_path = args.output_dir / "training_results.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Mean uses {len(common_epochs)} shared epochs across {len(runs)} seeds.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
