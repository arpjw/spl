"""Two hero plots for the sparse parity replication.

    results/sweep_plot.png
        Eval loss vs cumulative samples across the four sweep runs, log-x,
        with n^k = 64,000 marked as a reference scale. Shows the four
        observed trajectories, one seed at each budget.

    results/fourier_staircase.png
        All 2^k = 8 subset correlations of the network output with the
        Walsh-Hadamard basis of S = [20, 24, 32], over the 30k-50k drop
        window at fine resolution. Eval loss overlaid on a secondary axis
        so the "correlations depart from noise while loss is still near 1"
        story is readable at a glance.

Both plots read directly from the CSVs written by train.py and fourier.py.
No parameters, no external state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path("results")
REFERENCE_SAMPLES = 40 ** 3  # n^k for the sweep, n=40, k=3.


def load_log(path: Path) -> tuple[list[int], list[float]]:
    """Read (cumulative_samples, eval_loss) pairs from a training log CSV,
    dropping the pre-training (samples=0) row so log-x is clean."""
    xs: list[int] = []
    ys: list[float] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            cs = int(r["cumulative_samples"])
            el = r["eval_loss"]
            if cs == 0 or el == "":
                continue
            xs.append(cs)
            ys.append(float(el))
    return xs, ys


def load_fourier(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_sweep(output_path: Path) -> None:
    runs = [
        # (path, label, color, linestyle, linewidth)
        (RESULTS / "sweep_below" / "log.csv",
         "16k samples (0.25x reference): no learning in this run",
         "#d62728", "-", 1.8),
        (RESULTS / "sweep_near" / "log.csv",
         "64k samples (1.0x reference)",
         "#ff7f0e", "-", 1.8),
        (RESULTS / "sweep_above" / "log.csv",
         "250k samples (3.9x reference)",
         "#1f77b4", "-", 1.8),
        (RESULTS / "sweep_way_above" / "log.csv",
         "500k samples (7.8x reference)",
         "#2ca02c", "--", 1.4),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for path, label, color, ls, lw in runs:
        xs, ys = load_log(path)
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axvline(
        REFERENCE_SAMPLES, color="black", linestyle=":", linewidth=1.0, alpha=0.7,
        label=f"n^k reference = {REFERENCE_SAMPLES:,}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("cumulative samples seen  (log scale)")
    ax.set_ylabel("eval loss   (MSE against +/- 1 targets)")
    ax.set_title(
        "Sparse parity sample-complexity sweep\n"
        "n=40, k=3, 1 hidden layer x 64 units, plain SGD lr=0.1 momentum=0"
    )
    ax.set_ylim(-0.05, 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_fourier(output_path: Path, window: tuple[int, int] = (30000, 50000)) -> None:
    fourier_rows = load_fourier(RESULTS / "sweep_above" / "fourier.csv")
    log_xs, log_ys = load_log(RESULTS / "sweep_above" / "log.csv")

    lo, hi = window
    # Only take fine-cadence rows for a smoother curve through the drop.
    rows = [r for r in fourier_rows if lo <= int(r["cumulative_samples"]) <= hi]
    xs = [int(r["cumulative_samples"]) for r in rows]

    # Subsets colored by degree. Within a degree we use shades of the same
    # hue so the reader can see "these three are similar" at a glance while
    # still being able to trace individual curves.
    subsets = [
        # (csv_column, legend_label, color, linestyle, linewidth, zorder)
        ("corr_empty",       "empty ()",       "#888888", "--", 1.0, 1),
        ("corr_20",          "{20}",           "#9ecae1", "-",  1.4, 2),
        ("corr_24",          "{24}",           "#4292c6", "-",  1.4, 2),
        ("corr_32",          "{32}",           "#08519c", "-",  1.4, 2),
        ("corr_20_24",       "{20, 24}",       "#fdae6b", "-",  1.7, 3),
        ("corr_20_32",       "{20, 32}",       "#e6550d", "-",  1.7, 3),
        ("corr_24_32",       "{24, 32}",       "#a63603", "-",  1.7, 3),
        ("corr_20_24_32",    "{20, 24, 32}",   "#c81b1b", "-",  2.6, 4),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for col, label, color, ls, lw, z in subsets:
        ys = [float(r[col]) for r in rows]
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw,
                zorder=z)

    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.25, 1.1)
    ax.set_xlabel("cumulative samples seen")
    ax.set_ylabel("correlation with subset basis chi_T(x)  (left axis)")
    ax.set_title(
        "Fourier subset correlations across the sparse-parity drop window\n"
        "n=40, k=3, S = [20, 24, 32].  Fine cadence: one correlation estimate "
        "per 500 samples, batch=20,000."
    )
    ax.grid(True, alpha=0.25)

    # Overlay eval loss on secondary axis so the reader can see the
    # correlations depart from noise while the loss curve is still ~1.
    ax2 = ax.twinx()
    loss_pts = [(x, y) for x, y in zip(log_xs, log_ys) if lo <= x <= hi]
    ax2.plot(
        [x for x, _ in loss_pts], [y for _, y in loss_pts],
        color="black", linestyle=":", linewidth=1.8, alpha=0.8,
        label="eval loss (right axis)",
    )
    ax2.set_ylabel("eval loss  (right axis)")
    ax2.set_ylim(-0.05, 1.2)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Legend goes outside the axes on the right so it never covers the curves.
    ax.legend(
        handles1 + handles2, labels1 + labels2,
        loc="center left", bbox_to_anchor=(1.11, 0.5), fontsize=9,
        framealpha=0.95, title="subset  T", title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    sweep_out = RESULTS / "sweep_plot.png"
    fourier_out = RESULTS / "fourier_staircase.png"
    plot_sweep(sweep_out)
    print(f"[plot] wrote {sweep_out}")
    plot_fourier(fourier_out)
    print(f"[plot] wrote {fourier_out}")


if __name__ == "__main__":
    main()
