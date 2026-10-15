"""Fourier subset-correlation tracker for the sparse-parity replication.

For each subset T of the secret set S (there are 2^k of them, so 8 for k=3),
we compute the empirical Walsh-Hadamard correlation
    corr_T = E[f(x) * chi_T(x)],   chi_T(x) = product of x_i for i in T
under uniform inputs x in {-1, +1}^n. chi_{} = 1 identically, so corr_{}
is the network's mean output. chi_S(x) = y by definition of the parity task,
so corr_S is the correlation of the model with the true target.

Design decisions:
  1. This script re-runs the training loop of a given config with identical
     seeds. Model init, optimizer, and train_rng are all seeded the same as
     train.py, so the trajectory reproduces bit-for-bit. We are not adding
     an inline callback to train.py because we want fourier tracking to be
     opt-in for the one representative run, not paid every training run.
  2. Fourier eval batches come from a separate fourier_rng, not train_rng
     and not eval_rng. This keeps the correlation estimates unbiased (never
     reusing training samples) and keeps the reproduced trajectory identical
     to the original sweep_above run.
  3. Checkpoint cadence is coarse across the full run and fine through the
     observed drop window, so if low-degree subsets rise before the full-S
     subset that separation is not blurred by too-sparse sampling.
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from data import make_secret_subset, sample_batch
from model import ParityMLP
from train import build_optimizer, load_config


def enumerate_subsets(S: list[int]) -> list[tuple[int, ...]]:
    """All 2^k subsets of S, ordered by size then lexicographically.

    That ordering makes the CSV header easy to read: empty, then singletons,
    then pairs, then the full triple.
    """
    subs: list[tuple[int, ...]] = [()]
    for r in range(1, len(S) + 1):
        for c in combinations(sorted(S), r):
            subs.append(tuple(c))
    return subs


def subset_column_name(T: tuple[int, ...]) -> str:
    if not T:
        return "corr_empty"
    return "corr_" + "_".join(str(i) for i in T)


@torch.no_grad()
def compute_correlations(
    model: torch.nn.Module,
    subsets: list[tuple[int, ...]],
    x: torch.Tensor,
) -> list[float]:
    """One forward pass, then reuse f for every subset's chi_T product."""
    model.eval()
    f = model(x)
    out: list[float] = []
    for T in subsets:
        if len(T) == 0:
            chi = torch.ones_like(f)
        else:
            chi = x[:, list(T)].prod(dim=1)
        out.append((f * chi).mean().item())
    model.train()
    return out


def build_target_samples(
    total_samples: int,
    coarse: int,
    fine_window: tuple[int, int],
    fine: int,
) -> list[int]:
    """Union of a coarse grid across the whole run and a fine grid in the
    drop window. Deduped and sorted."""
    coarse_pts = list(range(0, total_samples + 1, coarse))
    lo, hi = fine_window
    hi = min(hi, total_samples)
    fine_pts = list(range(lo, hi + 1, fine)) if lo <= hi else []
    return sorted(set(coarse_pts + fine_pts + [total_samples]))


def track(
    cfg: dict,
    coarse_samples: int,
    fine_window: tuple[int, int],
    fine_samples: int,
    fourier_seed: int,
    fourier_batch_size: int,
) -> None:
    run_name = cfg["run_name"]
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(cfg["n"])
    k = int(cfg["k"])
    S = make_secret_subset(n, k, int(cfg["S_seed"]))
    print(f"[fourier] n={n}, k={k}, S={S}")
    if "expected_S" in cfg and cfg["expected_S"] is not None:
        assert S == sorted(cfg["expected_S"]), (
            f"config expected_S {cfg['expected_S']} does not match actual S {S}"
        )
    if cfg.get("fixed_pool_size"):
        raise ValueError(
            "Fourier tracker requires strict online mode. Refusing to run "
            "against a fixed-pool config: the story is about online SGD "
            "learning trajectories, not multi-pass dynamics."
        )

    subsets = enumerate_subsets(S)
    print(f"[fourier] tracking {len(subsets)} subsets:")
    for T in subsets:
        print(f"  |T|={len(T)}: {T if T else '(empty set)'}")

    batch_size = int(cfg["batch_size"])
    sample_budget = int(cfg["sample_budget"])
    if sample_budget <= 0 or batch_size <= 0:
        raise ValueError("sample_budget and batch_size must be positive")
    total_steps = (sample_budget + batch_size - 1) // batch_size

    torch.manual_seed(int(cfg["model_seed"]))
    device = torch.device(cfg.get("device", "cpu"))
    model = ParityMLP(n=n, hidden=int(cfg["hidden"])).to(device)
    optimizer = build_optimizer(model, cfg)
    print(
        f"[fourier] model + optimizer built with same seeds as train.py so "
        f"the reproduced trajectory should match {run_name}/log.csv"
    )

    train_rng = np.random.default_rng(int(cfg["train_data_seed"]))
    fourier_rng = np.random.default_rng(fourier_seed)

    targets = build_target_samples(
        sample_budget, coarse_samples, fine_window, fine_samples
    )
    print(
        f"[fourier] {len(targets)} checkpoints, coarse every {coarse_samples} "
        f"samples, fine every {fine_samples} samples in "
        f"[{fine_window[0]}, {fine_window[1]}]"
    )

    fourier_csv = out_dir / "fourier.csv"
    header = ["step", "cumulative_samples"] + [subset_column_name(T) for T in subsets]

    def log_checkpoint(writer, step: int, cum_samples: int) -> list[float]:
        x, _ = sample_batch(fourier_rng, n, S, fourier_batch_size)
        x = x.to(device)
        corrs = compute_correlations(model, subsets, x)
        writer.writerow([step, cum_samples] + [f"{c:.6f}" for c in corrs])
        return corrs

    with open(fourier_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        next_i = 0
        # Initial checkpoint at samples=0, before any updates.
        if targets and targets[0] == 0:
            log_checkpoint(writer, 0, 0)
            next_i = 1
        f.flush()

        t0 = time.time()
        for step in range(1, total_steps + 1):
            cum = min(step * batch_size, sample_budget)
            current_batch_size = cum - (step - 1) * batch_size
            x_t, y_t = sample_batch(train_rng, n, S, current_batch_size)
            x_t = x_t.to(device)
            y_t = y_t.to(device)

            pred = model(x_t)
            train_l = ((pred - y_t) ** 2).mean()
            optimizer.zero_grad()
            train_l.backward()
            optimizer.step()

            while next_i < len(targets) and cum >= targets[next_i]:
                log_checkpoint(writer, step, cum)
                f.flush()
                next_i += 1

        elapsed = time.time() - t0
        print(f"[fourier] done in {elapsed:.1f}s, wrote {fourier_csv}")

    print_departure_summary(fourier_csv, subsets, threshold_low=0.05, threshold_high=0.5)


def print_departure_summary(
    fourier_csv: Path,
    subsets: list[tuple[int, ...]],
    threshold_low: float,
    threshold_high: float,
) -> None:
    """Post-hoc: read the CSV and print, per subset, the first sample count
    at which |corr| exceeds a small threshold (departure from zero) and the
    first sample count at which it exceeds a large threshold (clear rise).
    """
    rows = []
    with open(fourier_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print()
    print("[fourier] subset departure summary "
          f"(|corr| > {threshold_low} = departure, "
          f"|corr| > {threshold_high} = clear rise):")
    print(f"  {'subset':<20} {'depart @ samples':>18} {'rise @ samples':>16} {'final corr':>12}")
    for T in subsets:
        col = subset_column_name(T)
        depart_at: int | None = None
        rise_at: int | None = None
        final_val = float("nan")
        for r in rows:
            v = float(r[col])
            cum = int(r["cumulative_samples"])
            if depart_at is None and abs(v) > threshold_low:
                depart_at = cum
            if rise_at is None and abs(v) > threshold_high:
                rise_at = cum
            final_val = v
        label = "(empty set)" if not T else str(T)
        d_str = f"{depart_at:,}" if depart_at is not None else "never"
        r_str = f"{rise_at:,}" if rise_at is not None else "never"
        print(f"  {label:<20} {d_str:>18} {r_str:>16} {final_val:>12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--coarse", type=int, default=2500,
                        help="coarse cadence in samples")
    parser.add_argument("--fine", type=int, default=500,
                        help="fine cadence in samples within the drop window")
    parser.add_argument("--fine-lo", type=int, default=32000,
                        help="drop-window lower bound in samples")
    parser.add_argument("--fine-hi", type=int, default=64000,
                        help="drop-window upper bound in samples")
    parser.add_argument("--fourier-seed", type=int, default=3,
                        help="RNG seed for the fourier-eval batches, "
                             "kept separate from train/eval seeds")
    parser.add_argument("--fourier-batch-size", type=int, default=20000,
                        help="batch size for each correlation estimate")
    args = parser.parse_args()
    cfg = load_config(args.config)
    track(
        cfg,
        coarse_samples=args.coarse,
        fine_window=(args.fine_lo, args.fine_hi),
        fine_samples=args.fine,
        fourier_seed=args.fourier_seed,
        fourier_batch_size=args.fourier_batch_size,
    )


if __name__ == "__main__":
    main()
