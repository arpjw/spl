"""Online SGD trainer for k-sparse parity.

Key invariants enforced here (see CLAUDE.md gotchas):
  1. Fresh batch every step in the default (online) mode. No fixed dataset.
  2. Plain SGD via torch.optim.SGD. Never Adam or any adaptive optimizer.
  3. Primary logging axis is cumulative samples seen, not step count.
  4. Eval batches are always drawn fresh from the true distribution, even in
     fixed-pool mode, so eval loss stays an unbiased test-error estimate.

Fixed-pool mode is a deliberate departure from invariant (1), used for a
separate multi-pass comparison. It samples repeatedly from a fixed pool of
rows. The fixed-pool and short online runs have different data and training
budgets, so they do not isolate the effect of repeated access.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from data import make_secret_subset, sample_batch
from model import ParityMLP


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build the optimizer from config, but refuse anything that is not plain SGD.

    The whole replication is about vanilla SGD dynamics against the SQ bound,
    so a config that quietly asked for Adam would silently invalidate the
    experiment. Fail loudly instead.
    """
    name = cfg.get("optimizer", "sgd").lower()
    if name != "sgd":
        raise ValueError(
            f"optimizer must be 'sgd' for this experiment, got {name!r}. "
            "Adaptive optimizers change the dynamics and muddy the story."
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg.get("momentum", 0.0)),
        weight_decay=0.0,
    )


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    rng: np.random.Generator,
    n: int,
    S: list[int],
    batch_size: int,
    device: torch.device,
) -> float:
    """MSE against +/- 1 targets on a freshly drawn batch."""
    model.eval()
    x, y = sample_batch(rng, n, S, batch_size)
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    loss = ((pred - y) ** 2).mean().item()
    model.train()
    return loss


def train(cfg: dict) -> None:
    run_name = cfg["run_name"]
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task.
    n = int(cfg["n"])
    k = int(cfg["k"])
    S = make_secret_subset(n, k, int(cfg["S_seed"]))
    print(f"[setup] n={n}, k={k}, S_seed={cfg['S_seed']}, S={S}")
    if "expected_S" in cfg and cfg["expected_S"] is not None:
        expected = sorted(cfg["expected_S"])
        assert S == expected, (
            f"actual S {S} does not match expected_S {expected} from config. "
            "Update the config or the seed rather than silently drifting."
        )
        print(f"[setup] expected_S matches actual S: OK")

    # Budget arithmetic.
    batch_size = int(cfg["batch_size"])
    sample_budget = int(cfg["sample_budget"])
    total_steps = sample_budget // batch_size
    reference_samples = n ** k
    print(
        f"[setup] sample_budget={sample_budget}, batch_size={batch_size}, "
        f"total_steps={total_steps}"
    )
    print(
        f"[setup] n^k reference={reference_samples}, "
        f"budget/reference={sample_budget / reference_samples:.2f}x"
    )

    # Model + optimizer.
    torch.manual_seed(int(cfg["model_seed"]))
    device = torch.device(cfg.get("device", "cpu"))
    model = ParityMLP(n=n, hidden=int(cfg["hidden"])).to(device)
    optimizer = build_optimizer(model, cfg)
    print(
        f"[setup] optimizer={type(optimizer).__name__}, "
        f"lr={cfg['lr']}, momentum={cfg.get('momentum', 0.0)}, "
        f"hidden={cfg['hidden']}, device={device}"
    )

    # Independent RNGs so train and eval never see the same samples.
    train_rng = np.random.default_rng(int(cfg["train_data_seed"]))
    eval_rng = np.random.default_rng(int(cfg["eval_data_seed"]))

    eval_every = int(cfg["eval_every_steps"])
    eval_batch_size = int(cfg["eval_batch_size"])

    # Data source: online (default) or fixed-pool (separate multi-pass run).
    fixed_pool_size = int(cfg.get("fixed_pool_size") or 0)
    if fixed_pool_size > 0:
        pool_x, pool_y = sample_batch(train_rng, n, S, fixed_pool_size)
        pool_x = pool_x.to(device)
        pool_y = pool_y.to(device)
        print(
            f"[setup] data_mode=fixed_pool, unique_samples={fixed_pool_size} "
            f"({fixed_pool_size / reference_samples:.2f}x n^k), "
            f"total sample_views over run={sample_budget} "
            f"({sample_budget / fixed_pool_size:.1f} pool-size-equivalent draws)"
        )

        def get_train_batch() -> tuple[torch.Tensor, torch.Tensor]:
            idx = train_rng.integers(0, fixed_pool_size, size=batch_size)
            return pool_x[idx], pool_y[idx]
    else:
        print("[setup] data_mode=online")

        def get_train_batch() -> tuple[torch.Tensor, torch.Tensor]:
            x, y = sample_batch(train_rng, n, S, batch_size)
            return x.to(device), y.to(device)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cumulative_samples", "train_loss", "eval_loss"])

        # Log the initial state (step 0) before any updates.
        init_eval = eval_loss(model, eval_rng, n, S, eval_batch_size, device)
        writer.writerow([0, 0, "", f"{init_eval:.6f}"])
        f.flush()
        print(f"[eval] step=0 samples=0 eval_loss={init_eval:.4f}")

        t0 = time.time()
        for step in range(1, total_steps + 1):
            x, y = get_train_batch()

            pred = model(x)
            train_l = ((pred - y) ** 2).mean()

            optimizer.zero_grad()
            train_l.backward()
            optimizer.step()

            cumulative_samples = step * batch_size

            if step % eval_every == 0 or step == total_steps:
                el = eval_loss(model, eval_rng, n, S, eval_batch_size, device)
                writer.writerow(
                    [step, cumulative_samples, f"{train_l.item():.6f}", f"{el:.6f}"]
                )
                f.flush()
                # Print sparingly to keep stdout readable.
                if step % (eval_every * 10) == 0 or step == total_steps:
                    elapsed = time.time() - t0
                    print(
                        f"[eval] step={step} samples={cumulative_samples} "
                        f"train_loss={train_l.item():.4f} eval_loss={el:.4f} "
                        f"({elapsed:.1f}s elapsed)"
                    )

    elapsed = time.time() - t0
    print(f"[done] wrote {log_path} in {elapsed:.1f}s "
          f"({total_steps} steps, {total_steps * batch_size} samples)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
