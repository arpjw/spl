"""Online data source for k-sparse parity.

We never build a fixed dataset. Every training step and every eval step draws a
fresh batch straight from the true distribution. Reusing a small pool would let
the model memorize and would defeat the sample-complexity experiment.
"""

from __future__ import annotations

import numpy as np
import torch


def make_secret_subset(n: int, k: int, seed: int) -> list[int]:
    """Pick the secret subset S of size k out of n coordinates, once, from a seed.

    Returned sorted so it is trivially comparable across runs.
    """
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


def sample_batch(
    rng: np.random.Generator,
    n: int,
    S: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a fresh batch of (x, y) from the true k-sparse parity distribution.

    x in {-1, +1}^n uniform. y is the product of x_i over i in S, so y in {-1, +1}.
    Returned as float32 torch tensors so they slot into the model without a cast.
    """
    x = rng.integers(0, 2, size=(batch_size, n), dtype=np.int8)
    x = x.astype(np.float32) * 2.0 - 1.0
    y = np.prod(x[:, S], axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":
    # Quick self-check: report the actual S for seed 0 so we can bake it into
    # the config without relying on a hand-derived value.
    n, k, seed = 30, 3, 0
    S = make_secret_subset(n, k, seed)
    print(f"n={n}, k={k}, seed={seed}")
    print(f"secret subset S = {S}")

    rng = np.random.default_rng(seed + 1)
    x, y = sample_batch(rng, n, S, batch_size=8)
    print(f"batch x shape: {tuple(x.shape)}, batch y shape: {tuple(y.shape)}")
    print(f"y values (should all be +/- 1): {y.tolist()}")
    # Sanity: recompute y from x[:, S] and confirm it matches.
    y_check = x[:, S].prod(dim=1)
    assert torch.allclose(y, y_check), "y does not match product over S"
    print("self-check passed: y equals product of x over S")
