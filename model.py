"""Small MLP for k-sparse parity, following the Barak et al. (2022) setup.

One hidden layer, ReLU, scalar output. Deliberately not a transformer: the
claim being replicated is about plain feedforward SGD dynamics.
"""

from __future__ import annotations

import torch
from torch import nn


class ParityMLP(nn.Module):
    def __init__(self, n: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        # Squeeze the output so it lines up with a [B]-shaped +/- 1 target.
        return self.fc2(h).squeeze(-1)
