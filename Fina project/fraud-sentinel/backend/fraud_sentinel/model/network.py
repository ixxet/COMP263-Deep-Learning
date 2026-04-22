"""PyTorch model definitions."""

from __future__ import annotations

import torch
from torch import nn


class DenseFraudClassifier(nn.Module):
    def __init__(self, input_dim: int = 30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 30, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

