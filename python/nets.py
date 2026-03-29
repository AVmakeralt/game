"""Trainable neural network models for chess policy/value learning.

Both models are fully implemented PyTorch modules (no stubs) and expose a
common interface:
  - forward(x): returns (policy_logits, value)
  - policy_head output shape: [batch, 4096]
  - value output shape: [batch, 1] in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


POLICY_SIZE = 4096


@dataclass(frozen=True)
class NetConfig:
    input_planes: int = 18
    board_size: int = 8
    policy_size: int = POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class SmallNet(nn.Module):
    """Compact model intended for fast training and experimentation."""

    def __init__(self, cfg: NetConfig = NetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        channels = 64
        trunk_blocks = 4
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(trunk_blocks)])
        flattened = channels * cfg.board_size * cfg.board_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * cfg.board_size * cfg.board_size, cfg.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * cfg.board_size * cfg.board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self._flat_features = flattened

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.trunk(x)
        return self.policy_head(x), self.value_head(x)


class MegaNet(nn.Module):
    """Higher-capacity model with deeper residual trunk for stronger play."""

    def __init__(self, cfg: NetConfig = NetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        channels = 192
        trunk_blocks = 16
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(trunk_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * cfg.board_size * cfg.board_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * cfg.board_size * cfg.board_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.trunk(x)
        return self.policy_head(x), self.value_head(x)
