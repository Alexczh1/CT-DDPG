"""Small time-conditioned networks used by CT-DDPG."""

import numpy as np
import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        input_dim = hidden_dim
    return nn.Sequential(*modules, nn.Linear(input_dim, output_dim))


def with_time(inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    time_features = torch.cat(
        [torch.cos(2 * torch.pi * time), torch.sin(2 * torch.pi * time)], dim=-1
    )
    return torch.cat([inputs, time_features], dim=-1)


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("scale", (high - low) / 2)
        self.register_buffer("bias", (high + low) / 2)
        self.net = mlp(state_dim + 2, low.numel(), hidden_dim, layers)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(with_time(state, time))) * self.scale + self.bias


class Value(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.net = mlp(state_dim + 2, 1, hidden_dim, layers)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.net(with_time(state, time))


class QRate(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, layers: int
    ) -> None:
        super().__init__()
        self.net = mlp(state_dim + action_dim + 2, 1, hidden_dim, layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return self.net(with_time(torch.cat([state, action], dim=-1), time))
