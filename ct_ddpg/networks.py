"""Time-conditioned networks used by CT-DDPG."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        previous_dim = input_dim
        for _ in range(layers):
            modules.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        modules.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def append_periodic_time(
    inputs: torch.Tensor, normalized_time: torch.Tensor
) -> torch.Tensor:
    """Append cos(2 pi t) and sin(2 pi t) to the final dimension."""
    time_features = torch.cat(
        (
            torch.cos(2.0 * torch.pi * normalized_time),
            torch.sin(2.0 * torch.pi * normalized_time),
        ),
        dim=-1,
    )
    return torch.cat((inputs, time_features.to(inputs.device)), dim=-1)


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 400,
        layers: int = 2,
        time_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.time_embedding = time_embedding
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer(
            "action_scale", (action_high_tensor - action_low_tensor) / 2.0
        )
        self.register_buffer(
            "action_bias", (action_high_tensor + action_low_tensor) / 2.0
        )
        input_dim = state_dim + (2 if time_embedding else 0)
        self.net = MLP(input_dim, action_low_tensor.numel(), hidden_dim, layers)

    def forward(
        self, states: torch.Tensor, normalized_time: torch.Tensor
    ) -> torch.Tensor:
        inputs = (
            append_periodic_time(states, normalized_time)
            if self.time_embedding
            else states
        )
        return torch.tanh(self.net(inputs)) * self.action_scale + self.action_bias


class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 400,
        layers: int = 2,
        time_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.time_embedding = time_embedding
        input_dim = state_dim + (2 if time_embedding else 0)
        self.net = MLP(input_dim, 1, hidden_dim, layers)

    def forward(
        self, states: torch.Tensor, normalized_time: torch.Tensor
    ) -> torch.Tensor:
        inputs = (
            append_periodic_time(states, normalized_time)
            if self.time_embedding
            else states
        )
        return self.net(inputs)


class QRateNetwork(nn.Module):
    """Continuous-time advantage-rate network q(s, a, t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 400,
        layers: int = 2,
        time_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.time_embedding = time_embedding
        input_dim = state_dim + action_dim + (2 if time_embedding else 0)
        self.net = MLP(input_dim, 1, hidden_dim, layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        normalized_time: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((states, actions), dim=-1)
        if self.time_embedding:
            inputs = append_periodic_time(inputs, normalized_time)
        return self.net(inputs)
