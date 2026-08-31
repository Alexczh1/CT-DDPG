"""Minimal implementation of CT-DDPG.

The critic is learned from short trajectory sequences through a continuous-time
martingale residual. The actor maximizes the learned advantage-rate network.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import nn

from .buffer import EpisodeSequenceBuffer, SequenceBatch, TerminalBatch
from .networks import DeterministicPolicy, QRateNetwork, ValueNetwork


@dataclass(frozen=True)
class NetworkConfig:
    hidden_dim: int = 400
    layers: int = 2
    time_embedding: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.layers < 0:
            raise ValueError("hidden_dim must be positive and layers non-negative")


@dataclass(frozen=True)
class CTDDPGConfig:
    dt: float = 0.05
    discount_factor: float = 0.8
    buffer_size: int = 10_000
    batch_size: int = 256
    sequence_length: int = 2
    max_sequence_length: int = 10
    max_policy_sequence_length: int = 10
    warmup_episodes: int = 5
    update_frequency: int = 1
    gradient_steps: int = 1
    critic_steps: int = 1
    exploration_noise: float = 0.1
    policy_learning_rate: float = 3.0e-4
    value_learning_rate: float = 3.0e-4
    q_learning_rate: float = 3.0e-4
    policy_weight_decay: float = 0.0
    value_weight_decay: float = 0.0
    q_weight_decay: float = 0.0
    lr_decay_step: int = 80_000
    lr_decay_gamma: float = 0.8
    tau: float = 0.005
    constraint_weight: float = 0.001
    q_constraint: bool = False
    policy_network: NetworkConfig = field(default_factory=NetworkConfig)
    value_network: NetworkConfig = field(default_factory=NetworkConfig)
    q_network: NetworkConfig = field(default_factory=NetworkConfig)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if self.buffer_size <= 0 or self.batch_size <= 0:
            raise ValueError("buffer_size and batch_size must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.max_sequence_length < self.sequence_length:
            raise ValueError(
                "max_sequence_length cannot be shorter than sequence_length"
            )
        if self.max_policy_sequence_length < 1:
            raise ValueError("max_policy_sequence_length must be positive")
        if self.warmup_episodes <= 0:
            raise ValueError("warmup_episodes must be positive")
        if self.update_frequency <= 0 or self.gradient_steps <= 0:
            raise ValueError("update_frequency and gradient_steps must be positive")
        if self.critic_steps <= 0:
            raise ValueError("critic_steps must be positive")
        if self.exploration_noise < 0:
            raise ValueError("exploration_noise cannot be negative")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.constraint_weight < 0:
            raise ValueError("constraint_weight cannot be negative")


class CTDDPG:
    """Continuous-time deterministic actor-critic with online sequence replay."""

    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        device: torch.device | str,
        config: CTDDPGConfig,
        seed: int = 0,
        output_dir: str | Path | None = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.device = torch.device(device)
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._eval_seeded = False

        observation_space = env.single_observation_space
        action_space = env.single_action_space
        if len(observation_space.shape) != 1 or len(action_space.shape) != 1:
            raise ValueError("CT-DDPG expects flat Box observation and action spaces")

        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)

        self.policy = DeterministicPolicy(
            state_dim,
            action_low,
            action_high,
            **asdict(config.policy_network),
        ).to(self.device)
        self.value = ValueNetwork(state_dim, **asdict(config.value_network)).to(
            self.device
        )
        self.value_target = ValueNetwork(state_dim, **asdict(config.value_network)).to(
            self.device
        )
        self.value_target.load_state_dict(self.value.state_dict())
        self.q_rate = QRateNetwork(
            state_dim, action_dim, **asdict(config.q_network)
        ).to(self.device)

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.policy_learning_rate,
            weight_decay=config.policy_weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(),
            lr=config.value_learning_rate,
            weight_decay=config.value_weight_decay,
        )
        self.q_optimizer = torch.optim.AdamW(
            self.q_rate.parameters(),
            lr=config.q_learning_rate,
            weight_decay=config.q_weight_decay,
        )

        scheduler_options = {
            "step_size": config.lr_decay_step,
            "gamma": config.lr_decay_gamma,
        }
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, **scheduler_options
        )
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(
            self.value_optimizer, **scheduler_options
        )
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q_optimizer, **scheduler_options
        )

    @torch.no_grad()
    def select_action(
        self,
        states: np.ndarray,
        normalized_time: np.ndarray,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        time_tensor = torch.as_tensor(
            normalized_time, dtype=torch.float32, device=self.device
        )
        actions = self.policy(state_tensor, time_tensor).cpu().numpy()
        if noise_scale:
            actions = actions + noise_scale * self.rng.normal(size=actions.shape)
        return np.clip(
            actions,
            self.env.single_action_space.low,
            self.env.single_action_space.high,
        ).astype(np.float32)

    def update_critic(
        self,
        sequence_batch: SequenceBatch,
        terminal_batch: TerminalBatch,
    ) -> dict[str, float]:
        states = self._tensor(sequence_batch.states)
        actions = self._tensor(sequence_batch.actions)
        rewards = self._tensor(sequence_batch.rewards)
        times = self._tensor(sequence_batch.times)

        q_values = self.q_rate(states, actions, times).squeeze(-1)
        if self.config.q_constraint:
            centered_q = q_values
        else:
            with torch.no_grad():
                policy_actions = self.policy(states, times)
            policy_q = self.q_rate(states, policy_actions, times).squeeze(-1)
            centered_q = q_values - policy_q

        interval_count = rewards.shape[1] - 1
        discount_powers = self.config.discount_factor ** (
            self.config.dt
            * torch.arange(interval_count, dtype=torch.float32, device=self.device)
        )
        integrated_return = torch.sum(
            (rewards[:, :-1] - centered_q[:, :-1]) * self.config.dt * discount_powers,
            dim=-1,
        )

        with torch.no_grad():
            target_value = self.value_target(states[:, -1], times[:, -1]).squeeze(-1)
            target_value = target_value * self.config.discount_factor ** (
                self.config.dt * interval_count
            )
        current_value = self.value(states[:, 0], times[:, 0]).squeeze(-1)
        martingale_loss = nn.functional.mse_loss(
            current_value, integrated_return + target_value
        )

        if self.config.q_constraint:
            with torch.no_grad():
                policy_actions = self.policy(states, times)
            q_constraint_loss = (
                self.q_rate(states, policy_actions, times).squeeze(-1).square().mean()
            )
        else:
            q_constraint_loss = torch.zeros((), device=self.device)

        terminal_states = self._tensor(terminal_batch.states)
        terminal_rewards = self._tensor(terminal_batch.rewards)
        terminal_times = self._tensor(terminal_batch.times)
        terminal_values = self.value(terminal_states, terminal_times).squeeze(-1)
        terminal_loss = nn.functional.mse_loss(terminal_values, terminal_rewards)

        critic_loss = martingale_loss + self.config.constraint_weight * (
            q_constraint_loss + terminal_loss
        )
        self.q_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optimizer.step()
        self.value_optimizer.step()
        self._soft_update_value_target()

        return {
            "critic_loss": float(critic_loss.detach()),
            "martingale_loss": float(martingale_loss.detach()),
            "q_constraint_loss": float(q_constraint_loss.detach()),
            "terminal_loss": float(terminal_loss.detach()),
        }

    def update_policy(self, sequence_batch: SequenceBatch) -> float:
        states = self._tensor(sequence_batch.states)
        times = self._tensor(sequence_batch.times)

        for parameter in self.q_rate.parameters():
            parameter.requires_grad_(False)
        try:
            policy_loss = -self.q_rate(
                states, self.policy(states, times), times
            ).squeeze(-1)
            policy_loss = self.config.dt * policy_loss.mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_optimizer.step()
        finally:
            for parameter in self.q_rate.parameters():
                parameter.requires_grad_(True)
        return float(policy_loss.detach())

    def train(
        self,
        episodes: int,
        eval_every: int,
        eval_episodes: int,
    ) -> list[dict[str, Any]]:
        if episodes <= 0 or eval_every <= 0 or eval_episodes <= 0:
            raise ValueError("episodes and evaluation arguments must be positive")

        buffer = EpisodeSequenceBuffer(
            self.config.buffer_size, self.env.num_envs, self.rng
        )
        history: list[dict[str, Any]] = []
        vector_steps = 0
        update_count = 0
        latest_losses: dict[str, float] = {}

        for episode in range(episodes):
            reset_options = {"seed": self.seed} if episode == 0 else {}
            states, info = self.env.reset(**reset_options)
            times = np.asarray(info["time"], dtype=np.float32)
            buffer.start_rollout()
            rollout_done = False

            while not rollout_done:
                actions = self.select_action(
                    states, times, self.config.exploration_noise
                )
                next_states, rewards, terminated, truncated, next_info = self.env.step(
                    actions
                )
                buffer.add_step(states, actions, rewards, times)
                vector_steps += 1

                states = next_states
                times = np.asarray(next_info["time"], dtype=np.float32)
                rollout_done = bool(np.all(np.logical_or(terminated, truncated)))

                if (
                    vector_steps % self.config.update_frequency == 0
                    and buffer.completed_size >= self.config.warmup_episodes
                ):
                    for _ in range(self.config.gradient_steps):
                        for _ in range(self.config.critic_steps):
                            critic_batch = buffer.sample_sequences(
                                self.config.batch_size,
                                self.config.sequence_length,
                                self.config.max_sequence_length,
                            )
                            terminal_batch = buffer.sample_terminals(
                                self.config.batch_size
                            )
                            latest_losses.update(
                                self.update_critic(critic_batch, terminal_batch)
                            )
                        policy_batch = buffer.sample_sequences(
                            self.config.batch_size,
                            1,
                            self.config.max_policy_sequence_length,
                        )
                        latest_losses["policy_loss"] = self.update_policy(policy_batch)
                        self.q_scheduler.step()
                        self.value_scheduler.step()
                        self.policy_scheduler.step()
                        update_count += 1

            terminal_rewards = np.asarray(
                self.env.terminal_reward(), dtype=np.float32
            ).reshape(self.env.num_envs)
            buffer.finish_rollout(states, terminal_rewards, times)

            if (episode + 1) % eval_every == 0:
                mean_return, std_return = self.evaluate(eval_episodes)
                record: dict[str, Any] = {
                    "episode": episode + 1,
                    "environment_transitions": vector_steps * self.env.num_envs,
                    "updates": update_count,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    **latest_losses,
                }
                history.append(record)
                print(
                    f"episode={episode + 1} transitions={record['environment_transitions']} "
                    f"return={mean_return:.3f} +/- {std_return:.3f} updates={update_count}",
                    flush=True,
                )
                self.save(history, episode + 1, vector_steps, update_count)

        self.save(history, episodes, vector_steps, update_count)
        return history

    @torch.no_grad()
    def evaluate(self, episodes: int) -> tuple[float, float]:
        returns: list[float] = []
        for _ in range(episodes):
            reset_options: dict[str, Any] = {}
            if not self._eval_seeded:
                reset_options["seed"] = self.seed + 10_000
                self._eval_seeded = True
            states, info = self.eval_env.reset(**reset_options)
            times = np.asarray(info["time"], dtype=np.float32)
            cumulative_reward = np.zeros(self.eval_env.num_envs, dtype=np.float64)
            rollout_done = False
            while not rollout_done:
                actions = self.select_action(states, times)
                states, rewards, terminated, truncated, info = self.eval_env.step(
                    actions
                )
                times = np.asarray(info["time"], dtype=np.float32)
                cumulative_reward += np.asarray(rewards) * self.config.dt
                rollout_done = bool(np.all(np.logical_or(terminated, truncated)))
            cumulative_reward += np.asarray(
                self.eval_env.terminal_reward(), dtype=np.float64
            ).reshape(self.eval_env.num_envs)
            returns.extend(cumulative_reward.tolist())
        return float(np.mean(returns)), float(np.std(returns))

    def save(
        self,
        history: list[dict[str, Any]],
        episode: int,
        vector_steps: int,
        update_count: int,
    ) -> None:
        if self.output_dir is None:
            return
        checkpoint = {
            "algorithm": "CT-DDPG",
            "config": asdict(self.config),
            "episode": episode,
            "vector_steps": vector_steps,
            "update_count": update_count,
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "value_target": self.value_target.state_dict(),
            "q_rate": self.q_rate.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / "checkpoint.pt")

        history_path = self.output_dir / "history.json"
        temporary_path = self.output_dir / "history.json.tmp"
        temporary_path.write_text(json.dumps(history, indent=2) + "\n")
        temporary_path.replace(history_path)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _soft_update_value_target(self) -> None:
        tau = self.config.tau
        for parameter, target_parameter in zip(
            self.value.parameters(), self.value_target.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
