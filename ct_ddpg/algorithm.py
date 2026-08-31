"""CT-DDPG with online trajectory-sequence updates."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.nn import functional as F

from .buffer import SequenceBatch, SequenceBuffer, TerminalBatch
from .networks import Policy, QRate, Value


@dataclass(frozen=True)
class CTDDPGConfig:
    dt: float = 0.05
    discount_factor: float = 0.8
    buffer_size: int = 10_000
    batch_size: int = 256
    min_sequence_length: int = 2
    max_sequence_length: int = 10
    warmup_episodes: int = 5
    update_frequency: int = 1
    exploration_noise: float = 0.1
    learning_rate: float = 3e-4
    tau: float = 0.005
    terminal_loss_weight: float = 0.002
    hidden_dim: int = 400
    layers: int = 2

    def __post_init__(self) -> None:
        if self.dt <= 0 or not 0 < self.discount_factor <= 1:
            raise ValueError("dt must be positive and discount_factor in (0, 1]")
        if not 2 <= self.min_sequence_length <= self.max_sequence_length:
            raise ValueError("require 2 <= min_sequence_length <= max_sequence_length")
        if (
            min(
                self.buffer_size,
                self.batch_size,
                self.warmup_episodes,
                self.update_frequency,
            )
            <= 0
        ):
            raise ValueError("buffer, batch, warmup, and update sizes must be positive")


class CTDDPG:
    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        device: str | torch.device,
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
        self.output_dir = Path(output_dir) if output_dir else None
        self.eval_seeded = False

        state_dim = env.single_observation_space.shape[0]
        action_space = env.single_action_space
        network_args = (config.hidden_dim, config.layers)
        self.policy = Policy(
            state_dim, action_space.low, action_space.high, *network_args
        ).to(self.device)
        self.value = Value(state_dim, *network_args).to(self.device)
        self.target_value = Value(state_dim, *network_args).to(self.device)
        self.target_value.load_state_dict(self.value.state_dict())
        self.q_rate = QRate(state_dim, action_space.shape[0], *network_args).to(
            self.device
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_rate.parameters(), lr=config.learning_rate
        )

    @torch.no_grad()
    def act(
        self, states: np.ndarray, times: np.ndarray, noise: float = 0
    ) -> np.ndarray:
        actions = self.policy(self.tensor(states), self.tensor(times)).cpu().numpy()
        if noise:
            actions += noise * self.rng.normal(size=actions.shape)
        return np.clip(
            actions, self.env.single_action_space.low, self.env.single_action_space.high
        ).astype(np.float32)

    def update_critic(
        self, sequences: SequenceBatch, terminals: TerminalBatch
    ) -> dict[str, float]:
        states, actions, rewards, times = map(self.tensor, sequences)

        with torch.no_grad():
            policy_actions = self.policy(states, times)
        centered_q = self.q_rate(states, actions, times).squeeze(-1) - self.q_rate(
            states, policy_actions, times
        ).squeeze(-1)

        intervals = rewards.shape[1] - 1
        discounts = self.config.discount_factor ** (
            self.config.dt * torch.arange(intervals, device=self.device)
        )
        running_return = (
            (rewards[:, :-1] - centered_q[:, :-1]) * self.config.dt * discounts
        ).sum(-1)

        with torch.no_grad():
            bootstrap = self.target_value(states[:, -1], times[:, -1]).squeeze(-1)
            bootstrap *= self.config.discount_factor ** (self.config.dt * intervals)
        value = self.value(states[:, 0], times[:, 0]).squeeze(-1)
        martingale_loss = F.mse_loss(value, running_return + bootstrap)

        terminal_states, terminal_rewards, terminal_times = map(self.tensor, terminals)
        terminal_loss = F.mse_loss(
            self.value(terminal_states, terminal_times).squeeze(-1), terminal_rewards
        )
        loss = martingale_loss + self.config.terminal_loss_weight * terminal_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.q_optimizer.step()
        self.value_optimizer.step()

        with torch.no_grad():
            for value_parameter, target_parameter in zip(
                self.value.parameters(), self.target_value.parameters()
            ):
                target_parameter.lerp_(value_parameter, self.config.tau)

        return {
            "critic_loss": loss.item(),
            "martingale_loss": martingale_loss.item(),
            "terminal_loss": terminal_loss.item(),
        }

    def update_policy(self, sequences: SequenceBatch) -> float:
        states, times = self.tensor(sequences.states), self.tensor(sequences.times)
        for parameter in self.q_rate.parameters():
            parameter.requires_grad_(False)
        loss = (
            -self.config.dt
            * self.q_rate(states, self.policy(states, times), times).mean()
        )
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        for parameter in self.q_rate.parameters():
            parameter.requires_grad_(True)
        return loss.item()

    def train(
        self, episodes: int, eval_every: int, eval_episodes: int
    ) -> list[dict[str, float | int]]:
        buffer = SequenceBuffer(self.config.buffer_size, self.env.num_envs, self.rng)
        history: list[dict[str, float | int]] = []
        updates = 0
        vector_steps = 0
        losses: dict[str, float] = {}

        for episode in range(1, episodes + 1):
            states, info = self.env.reset(seed=self.seed if episode == 1 else None)
            times = info["time"]
            buffer.start_rollout()
            done = False

            while not done:
                actions = self.act(states, times, self.config.exploration_noise)
                next_states, rewards, terminated, truncated, info = self.env.step(
                    actions
                )
                buffer.add(states, actions, rewards, times)
                states, times = next_states, info["time"]
                done = bool(np.all(terminated | truncated))
                vector_steps += 1

                if (
                    len(buffer.episodes) >= self.config.warmup_episodes
                    and vector_steps % self.config.update_frequency == 0
                ):
                    critic_batch = buffer.sample(
                        self.config.batch_size,
                        self.config.min_sequence_length,
                        self.config.max_sequence_length,
                    )
                    losses = self.update_critic(
                        critic_batch, buffer.sample_terminals(self.config.batch_size)
                    )
                    policy_batch = buffer.sample(
                        self.config.batch_size, 1, self.config.max_sequence_length
                    )
                    losses["policy_loss"] = self.update_policy(policy_batch)
                    updates += 1

            buffer.finish(states, np.zeros(self.env.num_envs), times)

            if episode % eval_every == 0:
                mean_return, std_return = self.evaluate(eval_episodes)
                record: dict[str, float | int] = {
                    "episode": episode,
                    "transitions": vector_steps * self.env.num_envs,
                    "updates": updates,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    **losses,
                }
                history.append(record)
                print(
                    f"episode={episode} transitions={record['transitions']} "
                    f"return={mean_return:.3f} +/- {std_return:.3f} updates={updates}",
                    flush=True,
                )
                self.save(history, episode)

        self.save(history, episodes)
        return history

    @torch.no_grad()
    def evaluate(self, episodes: int) -> tuple[float, float]:
        returns: list[float] = []
        for _ in range(episodes):
            seed = None if self.eval_seeded else self.seed + 10_000
            self.eval_seeded = True
            states, info = self.eval_env.reset(seed=seed)
            times = info["time"]
            total = np.zeros(self.eval_env.num_envs)
            done = False
            while not done:
                states, rewards, terminated, truncated, info = self.eval_env.step(
                    self.act(states, times)
                )
                times = info["time"]
                total += rewards * self.config.dt
                done = bool(np.all(terminated | truncated))
            returns.extend(total.tolist())
        return float(np.mean(returns)), float(np.std(returns))

    def save(self, history: list[dict[str, float | int]], episode: int) -> None:
        if self.output_dir is None:
            return
        torch.save(
            {
                "algorithm": "CT-DDPG",
                "episode": episode,
                "config": asdict(self.config),
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "q_rate": self.q_rate.state_dict(),
            },
            self.output_dir / "checkpoint.pt",
        )
        (self.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )

    def tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)
