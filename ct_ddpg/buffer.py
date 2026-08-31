"""Episode buffer and sequence sampler for online CT-DDPG updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


Step: TypeAlias = tuple[np.ndarray, np.ndarray | None, float, np.ndarray]


@dataclass(frozen=True)
class SequenceBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    times: np.ndarray


@dataclass(frozen=True)
class TerminalBatch:
    states: np.ndarray
    rewards: np.ndarray
    times: np.ndarray


class EpisodeSequenceBuffer:
    """Circular episode storage that also samples active online trajectories."""

    def __init__(
        self,
        max_episodes: int,
        num_envs: int,
        rng: np.random.Generator,
    ) -> None:
        if max_episodes <= 0 or num_envs <= 0:
            raise ValueError("max_episodes and num_envs must be positive")
        self.max_episodes = max_episodes
        self.num_envs = num_envs
        self.rng = rng
        self._episodes: list[list[Step]] = []
        self._terminals: list[tuple[np.ndarray, float, np.ndarray]] = []
        self._episode_ptr = 0
        self._terminal_ptr = 0
        self._active: list[list[Step]] | None = None

    @property
    def completed_size(self) -> int:
        return len(self._episodes)

    def start_rollout(self) -> None:
        self._active = [[] for _ in range(self.num_envs)]

    def add_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        times: np.ndarray,
    ) -> None:
        if self._active is None:
            raise RuntimeError("start_rollout() must be called before add_step()")
        for index in range(self.num_envs):
            self._active[index].append(
                (
                    np.asarray(states[index], dtype=np.float32).copy(),
                    np.asarray(actions[index], dtype=np.float32).copy(),
                    float(rewards[index]),
                    np.asarray(times[index], dtype=np.float32).copy(),
                )
            )

    def finish_rollout(
        self,
        final_states: np.ndarray,
        terminal_rewards: np.ndarray,
        final_times: np.ndarray,
    ) -> None:
        if self._active is None:
            raise RuntimeError("no active rollout to finish")
        for index, trajectory in enumerate(self._active):
            final_state = np.asarray(final_states[index], dtype=np.float32).copy()
            final_time = np.asarray(final_times[index], dtype=np.float32).copy()
            terminal_reward = float(terminal_rewards[index])

            # This terminal record anchors V at the horizon. Sequence sampling
            # uses it only as an exclusive endpoint, matching the source method.
            trajectory.append((final_state, None, terminal_reward, final_time))
            self._append_episode(trajectory)
            self._append_terminal((final_state, terminal_reward, final_time))
        self._active = None

    def sample_sequences(
        self,
        batch_size: int,
        min_length: int,
        max_length: int,
    ) -> SequenceBatch:
        if batch_size <= 0 or min_length <= 0 or max_length < min_length:
            raise ValueError("invalid sequence sampling arguments")

        candidates = list(self._episodes)
        if self._active is not None:
            candidates.extend(self._active)
        eligible = [
            trajectory for trajectory in candidates if len(trajectory) > min_length
        ]
        if not eligible:
            raise RuntimeError("no trajectory is long enough to sample")

        selected = [
            eligible[i] for i in self.rng.integers(0, len(eligible), batch_size)
        ]
        end_indices = np.asarray(
            [self.rng.integers(min_length, len(trajectory)) for trajectory in selected]
        )
        sampled_length = int(self.rng.integers(min_length, int(end_indices.min()) + 1))
        sampled_length = min(sampled_length, max_length)

        sequences = [
            trajectory[end - sampled_length : end]
            for trajectory, end in zip(selected, end_indices)
        ]
        if any(step[1] is None for sequence in sequences for step in sequence):
            raise RuntimeError("terminal actions must not appear in training sequences")

        return SequenceBatch(
            states=np.asarray(
                [[step[0] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            ),
            actions=np.asarray(
                [[step[1] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            ),
            rewards=np.asarray(
                [[step[2] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            ),
            times=np.asarray(
                [[step[3] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            ),
        )

    def sample_terminals(self, batch_size: int) -> TerminalBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._terminals:
            raise RuntimeError("no terminal states are available")
        sample_size = min(batch_size, len(self._terminals))
        selected = [
            self._terminals[i]
            for i in self.rng.integers(0, len(self._terminals), sample_size)
        ]
        return TerminalBatch(
            states=np.asarray([item[0] for item in selected], dtype=np.float32),
            rewards=np.asarray([item[1] for item in selected], dtype=np.float32),
            times=np.asarray([item[2] for item in selected], dtype=np.float32),
        )

    def _append_episode(self, trajectory: list[Step]) -> None:
        if len(self._episodes) < self.max_episodes:
            self._episodes.append(trajectory)
        else:
            self._episodes[self._episode_ptr] = trajectory
        self._episode_ptr = (self._episode_ptr + 1) % self.max_episodes

    def _append_terminal(self, terminal: tuple[np.ndarray, float, np.ndarray]) -> None:
        if len(self._terminals) < self.max_episodes:
            self._terminals.append(terminal)
        else:
            self._terminals[self._terminal_ptr] = terminal
        self._terminal_ptr = (self._terminal_ptr + 1) % self.max_episodes
