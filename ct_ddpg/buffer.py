"""Online episode-sequence replay for CT-DDPG."""

from collections import deque
from typing import NamedTuple

import numpy as np


Step = tuple[np.ndarray, np.ndarray | None, float, np.ndarray]


class SequenceBatch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    times: np.ndarray


class TerminalBatch(NamedTuple):
    states: np.ndarray
    rewards: np.ndarray
    times: np.ndarray


class SequenceBuffer:
    def __init__(self, capacity: int, num_envs: int, rng: np.random.Generator) -> None:
        self.episodes: deque[list[Step]] = deque(maxlen=capacity)
        self.terminals: deque[tuple[np.ndarray, float, np.ndarray]] = deque(
            maxlen=capacity
        )
        self.num_envs = num_envs
        self.rng = rng
        self.active: list[list[Step]] = []

    def start_rollout(self) -> None:
        self.active = [[] for _ in range(self.num_envs)]

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        times: np.ndarray,
    ) -> None:
        for i in range(self.num_envs):
            self.active[i].append(
                (
                    np.asarray(states[i], dtype=np.float32),
                    np.asarray(actions[i], dtype=np.float32),
                    float(rewards[i]),
                    np.asarray(times[i], dtype=np.float32),
                )
            )

    def finish(
        self, states: np.ndarray, rewards: np.ndarray, times: np.ndarray
    ) -> None:
        for i, episode in enumerate(self.active):
            terminal = (
                np.asarray(states[i], dtype=np.float32),
                float(rewards[i]),
                np.asarray(times[i], dtype=np.float32),
            )
            episode.append((terminal[0], None, terminal[1], terminal[2]))
            self.episodes.append(episode)
            self.terminals.append(terminal)
        self.active = []

    def sample(
        self, batch_size: int, min_sequence_length: int, max_sequence_length: int
    ) -> SequenceBatch:
        pool = [*self.episodes, *self.active]
        if not pool:
            raise RuntimeError("no trajectory is long enough to sample")

        indices = self.rng.integers(len(pool), size=batch_size)
        episodes = [
            pool[i] for i in indices if len(pool[i]) > min_sequence_length
        ]
        if not episodes:
            raise RuntimeError("sampled trajectories are too short")
        ends = np.asarray(
            [
                self.rng.integers(min_sequence_length, len(episode))
                for episode in episodes
            ]
        )
        length = min(
            int(self.rng.integers(min_sequence_length, ends.min() + 1)),
            max_sequence_length,
        )
        sequences = [
            episode[end - length : end] for episode, end in zip(episodes, ends)
        ]

        return SequenceBatch(
            *[
                np.asarray(
                    [[step[field] for step in sequence] for sequence in sequences],
                    dtype=np.float32,
                )
                for field in range(4)
            ]
        )

    def sample_terminals(self, batch_size: int) -> TerminalBatch:
        count = min(batch_size, len(self.terminals))
        samples = [
            self.terminals[i]
            for i in self.rng.integers(len(self.terminals), size=count)
        ]
        return TerminalBatch(
            *[
                np.asarray([sample[field] for sample in samples], dtype=np.float32)
                for field in range(3)
            ]
        )
