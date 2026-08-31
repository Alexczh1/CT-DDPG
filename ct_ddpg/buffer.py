"""Online trajectory replay for CT-DDPG."""

from collections import deque
from typing import NamedTuple

import numpy as np


Transition = tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]


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
        self.episodes: deque[list[Transition]] = deque(maxlen=capacity)
        self.terminals: deque[tuple[np.ndarray, float, np.ndarray]] = deque(
            maxlen=capacity
        )
        self.num_envs = num_envs
        self.rng = rng
        self.active: list[list[Transition]] = []

    def start_rollout(self) -> None:
        self.active = [[] for _ in range(self.num_envs)]

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        times: np.ndarray,
        next_times: np.ndarray,
    ) -> None:
        for i in range(self.num_envs):
            self.active[i].append(
                (
                    np.asarray(states[i], dtype=np.float32),
                    np.asarray(actions[i], dtype=np.float32),
                    float(rewards[i]),
                    np.asarray(next_states[i], dtype=np.float32),
                    np.asarray(times[i], dtype=np.float32),
                    np.asarray(next_times[i], dtype=np.float32),
                )
            )

    def finish(
        self, states: np.ndarray, rewards: np.ndarray, times: np.ndarray
    ) -> None:
        self.episodes.extend(self.active)
        self.terminals.extend(
            (
                np.asarray(states[i], dtype=np.float32),
                float(rewards[i]),
                np.asarray(times[i], dtype=np.float32),
            )
            for i in range(self.num_envs)
        )
        self.active = []

    def sample(
        self, batch_size: int, min_sequence_length: int, max_sequence_length: int
    ) -> SequenceBatch:
        length = int(self.rng.integers(min_sequence_length, max_sequence_length + 1))
        candidates = [
            episode
            for episode in [*self.episodes, *self.active]
            if len(episode) >= length
        ]
        if not candidates:
            raise RuntimeError(f"no trajectory contains {length} transitions")

        windows = []
        for index in self.rng.integers(len(candidates), size=batch_size):
            episode = candidates[index]
            start = int(self.rng.integers(len(episode) - length + 1))
            windows.append(episode[start : start + length])

        return SequenceBatch(
            states=np.asarray(
                [[step[0] for step in window] + [window[-1][3]] for window in windows],
                dtype=np.float32,
            ),
            actions=np.asarray(
                [[step[1] for step in window] for window in windows],
                dtype=np.float32,
            ),
            rewards=np.asarray(
                [[step[2] for step in window] for window in windows],
                dtype=np.float32,
            ),
            times=np.asarray(
                [[step[4] for step in window] + [window[-1][5]] for window in windows],
                dtype=np.float32,
            ),
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
