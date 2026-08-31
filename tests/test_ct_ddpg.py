import unittest

import gymnasium as gym
import numpy as np
import torch

from ct_ddpg import CTDDPG, CTDDPGConfig
from ct_ddpg.buffer import SequenceBatch, SequenceBuffer, TerminalBatch


class FakeVectorEnv:
    num_envs = 2
    single_observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(3,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)


class CTDDPGTests(unittest.TestCase):
    def test_sequence_buffer(self) -> None:
        buffer = SequenceBuffer(8, 2, np.random.default_rng(0))
        buffer.start_rollout()
        for step in range(4):
            buffer.add(
                np.full((2, 3), step),
                np.full((2, 2), step),
                np.full(2, step),
                np.full((2, 1), step / 4),
            )
        buffer.finish(np.ones((2, 3)), np.zeros(2), np.ones((2, 1)))

        batch = buffer.sample(5, min_sequence_length=2, max_sequence_length=3)
        self.assertEqual(batch.states.shape[0], 5)
        self.assertTrue(2 <= batch.states.shape[1] <= 3)
        self.assertEqual(buffer.sample_terminals(8).states.shape, (2, 3))

    def test_updates(self) -> None:
        torch.manual_seed(0)
        config = CTDDPGConfig(batch_size=4, hidden_dim=16, layers=1)
        algorithm = CTDDPG(FakeVectorEnv(), FakeVectorEnv(), "cpu", config)
        rng = np.random.default_rng(0)
        sequences = SequenceBatch(
            rng.normal(size=(4, 3, 3)).astype(np.float32),
            rng.uniform(-1, 1, size=(4, 3, 2)).astype(np.float32),
            rng.normal(size=(4, 3)).astype(np.float32),
            np.tile(
                np.array([0, 0.1, 0.2], dtype=np.float32)[None, :, None], (4, 1, 1)
            ),
        )
        terminals = TerminalBatch(
            rng.normal(size=(4, 3)).astype(np.float32),
            np.zeros(4, dtype=np.float32),
            np.ones((4, 1), dtype=np.float32),
        )

        losses = algorithm.update_critic(sequences, terminals)
        policy_loss = algorithm.update_policy(sequences)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertTrue(np.isfinite(policy_loss))


if __name__ == "__main__":
    unittest.main()
