from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from ct_ddpg import CTDDPGConfig, DDPGContinuousOnlineSequence, NetworkConfig
from ct_ddpg.buffer import EpisodeSequenceBuffer, SequenceBatch, TerminalBatch


class FakeVectorEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )


class CTDDPGTests(unittest.TestCase):
    def test_buffer_shapes_and_terminal_sampling(self) -> None:
        rng = np.random.default_rng(7)
        buffer = EpisodeSequenceBuffer(max_episodes=8, num_envs=2, rng=rng)
        buffer.start_rollout()
        for step in range(4):
            buffer.add_step(
                np.full((2, 3), step, dtype=np.float32),
                np.full((2, 2), 0.1 * step, dtype=np.float32),
                np.full(2, step, dtype=np.float32),
                np.full((2, 1), step / 4, dtype=np.float32),
            )
        buffer.finish_rollout(
            np.full((2, 3), 4, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )

        batch = buffer.sample_sequences(batch_size=5, min_length=2, max_length=3)
        self.assertEqual(batch.states.shape[0], 5)
        self.assertGreaterEqual(batch.states.shape[1], 2)
        self.assertLessEqual(batch.states.shape[1], 3)
        self.assertEqual(batch.actions.shape[-1], 2)
        terminals = buffer.sample_terminals(8)
        self.assertEqual(terminals.states.shape, (2, 3))
        self.assertEqual(terminals.times.shape, (2, 1))

    def test_critic_and_policy_updates_are_finite(self) -> None:
        torch.manual_seed(3)
        network = NetworkConfig(hidden_dim=16, layers=1)
        config = CTDDPGConfig(
            batch_size=4,
            buffer_size=8,
            policy_network=network,
            value_network=network,
            q_network=network,
        )
        env = FakeVectorEnv()
        algorithm = DDPGContinuousOnlineSequence(env, env, "cpu", config, seed=3)
        rng = np.random.default_rng(3)
        batch = SequenceBatch(
            states=rng.normal(size=(4, 3, 3)).astype(np.float32),
            actions=rng.uniform(-1, 1, size=(4, 3, 2)).astype(np.float32),
            rewards=rng.normal(size=(4, 3)).astype(np.float32),
            times=np.broadcast_to(
                np.asarray([[[0.0], [0.1], [0.2]]], dtype=np.float32),
                (4, 3, 1),
            ).copy(),
        )
        terminals = TerminalBatch(
            states=rng.normal(size=(4, 3)).astype(np.float32),
            rewards=np.zeros(4, dtype=np.float32),
            times=np.ones((4, 1), dtype=np.float32),
        )

        losses = algorithm.update_critic(batch, terminals)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        before = [
            parameter.detach().clone() for parameter in algorithm.policy.parameters()
        ]
        policy_loss = algorithm.update_policy(batch)
        self.assertTrue(np.isfinite(policy_loss))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, algorithm.policy.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
