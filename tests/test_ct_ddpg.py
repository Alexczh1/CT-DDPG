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

        lengths = set()
        for _ in range(20):
            batch = buffer.sample(5, min_sequence_length=2, max_sequence_length=3)
            lengths.add(batch.actions.shape[1])
            self.assertEqual(batch.states.shape[1], batch.actions.shape[1])
            self.assertEqual(batch.rewards.shape[1], batch.actions.shape[1])
            self.assertEqual(batch.times.shape[1], batch.actions.shape[1])
        self.assertEqual(lengths, {2, 3})
        self.assertEqual(buffer.sample_terminals(8).states.shape, (2, 3))

    def test_training_hyperparameters(self) -> None:
        config = CTDDPGConfig()
        self.assertEqual(
            (
                config.hidden_dim,
                config.layers,
                config.learning_rate,
                config.lr_decay_steps,
                config.lr_decay_gamma,
                config.batch_size,
                config.update_frequency,
                config.discount_factor,
                config.tau,
                config.terminal_loss_weight,
                config.min_sequence_length,
                config.max_sequence_length,
                config.exploration_noise,
            ),
            (400, 2, 3e-4, 80_000, 0.8, 256, 1, 0.8, 0.005, 0.002, 2, 10, 0.1),
        )

    def test_learning_rate_decay(self) -> None:
        torch.manual_seed(0)
        config = CTDDPGConfig(
            batch_size=2,
            hidden_dim=4,
            layers=0,
            lr_decay_steps=2,
            lr_decay_gamma=0.8,
        )
        algorithm = CTDDPG(FakeVectorEnv(), FakeVectorEnv(), "cpu", config)
        rng = np.random.default_rng(0)
        sequences = SequenceBatch(
            rng.normal(size=(2, 2, 3)).astype(np.float32),
            rng.uniform(-1, 1, size=(2, 2, 2)).astype(np.float32),
            rng.normal(size=(2, 2)).astype(np.float32),
            np.zeros((2, 2, 1), dtype=np.float32),
        )
        terminals = TerminalBatch(
            rng.normal(size=(2, 3)).astype(np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )

        for _ in range(2):
            algorithm.update_critic(sequences, terminals)
            algorithm.update_policy(sequences)

        expected = config.learning_rate * config.lr_decay_gamma
        for optimizer in (
            algorithm.policy_optimizer,
            algorithm.value_optimizer,
            algorithm.q_optimizer,
        ):
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

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
                np.array([0, 0.1, 0.2], dtype=np.float32)[None, :, None],
                (4, 1, 1),
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

    def test_original_discount_factor(self) -> None:
        config = CTDDPGConfig(batch_size=2, hidden_dim=4, layers=0)
        algorithm = CTDDPG(FakeVectorEnv(), FakeVectorEnv(), "cpu", config)
        for network in (algorithm.q_rate, algorithm.value, algorithm.target_value):
            for parameter in network.parameters():
                parameter.data.zero_()
        algorithm.target_value.net[-1].bias.data.fill_(1)

        sequences = SequenceBatch(
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((2, 3, 2), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.float32),
        )
        terminals = TerminalBatch(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
        losses = algorithm.update_critic(sequences, terminals)
        expected = config.discount_factor ** (2 * config.dt * 2)
        self.assertAlmostEqual(losses["martingale_loss"], expected, places=6)

    def test_original_sequence_uses_l_minus_one_transitions(self) -> None:
        config = CTDDPGConfig(batch_size=2, hidden_dim=4, layers=0)
        algorithm = CTDDPG(FakeVectorEnv(), FakeVectorEnv(), "cpu", config)
        for network in (algorithm.q_rate, algorithm.value, algorithm.target_value):
            for parameter in network.parameters():
                parameter.data.zero_()

        sequences = SequenceBatch(
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((2, 3, 2), dtype=np.float32),
            np.tile(np.array([1, 2, 1e6], dtype=np.float32), (2, 1)),
            np.zeros((2, 3, 1), dtype=np.float32),
        )
        terminals = TerminalBatch(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
        losses = algorithm.update_critic(sequences, terminals)
        expected_return = config.dt * (1 + config.discount_factor**config.dt * 2)
        self.assertAlmostEqual(losses["martingale_loss"], expected_return**2, places=6)


if __name__ == "__main__":
    unittest.main()
