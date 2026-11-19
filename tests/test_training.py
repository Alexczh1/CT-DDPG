import unittest

from ct_ddpg import CTDDPG, CTDDPGConfig
from test_ct_ddpg import FakeVectorEnv
from train import parse_args
from train_halfcheetah import parse_args as parse_halfcheetah_args


class TrainingConfigTests(unittest.TestCase):
    def test_halfcheetah_entry_point_preserves_defaults(self) -> None:
        args = parse_halfcheetah_args([])
        self.assertEqual(
            (
                args.env,
                args.dt,
                args.horizon,
                args.num_envs,
                args.eval_envs,
                args.eval_every,
                args.hidden_dim,
                args.force_noise_std,
                args.learning_rate,
                args.terminal_loss_weight,
            ),
            ("halfcheetah", 0.05, 50.0, 8, 4, 10, 400, 0.0, 3e-4, 0.002),
        )

    def test_task_defaults(self) -> None:
        expected = {
            "hopper": (0.0008, 8.0, 0.1, 5, 1000, 2, 10),
            "walker2d": (0.0004, 8.0, 1.0, 10, 1000, 2, 10),
            "pendulum": (0.005, 10.0, 1.0, 5, 100, 10, 100),
        }
        for environment, settings in expected.items():
            with self.subTest(environment=environment):
                args = parse_args(["--env", environment])
                self.assertEqual(
                    (
                        args.dt,
                        args.horizon,
                        args.force_noise_std,
                        args.update_frequency,
                        args.buffer_size,
                        args.min_sequence_length,
                        args.max_sequence_length,
                    ),
                    settings,
                )

    def test_explicit_overrides_and_noise_alias(self) -> None:
        args = parse_args(
            [
                "--env", "pendulum",
                "--dt", "0.05",
                "--noise-std", "0",
                "--value-learning-rate", "0.001",
                "--max-policy-sequence-length", "3",
            ]
        )
        self.assertEqual(args.dt, 0.05)
        self.assertEqual(args.force_noise_std, 0.0)
        self.assertEqual(args.value_learning_rate, 0.001)
        self.assertEqual(args.max_policy_sequence_length, 3)
        legacy = parse_args(["--force-noise-std", "0.5"])
        self.assertEqual(legacy.force_noise_std, 0.5)

    def test_pendulum_optimizer_rates(self) -> None:
        args = parse_args(["--env", "pendulum"])
        config = CTDDPGConfig(
            hidden_dim=4,
            layers=0,
            learning_rate=args.learning_rate,
            policy_learning_rate=args.policy_learning_rate,
            value_learning_rate=args.value_learning_rate,
            q_learning_rate=args.q_learning_rate,
            max_policy_sequence_length=args.max_policy_sequence_length,
        )
        algorithm = CTDDPG(FakeVectorEnv(), FakeVectorEnv(), "cpu", config)
        self.assertEqual(algorithm.policy_optimizer.param_groups[0]["lr"], 3e-3)
        self.assertEqual(algorithm.value_optimizer.param_groups[0]["lr"], 3e-2)
        self.assertEqual(algorithm.q_optimizer.param_groups[0]["lr"], 3e-2)
        self.assertEqual(config.max_policy_sequence_length, 10)

    def test_learning_rates_must_be_positive(self) -> None:
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    CTDDPGConfig(value_learning_rate=rate)


if __name__ == "__main__":
    unittest.main()
