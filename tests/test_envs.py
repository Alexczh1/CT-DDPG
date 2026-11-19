import unittest

import gymnasium as gym
import numpy as np

from ct_ddpg.envs import (
    HalfCheetahCT,
    HopperCT,
    PendulumCT,
    Walker2dCT,
    make_envs,
    make_halfcheetah_envs,
)


class BenchmarkEnvTests(unittest.TestCase):
    def test_invalid_time_and_noise_parameters(self):
        for constructor in (HalfCheetahCT, HopperCT, Walker2dCT, PendulumCT):
            for dt, horizon in (
                (0, 1), (-0.1, 1), (np.nan, 1), (np.inf, 1),
                (0.1, 0), (0.1, np.nan), (0.1, np.inf), (0.3, 1),
            ):
                with self.subTest(environment=constructor.__name__, dt=dt, horizon=horizon):
                    with self.assertRaises(ValueError):
                        constructor(dt, horizon)
            for noise in (-1, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    constructor(0.1, 1, noise)

    def test_pendulum_matches_source_dynamics_and_reward(self):
        env = PendulumCT(dt=0.05, horizon=0.1)
        try:
            env.reset(seed=12)
            env.state = np.array([0.5, -0.2], dtype=np.float32)
            theta, velocity = env.state
            torque = np.float32(1.5)
            expected_reward = -(theta**2 + 0.1 * velocity**2 + 0.001 * torque**2)
            expected_velocity = velocity + (15 * np.sin(theta) + 3 * torque) * 0.05
            expected_state = np.array(
                [theta + expected_velocity * 0.05, expected_velocity], dtype=np.float32
            )
            observation, reward, terminated, truncated, info = env.step(np.array([torque]))
            np.testing.assert_allclose(env.state, expected_state, rtol=0, atol=1e-7)
            np.testing.assert_allclose(
                observation,
                [np.cos(env.state[0]), np.sin(env.state[0]), env.state[1]],
                rtol=0,
                atol=1e-7,
            )
            self.assertAlmostEqual(reward, float(expected_reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            np.testing.assert_array_equal(info["time"], np.array([0.5], dtype=np.float32))
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()

    def test_pendulum_velocity_noise_scales_with_sqrt_dt(self):
        generator = np.random.default_rng(123)
        generator.uniform(-np.pi, np.pi)
        generator.uniform(-1, 1)
        normal = generator.normal()
        for dt in (0.01, 0.04):
            env = PendulumCT(dt=dt, horizon=0.08, noise_std=0.3)
            try:
                env.reset(seed=123)
                env.state = np.zeros(2, dtype=np.float32)
                env.step(np.zeros(1, dtype=np.float32))
                expected_velocity = 0.3 * np.sqrt(dt) * normal
                np.testing.assert_allclose(
                    env.state,
                    np.array([expected_velocity * dt, expected_velocity], dtype=np.float32),
                    rtol=0,
                    atol=1e-7,
                )
            finally:
                env.close()

    def test_pendulum_clips_torque_speed_and_wraps_angle(self):
        env = PendulumCT(dt=0.1, horizon=0.2)
        try:
            env.reset(seed=3)
            env.state = np.array([np.pi - 0.01, 8], dtype=np.float32)
            theta, velocity = env.state
            expected_reward = -(theta**2 + 0.1 * velocity**2 + 0.001 * 2**2)
            _, reward, _, _, _ = env.step(np.array([5], dtype=np.float32))
            self.assertEqual(env.state.dtype, np.float32)
            self.assertEqual(env.state[1], 8)
            self.assertGreaterEqual(env.state[0], -np.pi)
            self.assertLess(env.state[0], 0)
            self.assertAlmostEqual(reward, float(expected_reward))
        finally:
            env.close()

    def test_pendulum_seeded_reset_and_full_horizon(self):
        env = PendulumCT(dt=0.05, horizon=0.1, noise_std=0.4)
        try:
            observation, info = env.reset(seed=19)
            first_observation = observation.copy()
            first_step = env.step(np.zeros(1, dtype=np.float32))
            env.reset(seed=19)
            np.testing.assert_array_equal(env._observation(), first_observation)
            second_step = env.step(np.zeros(1, dtype=np.float32))
            np.testing.assert_array_equal(first_step[0], second_step[0])
            self.assertEqual(first_step[1:4], second_step[1:4])
            _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            np.testing.assert_array_equal(info["time"], np.ones(1, dtype=np.float32))
        finally:
            env.close()

    def test_mujoco_wrappers_preserve_base_dynamics_and_rewards(self):
        for constructor, environment in (
            (HalfCheetahCT, "HalfCheetah-v5"),
            (HopperCT, "Hopper-v5"),
            (Walker2dCT, "Walker2d-v5"),
        ):
            with self.subTest(environment=environment):
                env = constructor(dt=0.01, horizon=0.02)
                kwargs = {} if constructor is HalfCheetahCT else {"terminate_when_unhealthy": False}
                reference = gym.make(environment, max_episode_steps=2, **kwargs)
                try:
                    reference.unwrapped.model.opt.timestep = 0.01 / reference.unwrapped.frame_skip
                    observation, info = env.reset(seed=42)
                    expected, _ = reference.reset(seed=42)
                    np.testing.assert_array_equal(observation, expected)
                    if constructor is not HalfCheetahCT:
                        self.assertFalse(env.unwrapped._terminate_when_unhealthy)
                    for step in (1, 2):
                        action = np.full(env.action_space.shape, 0.1, dtype=np.float32)
                        observation, reward, terminated, truncated, info = env.step(action)
                        expected = reference.step(action)
                        np.testing.assert_array_equal(observation, expected[0])
                        self.assertEqual((reward, terminated, truncated), expected[1:4])
                        np.testing.assert_array_equal(info["time"], np.array([step / 2], dtype=np.float32))
                finally:
                    env.close()
                    reference.close()

    def test_mujoco_force_noise_is_seeded_and_reset_clears_it(self):
        for constructor in (HalfCheetahCT, HopperCT, Walker2dCT):
            with self.subTest(environment=constructor.__name__):
                env = constructor(0.01, 0.02, 0.2)
                other = constructor(0.01, 0.02, 0.2)
                try:
                    env.reset(seed=5)
                    other.reset(seed=5)
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    actual = env.step(action)
                    expected = other.step(action)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(
                        env.unwrapped.data.qfrc_applied, other.unwrapped.data.qfrc_applied
                    )
                    self.assertTrue(np.any(env.unwrapped.data.qfrc_applied != 0))
                    env.reset(seed=5)
                    self.assertTrue(np.all(env.unwrapped.data.qfrc_applied == 0))
                finally:
                    env.close()
                    other.close()

    def test_vector_factory_covers_all_benchmarks_and_legacy_factory(self):
        for environment in ("halfcheetah", "hopper", "walker2d", "pendulum"):
            with self.subTest(environment=environment):
                env = make_envs(environment, 2, 0.01, 0.02)
                try:
                    observation, info = env.reset(seed=3)
                    self.assertEqual(observation.shape[0], 2)
                    np.testing.assert_array_equal(info["time"], np.zeros((2, 1), dtype=np.float32))
                    for _ in range(2):
                        _, _, terminated, truncated, info = env.step(
                            np.zeros(env.action_space.shape, dtype=np.float32)
                        )
                    self.assertFalse(np.any(terminated))
                    self.assertTrue(np.all(truncated))
                    np.testing.assert_array_equal(info["time"], np.ones((2, 1), dtype=np.float32))
                finally:
                    env.close()
        env = make_halfcheetah_envs(1, 0.01, 0.02)
        try:
            self.assertIsInstance(env.envs[0], HalfCheetahCT)
        finally:
            env.close()

    def test_factory_rejects_unknown_environments_and_invalid_sizes(self):
        with self.assertRaises(ValueError):
            make_envs("unknown", 1, 0.01, 0.02)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                make_envs("pendulum", count, 0.01, 0.02)


if __name__ == "__main__":
    unittest.main()
