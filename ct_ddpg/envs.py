"""Continuous-time HalfCheetah environment used by the example."""

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


class HalfCheetahCT(gym.Wrapper):
    def __init__(
        self, dt: float = 0.05, horizon: float = 50.0, force_noise_std: float = 0.0
    ) -> None:
        steps = round(horizon / dt)
        if dt <= 0 or horizon <= 0 or not np.isclose(steps * dt, horizon):
            raise ValueError("horizon must be a positive integer multiple of dt")

        super().__init__(gym.make("HalfCheetah-v5", max_episode_steps=steps))
        self.steps = steps
        self.force_noise_std = force_noise_std
        self.unwrapped.model.opt.timestep = dt / self.unwrapped.frame_skip
        self.step_count = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.unwrapped.data.qfrc_applied[:] = 0
        info["time"] = np.array([0.0], dtype=np.float32)
        return observation, info

    def step(self, action):
        self.unwrapped.data.qfrc_applied[:] = self.np_random.normal(
            0, self.force_noise_std, self.unwrapped.data.qfrc_applied.shape
        )
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        truncated = truncated or self.step_count >= self.steps
        info["time"] = np.array([self.step_count / self.steps], dtype=np.float32)
        return observation, reward, terminated, truncated, info


def make_halfcheetah_envs(
    num_envs: int, dt: float, horizon: float, force_noise_std: float = 0.0
) -> SyncVectorEnv:
    def make_env() -> HalfCheetahCT:
        return HalfCheetahCT(dt, horizon, force_noise_std)

    return SyncVectorEnv([make_env for _ in range(num_envs)])
