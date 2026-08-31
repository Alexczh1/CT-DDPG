"""HalfCheetah environment helpers for the CT-DDPG example."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


class ContinuousTimeHalfCheetah(gym.Wrapper):
    """HalfCheetah-v5 with explicit physical time and a fixed horizon."""

    def __init__(
        self,
        dt: float = 0.05,
        horizon: float = 50.0,
        force_noise_std: float = 0.0,
    ) -> None:
        if dt <= 0 or horizon <= 0:
            raise ValueError("dt and horizon must be positive")
        if force_noise_std < 0:
            raise ValueError("force_noise_std cannot be negative")
        max_steps_float = horizon / dt
        max_steps = round(max_steps_float)
        if not np.isclose(max_steps_float, max_steps):
            raise ValueError("horizon must be an integer multiple of dt")

        env = gym.make("HalfCheetah-v5", max_episode_steps=max_steps)
        super().__init__(env)
        self.dt = float(dt)
        self.horizon = float(horizon)
        self.max_steps = int(max_steps)
        self.force_noise_std = float(force_noise_std)
        self.unwrapped.model.opt.timestep = self.dt / self.unwrapped.frame_skip
        self._step_count = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._step_count = 0
        self.unwrapped.data.qfrc_applied[:] = 0.0
        info["time"] = np.asarray([0.0], dtype=np.float32)
        return observation, info

    def step(self, action):
        if self.force_noise_std:
            force = self.np_random.normal(
                0.0,
                self.force_noise_std,
                size=self.unwrapped.data.qfrc_applied.shape,
            )
            self.unwrapped.data.qfrc_applied[:] = force
        else:
            self.unwrapped.data.qfrc_applied[:] = 0.0

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        truncated = bool(truncated or self._step_count >= self.max_steps)
        info["time"] = np.asarray([self._step_count / self.max_steps], dtype=np.float32)
        return observation, reward, terminated, truncated, info

    def terminal_reward(self) -> float:
        return 0.0


class ContinuousTimeVectorEnv(SyncVectorEnv):
    @property
    def dt(self) -> float:
        return float(self.envs[0].dt)

    def terminal_reward(self) -> np.ndarray:
        return np.asarray(
            [environment.terminal_reward() for environment in self.envs],
            dtype=np.float32,
        )


def make_halfcheetah_vector_env(
    num_envs: int,
    dt: float,
    horizon: float,
    force_noise_std: float = 0.0,
) -> ContinuousTimeVectorEnv:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    def factory() -> ContinuousTimeHalfCheetah:
        return ContinuousTimeHalfCheetah(
            dt=dt,
            horizon=horizon,
            force_noise_std=force_noise_std,
        )

    factories: list[Callable[[], ContinuousTimeHalfCheetah]] = [
        factory for _ in range(num_envs)
    ]
    return ContinuousTimeVectorEnv(factories)
