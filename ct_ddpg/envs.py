"""Continuous-time benchmark environments with normalized episode time."""

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


def _episode_steps(dt: float, horizon: float) -> int:
    if not np.isfinite(dt) or not np.isfinite(horizon) or dt <= 0 or horizon <= 0:
        raise ValueError("dt and horizon must be finite and positive")
    steps = round(horizon / dt)
    if steps < 1 or not np.isclose(steps * dt, horizon):
        raise ValueError("horizon must be a positive integer multiple of dt")
    return steps


def _validate_noise(noise_std: float) -> None:
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise standard deviation must be finite and non-negative")


class _MuJoCoCT(gym.Wrapper):
    def __init__(
        self, environment: str, dt: float, horizon: float, force_noise_std: float,
        **kwargs,
    ) -> None:
        steps = _episode_steps(dt, horizon)
        _validate_noise(force_noise_std)
        super().__init__(gym.make(environment, max_episode_steps=steps, **kwargs))
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


class HalfCheetahCT(_MuJoCoCT):
    def __init__(
        self, dt: float = 0.05, horizon: float = 50.0, force_noise_std: float = 0.0
    ) -> None:
        super().__init__("HalfCheetah-v5", dt, horizon, force_noise_std)


class HopperCT(_MuJoCoCT):
    def __init__(
        self, dt: float = 0.008, horizon: float = 8.0, force_noise_std: float = 0.0
    ) -> None:
        super().__init__(
            "Hopper-v5", dt, horizon, force_noise_std, terminate_when_unhealthy=False
        )


class Walker2dCT(_MuJoCoCT):
    def __init__(
        self, dt: float = 0.008, horizon: float = 8.0, force_noise_std: float = 0.0
    ) -> None:
        super().__init__(
            "Walker2d-v5", dt, horizon, force_noise_std, terminate_when_unhealthy=False
        )


class PendulumCT(gym.Env):
    """Source Pendulum dynamics with Gaussian velocity noise of size sigma sqrt(h)."""

    def __init__(
        self, dt: float = 0.05, horizon: float = 10.0, noise_std: float = 0.0
    ) -> None:
        self.steps = _episode_steps(dt, horizon)
        _validate_noise(noise_std)
        self.dt = dt
        self.noise_std = noise_std
        self.max_torque = 2.0
        self.max_speed = 8.0
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -self.max_torque, self.max_torque, shape=(1,), dtype=np.float32
        )
        self.state: np.ndarray | None = None
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = np.array(
            [self.np_random.uniform(-np.pi, np.pi), self.np_random.uniform(-1, 1)],
            dtype=np.float32,
        )
        self.step_count = 0
        return self._observation(), {"time": np.array([0.0], dtype=np.float32)}

    def step(self, action):
        if self.state is None:
            raise gym.error.ResetNeeded("call reset before step")
        theta, theta_dot = self.state
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        acceleration = (3 * self.g / (2 * self.l)) * np.sin(theta) + (
            3.0 / (self.m * self.l**2)
        ) * torque
        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)
        theta_dot = theta_dot + acceleration * self.dt + (
            self.noise_std * np.sqrt(self.dt) * self.np_random.normal()
        )
        theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)
        theta = theta + theta_dot * self.dt
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.state = np.array([theta, theta_dot], dtype=np.float32)
        self.step_count += 1
        info = {"time": np.array([self.step_count / self.steps], dtype=np.float32)}
        return self._observation(), float(reward), False, self.step_count >= self.steps, info

    def _observation(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


_ENVIRONMENTS = {
    "halfcheetah": HalfCheetahCT,
    "hopper": HopperCT,
    "walker2d": Walker2dCT,
    "pendulum": PendulumCT,
}


def make_envs(
    environment: str, num_envs: int, dt: float, horizon: float, noise_std: float = 0.0
) -> SyncVectorEnv:
    """Build vector environments; noise acts on MuJoCo forces or Pendulum velocity."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    try:
        environment_class = _ENVIRONMENTS[environment.lower()]
    except KeyError:
        raise ValueError(f"unknown environment: {environment}") from None

    def make_env():
        return environment_class(dt, horizon, noise_std)

    return SyncVectorEnv([make_env for _ in range(num_envs)])


def make_halfcheetah_envs(
    num_envs: int, dt: float, horizon: float, force_noise_std: float = 0.0
) -> SyncVectorEnv:
    return make_envs("halfcheetah", num_envs, dt, horizon, force_noise_std)
