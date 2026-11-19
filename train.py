#!/usr/bin/env python3
"""Train CT-DDPG on HalfCheetah, Hopper, Walker2d, or Pendulum."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ct_ddpg import CTDDPG, CTDDPGConfig
from ct_ddpg.envs import make_envs


# New tasks follow configs/<environment>/ddpg_continuous_online_seq_*.yml
# in ddpg_continuous. HalfCheetah retains this release's existing defaults.
ENVIRONMENT_DEFAULTS = {
    "halfcheetah": {},
    "hopper": {
        "dt": 0.0008,
        "horizon": 8.0,
        "noise_std": 0.1,
        "buffer_size": 1000,
        "update_frequency": 5,
        "eval_envs": 8,
        "eval_every": 2,
        "eval_episodes": 2,
        "lr_decay_steps": 20_000,
        "terminal_loss_weight": 0.001,
    },
    "walker2d": {
        "dt": 0.0004,
        "horizon": 8.0,
        "noise_std": 1.0,
        "buffer_size": 1000,
        "update_frequency": 10,
        "eval_envs": 8,
        "eval_every": 2,
        "lr_decay_steps": 20_000,
        "lr_decay_gamma": 1.0,
        "terminal_loss_weight": 0.001,
    },
    "pendulum": {
        "dt": 0.005,
        "horizon": 10.0,
        "noise_std": 1.0,
        "episodes": 100,
        "hidden_dim": 64,
        "learning_rate": 3e-3,
        "value_learning_rate": 3e-2,
        "q_learning_rate": 3e-2,
        "buffer_size": 100,
        "min_sequence_length": 10,
        "max_sequence_length": 100,
        "max_policy_sequence_length": 10,
        "update_frequency": 5,
        "eval_envs": 8,
        "eval_every": 2,
        "eval_episodes": 2,
        "lr_decay_steps": 50_000,
        "lr_decay_gamma": 1.0,
        "terminal_loss_weight": 0.001,
    },
}

ENVIRONMENT_NAMES = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "pendulum": "Pendulum",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENT_DEFAULTS, default="halfcheetah")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--min-sequence-length", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--max-policy-sequence-length", type=int)
    parser.add_argument("--warmup-episodes", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--policy-learning-rate", type=float)
    parser.add_argument("--value-learning-rate", type=float)
    parser.add_argument("--q-learning-rate", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--update-frequency", type=int)
    parser.add_argument("--lr-decay-steps", type=int)
    parser.add_argument("--lr-decay-gamma", type=float)
    parser.add_argument("--noise-std", "--force-noise-std", dest="force_noise_std", type=float)
    parser.add_argument("--exploration-noise", type=float)
    parser.add_argument("--discount-factor", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--terminal-loss-weight", type=float)
    args = parser.parse_args(argv)

    defaults = {
        **asdict(CTDDPGConfig()),
        "num_envs": 8,
        "eval_envs": 4,
        "episodes": 200,
        "eval_every": 10,
        "eval_episodes": 1,
        "horizon": 50.0,
        "noise_std": 0.0,
        **ENVIRONMENT_DEFAULTS[args.env],
    }
    defaults["force_noise_std"] = defaults["noise_std"]
    for name, value in vars(args).items():
        if value is None and name in defaults:
            setattr(args, name, defaults[name])

    for name in ("num_envs", "eval_envs", "episodes", "eval_every", "eval_episodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or Path("runs") / f"{args.env}-seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = CTDDPGConfig(
        dt=args.dt,
        discount_factor=args.discount_factor,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        max_policy_sequence_length=args.max_policy_sequence_length,
        warmup_episodes=args.warmup_episodes,
        update_frequency=args.update_frequency,
        exploration_noise=args.exploration_noise,
        learning_rate=args.learning_rate,
        policy_learning_rate=args.policy_learning_rate,
        value_learning_rate=args.value_learning_rate,
        q_learning_rate=args.q_learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_gamma=args.lr_decay_gamma,
        tau=args.tau,
        terminal_loss_weight=args.terminal_loss_weight,
        hidden_dim=args.hidden_dim,
    )

    env = make_envs(
        args.env, args.num_envs, args.dt, args.horizon, args.force_noise_std
    )
    try:
        eval_env = make_envs(
            args.env, args.eval_envs, args.dt, args.horizon, args.force_noise_std
        )
        try:
            output.mkdir(parents=True)
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "algorithm": "CT-DDPG",
                        "environment": ENVIRONMENT_NAMES[args.env],
                        "arguments": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                        "resolved_device": device,
                        "config": asdict(config),
                    },
                    indent=2,
                )
                + "\n"
            )
            CTDDPG(env, eval_env, device, config, args.seed, output).train(
                args.episodes, args.eval_every, args.eval_episodes
            )
        finally:
            eval_env.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
