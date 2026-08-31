#!/usr/bin/env python3
"""Train CT-DDPG on Gymnasium HalfCheetah-v5."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ct_ddpg import CTDDPGConfig, DDPGContinuousOnlineSequence, NetworkConfig
from ct_ddpg.envs import make_halfcheetah_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=50.0)
    parser.add_argument("--discount-factor", type=float, default=0.8)
    parser.add_argument("--force-noise-std", type=float, default=0.0)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=10)
    parser.add_argument("--max-policy-sequence-length", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--update-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--constraint-weight", type=float, default=0.001)
    parser.add_argument(
        "--q-constraint",
        action="store_true",
        help="Penalize q(s, pi(s), t)^2 instead of explicitly centering q.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    output = args.output or Path("runs") / f"halfcheetah-seed{args.seed}"
    if output.exists():
        raise FileExistsError(
            f"output path already exists: {output}. Choose a fresh --output path."
        )

    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    network = NetworkConfig(hidden_dim=args.hidden_dim, layers=args.layers)
    config = CTDDPGConfig(
        dt=args.dt,
        discount_factor=args.discount_factor,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        max_sequence_length=args.max_sequence_length,
        max_policy_sequence_length=args.max_policy_sequence_length,
        warmup_episodes=args.warmup_episodes,
        update_frequency=args.update_frequency,
        gradient_steps=args.gradient_steps,
        critic_steps=args.critic_steps,
        exploration_noise=args.exploration_noise,
        policy_learning_rate=args.learning_rate,
        value_learning_rate=args.learning_rate,
        q_learning_rate=args.learning_rate,
        tau=args.tau,
        constraint_weight=args.constraint_weight,
        q_constraint=args.q_constraint,
        policy_network=network,
        value_network=network,
        q_network=network,
    )

    output.mkdir(parents=True)
    metadata = {
        "algorithm": "DDPG_continuous_online_seq",
        "environment": "HalfCheetah-v5",
        "device": str(device),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": asdict(config),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    train_env = make_halfcheetah_vector_env(
        args.num_envs, args.dt, args.horizon, args.force_noise_std
    )
    eval_env = make_halfcheetah_vector_env(
        args.eval_envs, args.dt, args.horizon, args.force_noise_std
    )
    try:
        algorithm = DDPGContinuousOnlineSequence(
            train_env,
            eval_env,
            device,
            config,
            seed=args.seed,
            output_dir=output,
        )
        algorithm.train(args.episodes, args.eval_every, args.eval_episodes)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
