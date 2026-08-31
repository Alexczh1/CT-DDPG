#!/usr/bin/env python3
"""Train CT-DDPG on Gymnasium HalfCheetah-v5."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ct_ddpg import CTDDPG, CTDDPGConfig
from ct_ddpg.envs import make_halfcheetah_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-sequence-length", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--lr-decay-steps", type=int, default=80_000)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.8)
    parser.add_argument("--force-noise-std", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path("runs") / f"halfcheetah-seed{args.seed}"
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
        batch_size=args.batch_size,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        warmup_episodes=args.warmup_episodes,
        hidden_dim=args.hidden_dim,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "algorithm": "CT-DDPG",
                "environment": "HalfCheetah-v5",
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

    env = make_halfcheetah_envs(
        args.num_envs, args.dt, args.horizon, args.force_noise_std
    )
    eval_env = make_halfcheetah_envs(
        args.eval_envs, args.dt, args.horizon, args.force_noise_std
    )
    try:
        CTDDPG(env, eval_env, device, config, args.seed, output).train(
            args.episodes, args.eval_every, args.eval_episodes
        )
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
