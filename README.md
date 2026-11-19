# CT-DDPG

A minimal implementation of continuous-time Deep Deterministic Policy Gradient
(CT-DDPG), with HalfCheetah, Hopper, Walker2d, and Pendulum examples.

Paper: [Deterministic Policy Gradient for Reinforcement Learning with Continuous
Time and State](https://arxiv.org/pdf/2509.23711) — Ziheng Cheng, Xin Guo, and
Yufei Zhang.


## Method

CT-DDPG learns a deterministic policy `pi`, a value function `V`, and an
action-value rate `q`. The centered rate is

```text
q_centered(s, a, t) = q(s, a, t) - q(s, pi(s, t), t).
```

```text
V(s_0,t_0) = sum_{k=0}^{L-2} gamma^(k h)
             [r_k - q_centered(s_k,a_k,t_k)] h
             + gamma^((L-1)h) V_target(s_{L-1},t_{L-1}).
```

The value target is updated softly, a terminal constraint anchors `V`, and the
policy maximizes `q(s, pi(s,t), t)`. Time is normalized to `[0, 1]` and supplied
to every network through sine and cosine features.

Replay sampling also preserves the source behavior: trajectory indices are
drawn before short active trajectories are discarded, so batches at the start
of an episode can contain fewer than the requested number of sequences.
Terminal states are retained for the entire run and sampled separately for
the terminal-value penalty.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The MuJoCo Python bindings are installed through `gymnasium[mujoco]`; no
separate MuJoCo download is required.

## HalfCheetah experiment

Run the default HalfCheetah configuration on one CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_gpu.sh 0
```

The first argument is the seed. The optional second argument is a fresh output
directory:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_gpu.sh 42 runs/halfcheetah-seed42
```

The launcher checks CUDA availability and refuses to reuse an existing output
path. It uses the following HalfCheetah settings:

| Setting | Value |
|---|---:|
| Environment | `HalfCheetah-v5` |
| Parallel training environments | 8 |
| Episodes | 200 |
| Horizon | 50 time units / 1,000 steps |
| Time step `h` | 0.05 |
| Dynamic force noise | 0.0 |
| Exploration-action noise | 0.1 |
| Network architecture | 3 linear layers (2 hidden), width 400, ReLU |
| Optimizer | Adam, initial learning rate `3e-4`, no weight decay |
| LR schedule | Multiply by `0.8` every 80,000 optimizer updates |
| Batch size | 256 |
| Update frequency | 1 update per environment step |
| Stored sequence length | 2–10 |
| Discount factor `gamma` | 0.8, applied as `gamma ** h` |
| Target update `tau` | 0.005 |
| Terminal constraint weight | 0.002 |

Evaluation reports the continuous-time return `sum(h * reward)`. Divide by
`h = 0.05` to compare it with the unscaled Gymnasium episode return.

Each run writes:

```text
runs/halfcheetah-seed0/
├── checkpoint.pt
├── config.json
└── history.json
```

To inspect the available launcher overrides:

```bash
python train_halfcheetah.py --help
```

## Hopper, Walker2d, and Pendulum

Select an environment with the shared training entry point:

```bash
python train.py --env hopper --device cuda --seed 0
python train.py --env walker2d --device cuda --seed 0
python train.py --env pendulum --device cuda --seed 0
```

The new task defaults follow the current
`configs/<task>/ddpg_continuous_online_seq_<task>.yml` settings in
`ddpg_continuous`:

| Setting | Hopper | Walker2d | Pendulum |
|---|---:|---:|---:|
| Time step `h` | 0.0008 | 0.0004 | 0.005 |
| Horizon | 8 | 8 | 10 |
| Dynamic noise SD | 0.1 | 1.0 | 1.0 |
| Episodes | 200 | 200 | 100 |
| Parallel training environments | 8 | 8 | 8 |
| Update frequency in environment steps | 5 | 10 | 5 |
| Hidden width | 400 | 400 | 64 |
| Actor learning rate | `3e-4` | `3e-4` | `3e-3` |
| Value / advantage-rate learning rate | `3e-4` | `3e-4` | `3e-2` |
| Replay capacity in trajectories | 1,000 | 1,000 | 100 |
| Critic stored sequence length | 2–10 | 2–10 | 10–100 |
| Actor maximum stored sequence length | 10 | 10 | 10 |
| LR decay interval / multiplier | 20,000 / 0.8 | 20,000 / 1.0 | 50,000 / 1.0 |
| Terminal constraint weight | 0.001 | 0.001 | 0.001 |
| Evaluation interval in episodes | 2 | 2 | 2 |
| Evaluation rollouts per environment | 2 | 1 | 2 |

Hopper and Walker2d use the Gymnasium v5 environments with
`terminate_when_unhealthy=False` and end at the configured horizon. Gaussian
forces are applied through MuJoCo's `qfrc_applied` at each environment step.
Pendulum uses the source repository's custom equations of motion, torque and
speed clipping, and Gaussian velocity noise with scale `noise_std * sqrt(h)`.

All tasks supply normalized time to the networks and use `sum(h * reward)` for
evaluation. `--noise-std` controls dynamics noise; `--exploration-noise` controls
the noise added to the actor's actions. The existing `--force-noise-std` name
remains available as an alias.

Individual settings can be overridden. Pendulum's value and advantage-rate
learning rates can be set with `--value-learning-rate` and
`--q-learning-rate`. For example, run Hopper with its standard time step
and zero dynamics noise:

```bash
python train.py --env hopper --device cuda --dt 0.008 --noise-std 0.0
```

The GPU shell launcher accepts the environment as its third argument:

```bash
./run_gpu.sh 0 runs/hopper-seed0 hopper
```

Each run records the resolved environment and training settings in
`config.json`. `python train_halfcheetah.py` and `./run_gpu.sh 0` continue to
launch HalfCheetah with their existing defaults.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover sequence sampling, the original `L - 1` target convention,
discounting, default hyperparameters, and finite critic/policy updates.

## Repository layout

```text
ct_ddpg/
├── algorithm.py       # CT-DDPG updates and training loop
├── buffer.py          # online episode-sequence replay
├── envs.py            # continuous-time robotic task environments
└── networks.py        # policy, value, and action-value-rate networks
run_gpu.sh             # reproducible CUDA launcher
train.py              # shared command-line training entry point
train_halfcheetah.py   # HalfCheetah entry point
requirements.txt       # runtime dependencies
tests/                 # unit tests
```

## Citation

If this code is useful in your research, please cite the paper:

```bibtex
@article{cheng2025deterministic,
  title   = {Deterministic Policy Gradient for Reinforcement Learning with Continuous Time and State},
  author  = {Cheng, Ziheng and Guo, Xin and Zhang, Yufei},
  journal = {arXiv preprint arXiv:2509.23711},
  year    = {2025}
}
```
