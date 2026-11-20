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

## Training

Run any of the four supported tasks on a CUDA GPU:

```bash
python train.py --env halfcheetah --device cuda --seed 0
python train.py --env hopper --device cuda --seed 0
python train.py --env walker2d --device cuda --seed 0
python train.py --env pendulum --device cuda --seed 0
```

Set the discretization step `h` with `--dt` and the dynamics-noise level
with `--noise-std`. MuJoCo tasks use Gaussian force noise, while
Pendulum uses velocity noise with scale `noise_std * sqrt(h)`. Set action
exploration noise separately with `--exploration-noise`. The episode duration
`--horizon` must be a positive integer multiple of `--dt`.

Use `--device cpu` to run on CPU. To view available options:

```bash
python train.py --help
```

Each run saves `checkpoint.pt`, `config.json`, and `history.json` under
`runs/<environment>-seed<seed>/`. Evaluation reports the continuous-time return
`sum(h * reward)`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover replay sampling, environment behavior, and critic/policy updates.

## Repository layout

```text
ct_ddpg/
├── algorithm.py       # CT-DDPG updates and training loop
├── buffer.py          # online episode-sequence replay
├── envs.py            # continuous-time robotic task environments
└── networks.py        # policy, value, and action-value-rate networks
run_gpu.sh             # reproducible CUDA launcher
train.py               # shared command-line training entry point
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
