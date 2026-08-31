# CT-DDPG

A minimal implementation of continuous-time Deep Deterministic Policy Gradient
(CT-DDPG), with a reproducible `HalfCheetah-v5` example.

Paper: [Deterministic Policy Gradient for Reinforcement Learning with Continuous
Time and State](https://arxiv.org/pdf/2509.23711) — Ziheng Cheng, Xin Guo, and
Yufei Zhang.

This repository contains only the CT-DDPG algorithm, its online sequence replay
buffer, time-conditioned neural networks, a continuous-time Gymnasium wrapper,
and one GPU training launcher. It does not include the comparison algorithms
from the original research repository.

## Method

CT-DDPG learns a deterministic policy `pi`, a value function `V`, and an
action-value rate `q`. The centered rate is

```text
q_centered(s, a, t) = q(s, a, t) - q(s, pi(s, t), t).
```

Following the original `DDPG_continuous_online_seq` implementation, a sampled
sequence contains `L` stored items and contributes `L - 1` transitions to the
critic target:

```text
V(s_0,t_0) = sum_{k=0}^{L-2} gamma^(k h)
             [r_k - q_centered(s_k,a_k,t_k)] h
             + gamma^((L-1)h) V_target(s_{L-1},t_{L-1}).
```

The value target is updated softly, a terminal constraint anchors `V`, and the
policy maximizes `q(s, pi(s,t), t)`. Time is normalized to `[0, 1]` and supplied
to every network through sine and cosine features.

> **Implementation note:** the source implementation treats `0.8` as a discount
> factor and applies `gamma ** h`. This behavior is intentionally preserved. It
> corresponds to the exponential rate `-log(gamma)`, rather than applying
> `exp(-0.8 h)` directly.

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
path. The network, optimizer, vectorization, and environment defaults follow
Section 5.2 of the paper; discounting retains the source-code behavior noted
above. The main experiment settings are:

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
├── envs.py            # continuous-time HalfCheetah wrapper
└── networks.py        # policy, value, and action-value-rate networks
run_gpu.sh             # reproducible CUDA launcher
train_halfcheetah.py   # command-line training entry point
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
