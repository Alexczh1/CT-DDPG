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
path.

Evaluation reports the continuous-time return `sum(h * reward)`.

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

The GPU shell launcher accepts the environment as its third argument:

```bash
./run_gpu.sh 0 runs/hopper-seed0 hopper
```

Each run records its configuration in `config.json`. To view available options:

```bash
python train.py --help
```

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
