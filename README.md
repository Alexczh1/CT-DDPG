# CT-DDPG

CT-DDPG is a minimal implementation of the continuous-time deterministic
actor-critic algorithm named `DDPG_continuous_online_seq` in the original
research code. This repository contains that algorithm only, together with a
Gymnasium `HalfCheetah-v5` training example.

## What is different from standard DDPG?

The algorithm learns three time-conditioned functions:

- a deterministic policy `pi(s, t)`;
- a value function `V(s, t)`;
- a continuous-time action-value rate `q(s, a, t)`.

For a sampled sequence of length `L`, the critic minimizes the martingale
residual

```text
V(s_0, t_0) - [sum_{k=0}^{L-2} gamma^(k dt)
                 (r_k - q_centered(s_k, a_k, t_k)) dt
               + gamma^((L-1) dt) V_target(s_{L-1}, t_{L-1})].
```

By default,
`q_centered(s,a,t) = q(s,a,t) - q(s,pi(s,t),t)`. A terminal constraint anchors
the value function, and the policy maximizes `q(s, pi(s,t), t)`. The value
target is updated by Polyak averaging. Short sequences are sampled online from
both completed episodes and the active vectorized rollout.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirement.txt` is also provided as a compatibility alias for the requested
filename; it includes `requirements.txt`.

## Run HalfCheetah

The default command uses 8 synchronous environments so it is practical on a
single workstation:

```bash
python train_halfcheetah.py --device auto
```

Outputs are written to `runs/halfcheetah-seed0/`. The script refuses to reuse
an existing output path; pass a fresh path for each run:

```bash
python train_halfcheetah.py \
  --seed 1 \
  --output runs/halfcheetah-seed1
```

One `--episode` is one synchronized vector rollout. To match the main settings
from the original HalfCheetah configuration, use:

```bash
python train_halfcheetah.py \
  --num-envs 256 \
  --eval-envs 256 \
  --episodes 200 \
  --eval-every 2 \
  --batch-size 256 \
  --hidden-dim 400 \
  --layers 2 \
  --dt 0.05 \
  --horizon 50 \
  --discount-factor 0.8 \
  --output runs/halfcheetah-original-scale-seed0
```

For a fast end-to-end smoke run:

```bash
python train_halfcheetah.py \
  --device cpu \
  --num-envs 2 \
  --eval-envs 1 \
  --episodes 2 \
  --eval-every 1 \
  --eval-episodes 1 \
  --horizon 0.15 \
  --batch-size 4 \
  --warmup-episodes 2 \
  --hidden-dim 32 \
  --output runs/smoke
```

Each run stores the complete configuration in `config.json`, evaluation
history in `history.json`, and the latest model/optimizer state in
`checkpoint.pt`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover sequence sampling, terminal sampling, the martingale critic
update, and the deterministic policy update.

## Layout

```text
ct_ddpg/algorithm.py  CT-DDPG updates and training loop
ct_ddpg/buffer.py     online episode-sequence replay
ct_ddpg/envs.py       continuous-time HalfCheetah wrapper and vector env
ct_ddpg/networks.py   time-conditioned policy, value, and q-rate networks
train_halfcheetah.py  runnable example
```
