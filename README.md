# CT-DDPG

A minimal implementation of CT-DDPG and a Gymnasium `HalfCheetah-v5` example.

CT-DDPG learns a time-conditioned deterministic policy `pi`, value function
`V`, and continuous-time action-value rate `q`. Matching the original
`DDPG_continuous_online_seq` implementation, a sampled sequence contains `L`
stored items and contributes `L-1` transitions to the critic target:

```text
V(s_0,t_0) = sum_{k=0}^{L-2} gamma^(k dt)
             [r_k - q_centered(s_k,a_k,t_k)] dt
             + gamma^((L-1)dt) V_target(s_{L-1},t_{L-1}),
```

where `q_centered(s,a,t) = q(s,a,t) - q(s,pi(s,t),t)`. A terminal value loss
anchors `V`, and the policy maximizes `q(s,pi(s,t),t)`. Sequences are sampled
online from completed and active vectorized rollouts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## GPU run

```bash
CUDA_VISIBLE_DEVICES=0 ./run_gpu.sh 0
```

The first argument is the random seed. An optional second argument selects the
output directory:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_gpu.sh 42 runs/halfcheetah-paper-seed42
```

The launcher verifies CUDA availability and refuses to reuse an output path.

## Training hyperparameters

The GPU launcher uses the Section 5.2 settings from the
[CT-DDPG paper](https://arxiv.org/pdf/2509.23711), except that replay and
discounting deliberately retain the previous repository's implementation:

| Setting | HalfCheetah value |
|---|---:|
| Parallel environments | 8 |
| Fully connected layers | 3 (2 hidden + output) |
| Hidden dimension | 400 |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Batch size | 256 |
| Update frequency `m` | 1 |
| Discount factor `gamma` | 0.8, applied as `gamma ** dt` |
| Soft target update `tau` | 0.005 |
| Terminal constraint weight `alpha` | 0.002 |
| Stored sequence length `L` | Sampled from 2 through 10 |
| Exploration noise standard deviation | 0.1 |
| HalfCheetah step size | 0.05 |
| Dynamic noise standard deviation | 0.0 |

The GPU script trains for 200 episodes with the standard 1,000-step
HalfCheetah horizon. Runs save `config.json`, `history.json`, and
`checkpoint.pt`.

## Test

```bash
python -m unittest discover -s tests -v
```

## Files

```text
ct_ddpg/algorithm.py  CT-DDPG update and training loop
ct_ddpg/buffer.py     online sequence replay
ct_ddpg/envs.py       continuous-time HalfCheetah wrapper
ct_ddpg/networks.py   policy, value, and q-rate networks
train_halfcheetah.py  runnable example
run_gpu.sh             paper-aligned GPU launcher
```
