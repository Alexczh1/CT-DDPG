# CT-DDPG

A minimal implementation of CT-DDPG and a Gymnasium `HalfCheetah-v5` example.

CT-DDPG learns a time-conditioned deterministic policy `pi`, value function
`V`, and continuous-time action-value rate `q`. For a sampled trajectory of
`L` transitions, the critic fits

```text
V(s_0,t_0) = sum_{k=0}^{L-1} exp(-beta k dt)
             [r_k - q_centered(s_k,a_k,t_k)] dt
             + exp(-beta L dt) V_target(s_L,t_L),
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

## Paper hyperparameters

The paper-specified defaults and GPU launcher follow Section 5.2 of the
[CT-DDPG paper](https://arxiv.org/pdf/2509.23711):

| Setting | HalfCheetah value |
|---|---:|
| Parallel environments | 8 |
| Fully connected layers | 3 (2 hidden + output) |
| Hidden dimension | 400 |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Batch size | 256 |
| Update frequency `m` | 1 |
| Discount rate `beta` | 0.8, applied as `exp(-beta * dt)` |
| Soft target update `tau` | 0.005 |
| Terminal constraint weight `alpha` | 0.002 |
| Trajectory length `L` | Uniform integer in `[2, 10]` |
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
