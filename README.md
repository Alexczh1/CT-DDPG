# CT-DDPG

A minimal implementation of CT-DDPG and a Gymnasium `HalfCheetah-v5` example.

CT-DDPG learns a time-conditioned deterministic policy `pi`, value function
`V`, and continuous-time action-value rate `q`. For a sampled sequence of
length `L`, the critic fits

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

## Run

```bash
python train_halfcheetah.py --device auto
```

The script refuses to reuse an output directory. A small end-to-end run is:

```bash
python train_halfcheetah.py \
  --device cpu \
  --num-envs 2 \
  --eval-envs 1 \
  --episodes 2 \
  --eval-every 1 \
  --horizon 0.15 \
  --batch-size 4 \
  --warmup-episodes 2 \
  --hidden-dim 32 \
  --output runs/smoke
```

The original-scale vectorization can be selected with `--num-envs 256
--eval-envs 256 --eval-every 2`. Defaults for the algorithm, including
`min_sequence_length=2`, are in `CTDDPGConfig`.

Runs save `config.json`, `history.json`, and `checkpoint.pt`.

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
```
