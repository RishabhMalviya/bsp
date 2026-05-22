# bsp

Reinforcement learning project built on DM-Control (via Shimmy + Gymnasium).

## Setup

```bash
pip install -e ".[dev]"
```

## Layout

- `src/bsp/` — package source
  - `env.py` — DM-Control → Gymnasium environment construction
  - `agent.py` — policy / value networks
  - `buffer.py` — replay buffer
  - `train.py` — Hydra-decorated training entry point
  - `utils.py` — helpers (seeding, logging, etc.)
- `configs/config.yaml` — Hydra config (edit hyperparameters here)
- `notebooks/` — exploratory notebooks

## Run

```bash
python -m bsp.main                          # uses configs/config.yaml
python -m bsp.main seed=1 env.task=balance  # override from CLI
python -m bsp.main --config-name=other      # pick a different config file
```
