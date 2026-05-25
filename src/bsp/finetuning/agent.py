"""Policy and value networks."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(self, cfg: DictConfig, obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.policy = MLP(obs_dim, act_dim, hidden=cfg.agent.hidden, depth=cfg.agent.depth)
        self.value = MLP(obs_dim, 1, hidden=cfg.agent.hidden, depth=cfg.agent.depth)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError