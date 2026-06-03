from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bsp.common.nn_modules import DynamicsTransformer, MLP


############################################
# Actor
############################################
class ActionPredictionHead(nn.Module):
    def __init__(self, ac_dim: int, d_model: int, hidden: int):
        super().__init__()

        layers = [
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ac_dim)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class BSPPolicyNet(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        obs_dim: int,
        d_model: int,
        embedder_hidden_dim: int,
        simnorm_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        H_max: int
    ):
        super().__init__()

        self.dynamics_transformer = DynamicsTransformer(
            ac_dim=ac_dim,
            obs_dim=obs_dim,
            d_model=d_model,
            embedder_hidden_dim=embedder_hidden_dim,
            simnorm_dim=simnorm_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            H_max=H_max
        )

        self.action_predictor = ActionPredictionHead(ac_dim=ac_dim, d_model=d_model, hidden=embedder_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        ac: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.dynamics_transformer(obs, ac, state_mask, action_mask)

        x = self.action_predictor(x)
        predicted_actions = F.tanh(x.mean(dim=-2))
    
        return predicted_actions


############################################
# Critic
############################################
class TaskValueNet(MLP):
    def __init__(self, obs_dim: int, ac_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__(in_dim=obs_dim + ac_dim, out_dim=1, hidden=hidden, depth=depth)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return super().forward(obs) 
