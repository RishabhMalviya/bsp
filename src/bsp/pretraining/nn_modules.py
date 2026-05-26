from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bsp.common.nn_modules import DynamicsTransformer


############################################
# Action Enc/Unenc
############################################
class ActionUnEmbedder(nn.Module):
    def __init__(self, ac_dim: int, d_model: int, hidden: int):
        super().__init__()

        layers = [
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ac_dim), nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)

############################################
# State Enc/Unenc
############################################
class StateUnEmbedder(nn.Module):
    def __init__(self, obs_dim: int, d_model: int, hidden: int):
        super().__init__()

        layers = [
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim), nn.ReLU()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)  


############################################
# Dynamics Predictor
############################################
class DynamicsPredictorModule(nn.Module):
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

        self.action_unembedder = ActionUnEmbedder(ac_dim=ac_dim, d_model=d_model, hidden=embedder_hidden_dim)
        self.state_unembedder = StateUnEmbedder(obs_dim=obs_dim, d_model=d_model, hidden=embedder_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        ac: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dynamics_transformer(obs, ac, state_mask, action_mask)

        predicted_states = self.state_unembedder(x[:, ::2, :])
        predicted_actions = self.action_unembedder(x[:, 1::2, :])

        return predicted_states, predicted_actions


############################################
# Policy and Value Net MLPs
############################################
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, final_activation: nn.Module | None = None, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


