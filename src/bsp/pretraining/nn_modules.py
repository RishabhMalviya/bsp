from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bsp.common.nn_modules import DynamicsTransformer
from bsp.common.nn_modules import MLP


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
class LearnableScale(nn.Module):
    def __init__(self, dim: int, init_value: float | None = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        if init_value is not None:
            nn.init.constant_(self.scale, init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class StateUnEmbedder(nn.Module):
    def __init__(self, obs_dim: int, d_model: int, hidden: int):
        super().__init__()

        layers = [
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim)
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

        # Diagnostics: grad norms at the state/action slices of the transformer
        # output, populated by backward hooks registered in forward().
        self.last_state_slice_grad_norm: float | None = None
        self.last_action_slice_grad_norm: float | None = None

    def forward(
        self,
        obs: torch.Tensor,
        ac: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dynamics_transformer(obs, ac, state_mask, action_mask)

        state_slice = x[:, ::2, :]
        action_slice = x[:, 1::2, :]

        if state_slice.requires_grad:
            state_slice.register_hook(
                lambda g: setattr(self, 'last_state_slice_grad_norm', g.detach().norm().item())
            )
        if action_slice.requires_grad:
            action_slice.register_hook(
                lambda g: setattr(self, 'last_action_slice_grad_norm', g.detach().norm().item())
            )

        predicted_states = self.state_unembedder(state_slice)
        predicted_actions = self.action_unembedder(action_slice)

        return predicted_states, predicted_actions


############################################
# Policy and Value Net MLPs
############################################
class CuriosityPolicyNet(MLP):
    def __init__(self, obs_dim: int, ac_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__(in_dim=obs_dim, out_dim=ac_dim, hidden=hidden, depth=depth)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        ac = super().forward(obs)
        ac = torch.tanh(ac)        

        return ac

class CuriosityValueNet(MLP):
    def __init__(self, obs_dim: int, ac_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__(in_dim=obs_dim + ac_dim, out_dim=1, hidden=hidden, depth=depth)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return super().forward(obs) 
