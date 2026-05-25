import math

import torch
from torch import nn

from bsp.common.nn_modules import SimNorm
from bsp.pretraining.dynamics_predictor import DynamicsPredictor


############################################
# Action Enc/Unenc
############################################
class ActionEmbedder(nn.Module):
    def __init__(self, ac_dim: int, hidden: int, d_model: int, simnorm_dim: int):
        super().__init__()

        layers = [
            nn.Linear(ac_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_model), nn.ReLU(),
            SimNorm(simnorm_dim)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class ActionUnEmbedder(nn.Module):
    def __init__(self, ac_dim: int, d_model: int, hidden: int):
        super().__init__()

        layers = [
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ac_dim), nn.ReLU()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)

############################################
# State Enc/Unenc
############################################
class StateEmbedder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, d_model: int, simnorm_dim: int):
        super().__init__()

        layers = [
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_model), nn.ReLU(),
            SimNorm(simnorm_dim)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)    

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
# Dynamics Prediction
############################################
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


class DynamicsTransformer(DynamicsPredictor):
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
        super().__init__(obs_dim, ac_dim)

        self.action_embedder = ActionEmbedder(ac_dim=ac_dim, hidden=embedder_hidden_dim, d_model=d_model, simnorm_dim=simnorm_dim)
        self.state_embedder = StateEmbedder(obs_dim=obs_dim, hidden=embedder_hidden_dim, d_model=d_model, simnorm_dim=simnorm_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.positional_embedding = PositionalEmbedding(max_len=H_max*2, d_model=d_model)

        self.action_unembedder = ActionUnEmbedder(ac_dim=ac_dim, d_model=d_model, hidden=embedder_hidden_dim)
        self.state_unembedder = StateUnEmbedder(obs_dim=obs_dim, d_model=d_model, hidden=embedder_hidden_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_actions = self.action_embedder(actions)
        embedded_states = self.state_embedder(states)

        x = torch.stack([embedded_states, embedded_actions], dim=2).flatten(1,2)

        x = self.positional_embedding(x)
        x = self.transformer(x)

        predicted_states = self.state_unembedder(x[:, ::2, :])
        predicted_actions = self.action_unembedder(x[:, 1::2, :])

        return predicted_states, predicted_actions


############################################
# Policy and Value Net MLPs
############################################
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


