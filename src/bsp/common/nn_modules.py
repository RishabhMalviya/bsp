import torch
from torch import nn
import torch.nn.functional as F


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, simnorm_dim: int):
		super().__init__()
		self.dim = simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


class DynamicsTransformer(nn.Module):
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


        # Masking Parameters
        self.mask_token_state  = nn.Parameter(torch.zeros(d_model))
        self.mask_token_action = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token_state,  std=0.02)
        nn.init.normal_(self.mask_token_action, std=0.02)


        # Embedders and Transformer
        self.action_embedder = ActionEmbedder(ac_dim=ac_dim, hidden=embedder_hidden_dim, d_model=d_model, simnorm_dim=simnorm_dim)
        self.state_embedder = StateEmbedder(obs_dim=obs_dim, hidden=embedder_hidden_dim, d_model=d_model, simnorm_dim=simnorm_dim)

        self.transformer = nn.TransformerEncoder(
             nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=dim_feedforward, 
                batch_first=True
            ), 
            num_layers=num_layers
        )

        self.positional_embedding = PositionalEmbedding(max_len=H_max*2, d_model=d_model)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_states = self.state_embedder(states)
        embedded_actions = self.action_embedder(actions)

        if state_mask is not None:
            embedded_states = torch.where(state_mask.unsqueeze(-1), self.mask_token_state, embedded_states)
        if action_mask is not None:
            embedded_actions = torch.where(action_mask.unsqueeze(-1), self.mask_token_action, embedded_actions)

        x = torch.stack([embedded_states, embedded_actions], dim=2).flatten(1, 2)

        x = self.positional_embedding(x)
        x = self.transformer(x)

        return x


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
