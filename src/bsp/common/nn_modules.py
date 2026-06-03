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
		x = x.view(*shp[:-1], x.shape[-1]//self.dim, self.dim)
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
        if d_model % simnorm_dim != 0:
            raise ValueError(f"Model dimension for DP Transformer must be divisible by simnorm_dim={simnorm_dim} for the SimNorm layer to work properly. Got d_model={d_model}, simnorm_dim={simnorm_dim}.")
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

    def _compensate_for_missing_actions(self, embedded_states: torch.Tensor, embedded_actions: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """If action sequence is shorter than states sequence, pad with zeros, and mask the padded positions with the action mask token."""

        if embedded_actions.shape[1] < embedded_states.shape[1]:
            num_missing_actions = embedded_states.shape[1] - embedded_actions.shape[1]

            embedded_actions = torch.cat([embedded_actions, torch.zeros((embedded_actions.shape[0], num_missing_actions, embedded_actions.shape[2]), device=embedded_actions.device)], dim=1)

            # Mask the padded action positions
            if action_mask is None:
                action_mask = torch.zeros((embedded_actions.shape[0], embedded_actions.shape[1]), dtype=torch.bool, device=embedded_actions.device)
            else:
                 action_mask = torch.cat([action_mask, torch.ones((embedded_actions.shape[0], num_missing_actions), dtype=torch.bool, device=action_mask.device)], dim=1)
            action_mask[:, -num_missing_actions:] = True

        return embedded_actions, action_mask

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_states = self.state_embedder(states)
        embedded_actions = self.action_embedder(actions)

        # Because DP Transformer currently requires obs_seq and actions_seq to be of the same length (dim 1)
        embedded_actions, action_mask = self._compensate_for_missing_actions(embedded_states, embedded_actions, action_mask)

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
