import torch
from torch import nn

from bsp.pretraining.nn_modules import DynamicsTransformer


class DynamicsPredictor(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int):
        super().__init__()

        self.obs_dim = obs_dim
        self.ac_dim = ac_dim

        self.dynamics_transformer = DynamicsTransformer(obs_dim, ac_dim)

    def forward(self, obs: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_intrinsic_reward(self, obs: torch.Tensor, ac: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        pred_next_obs = self(obs, ac)
        intrinsic_reward = F.mse_loss(pred_next_obs, next_obs, reduction='none').mean(dim=-1)
        return intrinsic_reward