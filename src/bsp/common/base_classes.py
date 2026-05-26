import torch
import gymnasium as gym
from omegaconf import DictConfig

from bsp.utils import make_env
from bsp.common.replay_buffer import ReplayBuffer



class BaseAgent:
    def __init__(self, cfg: DictConfig,obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, cfg.replay_buffer.capacity)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError


class BaseTrainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, logger):
		self.cfg = cfg
		self.logger = logger

		self.env = make_env(cfg)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		act_dim = gym.spaces.flatdim(self.env.action_space)
            
		self.agent = BaseAgent(cfg, obs_dim, act_dim)

	def _eval(self, cfg: DictConfig) -> None:
		raise NotImplementedError
            
	def train(self, cfg: DictConfig) -> None:
		raise NotImplementedError
