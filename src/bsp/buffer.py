"""Replay buffer.

Stores transitions as numpy arrays in a ring buffer; `sample` returns a dict
of torch Tensors on the project device.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from bsp.utils import get_device


class ReplayBuffer:
    def __init__(self, cfg: DictConfig, obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.capacity = cfg.buffer.capacity
        self.device = get_device()

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.truncated = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(self.device),
            "action": torch.from_numpy(self.action[idx]).to(self.device),
            "reward": torch.from_numpy(self.reward[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "terminated": torch.from_numpy(self.terminated[idx]).to(self.device),
            "truncated": torch.from_numpy(self.truncated[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size
