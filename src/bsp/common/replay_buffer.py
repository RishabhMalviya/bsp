"""Replay buffer.

Stores transitions as numpy arrays in a ring buffer; `sample` returns a dict
of torch Tensors on the project device.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from bsp.utils import get_device


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.device = get_device()

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0


    def add(self, obs, action, reward, next_obs, dones) -> None:
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size: int, L: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if L == 1:
            idx = np.random.choice(self.size, size=batch_size, replace=False)
        else:
            if L > self.size:
                raise ValueError(f"Sequence length L={L} exceeds buffer size {self.size}")

            # Enumerate window starts in temporal order so windows can't straddle
            # the ring-buffer wrap seam (newest-to-oldest discontinuity at self.ptr).
            if self.size < self.capacity:
                starts = np.arange(self.size - L + 1)
            else:
                starts = (self.ptr + np.arange(self.capacity - L + 1)) % self.capacity

            # A sequence is invalid if any of the first H-1 transitions ended an
            # episode; terminated/truncated on the final step is allowed.
            break_idx = (starts[:, None] + np.arange(L - 1)[None, :]) % self.capacity
            invalid = self.dones[break_idx] > 0
            valid_starts = starts[~invalid.any(axis=1)]

            if len(valid_starts) < batch_size:
                raise ValueError(
                    f"Only {len(valid_starts)} valid length-{L} sequences available, "
                    f"requested batch_size={batch_size}"
                )

            chosen = np.random.choice(valid_starts, size=batch_size, replace=False)
            idx = (chosen[:, None] + np.arange(L)[None, :]) % self.capacity

        return (
            torch.from_numpy(self.obs[idx]).to(self.device),
            torch.from_numpy(self.action[idx]).to(self.device),
            torch.from_numpy(self.reward[idx]).to(self.device),
            torch.from_numpy(self.next_obs[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size
