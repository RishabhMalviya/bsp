from typing import Tuple
from abc import ABC, abstractmethod

import torch


class BaseAgent(ABC):
	@abstractmethod
	def act(self, obs) -> torch.Tensor:
		raise NotImplementedError
    
	@abstractmethod
	def update(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
		raise NotImplementedError


class BaseTrainer(ABC):
	@abstractmethod
	def _eval(self) -> None:
		raise NotImplementedError

	@abstractmethod            
	def train(self) -> None:
		raise NotImplementedError
