import gymnasium as gym
from omegaconf import DictConfig

from bsp.common.replay_buffer import ReplayBuffer
from bsp.finetuning.agent import Agent
from bsp.utils import make_env, set_seed


class TaskSpecificTrainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, logger):
		self.cfg = cfg
		self.logger = logger

		set_seed(cfg.seed)

		self.env = make_env(cfg)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		act_dim = gym.spaces.flatdim(self.env.action_space)

		self.agent = Agent(cfg, obs_dim, act_dim)
		self.buffer = ReplayBuffer(cfg, obs_dim, act_dim)

	def eval(self, cfg: DictConfig) -> None:
		"""Evaluate an agent."""
		eval_returns = []

		for episode in range(cfg.eval.num_episodes):
			obs, _ = self.env.reset(seed=cfg.seed + episode)
			done = False
			while not done:
				action = self.agent.act(obs)
				obs, reward, terminated, truncated, info = self.env.step(action)
				eval_returns[episode] += reward
				done = terminated or truncated

		avg_return = sum(eval_returns) / len(eval_returns)
		self.logger.log({'Eval Average Return': avg_return}, step=cfg.train.total_steps)

	def train(self, cfg: DictConfig) -> None:
		"""Train an agent."""
		# Collect Data
		obs, info = self.env.reset(seed=cfg.seed)
		for _ in range(cfg.train.total_episodes):
			action = self.agent.act(obs)
			next_obs, reward, terminated, truncated, info = self.env.step(action)
			obs = next_obs
			if terminated or truncated:
				obs, info = self.env.reset()

		# Train Agent
		for update_step in range(cfg.train.utd):
			train_metrics = self.agent.update(self.buffer.sample(cfg.train.batch_size))
			self.logger.log(train_metrics, step=update_step)
