import gymnasium as gym
from omegaconf import DictConfig

from bsp.common.base_classes import BaseTrainer
from bsp.finetuning.agent import Agent
from bsp.common.utils import make_env, set_seed


class TaskSpecificTrainer(BaseTrainer):
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, logger):
		self.cfg = cfg
		self.logger = logger

		set_seed(cfg.seed)

		self.timestep = 0
		self.collected_episodes = 0

		self.env = make_env(cfg.env.domain, cfg.env.downstream_task, cfg.env.max_episode_timesteps, seed=cfg.seed)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		ac_dim = gym.spaces.flatdim(self.env.action_space)

		self.agent = Agent(cfg, obs_dim, ac_dim)

	def _eval(self) -> None:
		"""Evaluate an agent."""
		eval_returns = []

		for episode in range(self.cfg.eval.num_episodes):
			obs, _ = self.env.reset(seed=self.cfg.seed + episode)
			done = False
			while not done:
				action = self.agent.act(obs)
				obs, reward, terminated, truncated, info = self.env.step(action)
				eval_returns[episode] += reward
				done = terminated or truncated

		avg_return = sum(eval_returns) / len(eval_returns)
		self.logger.log({'Eval Average Return': avg_return}, step=self.timestep)

	def _collect_episodes(self) -> None:
		for _ in range(self.cfg.task_training.num_collections_per_loop):
			obs, _ = self.env.reset(seed=self.cfg.seed)
			for _ in range(self.cfg.task_training.max_episode_len):
				action = self.agent.act(obs)
				next_obs, reward, terminated, truncated, info = self.env.step(action)
				obs = next_obs
				if terminated or truncated:
					obs, info = self.env.reset()
					break

				self.timestep += 1
			
			self.collected_episodes += 1

	def _train_agent(self) -> None:
		for _ in range(self.cfg.task_training.curiosity_training_iterations):
			batch = self.agent.replay_buffer.sample(self.cfg.task_training.batch_size)
			train_metrics = self.agent.update(batch)
			self.logger.log(train_metrics, step=self.timestep)

	def train(self, cfg: DictConfig) -> None:
		while self._collect_episodes < cfg.task_training.total_num_episodes:
			# Collect Episodes
			with self.logger.timer('time/collect_s', step=lambda: self.timestep):
				self._collect_episodes()

			# Train Agent
			with self.logger.timer('time/agent_train_s', step=lambda: self.timestep):
				self._train_agent()

			# Eval
			if self.collected_episodes % self.cfg.task_training.eval_interval == 0:
				self._eval()
