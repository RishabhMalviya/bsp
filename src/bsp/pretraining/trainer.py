import gymnasium as gym
from omegaconf import DictConfig

from bsp.common.base_classes import BaseTrainer
from bsp.pretraining.agent import CuriosityAgent
from bsp.pretraining.dynamics_predictor import DynamicsPredictor
from bsp.utils import make_env, sample_seq_length, set_seed



class BodySchemaTrainer(BaseTrainer):
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, logger):
		self.cfg = cfg
		self.logger = logger

		set_seed(cfg.seed)

		self.env = make_env(cfg)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		act_dim = gym.spaces.flatdim(self.env.action_space)

		self.agent = CuriosityAgent(cfg, obs_dim, act_dim)
		self.dynamics_predictor = DynamicsPredictor(obs_dim, act_dim)

	def _eval(self, cfg: DictConfig, timestep: int) -> None:
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
		self.logger.log({'Eval Average Return': avg_return}, step=timestep)

	def train(self, cfg: DictConfig) -> None:
		"""Train an agent."""
		timestep = 0

		# Collect Trajectories
		obs, info = self.env.reset(seed=cfg.seed)
		for _ in range(cfg.curiosity_pre_training.num_collection_episodes):
			action = self.agent.act(obs)

			next_obs, reward, terminated, truncated, info = self.env.step(action)
			self.agent.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

			obs = next_obs
			if terminated or truncated:
				obs, info = self.env.reset()
			
			timestep += 1

		# Train DynamicsTransfomer
		for _ in range(cfg.curiosity_pre_training.dynamics_training_batches):
			L = sample_seq_length(cfg.curiosity_pre_training.H_max)
			dynamics_metrics = self.agent.dynamics_predictor.update(batch_size=cfg.batch_size, seq_length=L)
			self.logger.log(dynamics_metrics, step=timestep)

			# Eval
			if timestep % cfg.curiosity_pre_training.eval_interval == 0:
				self._eval(cfg, timestep)


		# Train Agent
		for _ in range(cfg.curiosity_pre_training.curiosity_training_batches):
			L = sample_seq_length(cfg.curiosity_pre_training.H_max)
			train_metrics = self.agent.update()
			self.logger.log(train_metrics, step=timestep)
		
		# Eval
		if timestep % cfg.curiosity_pre_training.eval_interval == 0:
			self._eval(cfg, timestep)
