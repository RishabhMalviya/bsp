import random
import time

import gymnasium as gym
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

from bsp.common.base_classes import BaseTrainer
from bsp.pretraining.agent import CuriosityAgent
from bsp.pretraining.dynamics_predictor import DynamicsPredictor
from bsp.utils import make_env, sample_seq_length, set_seed



class BodySchemaTrainer(BaseTrainer):
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg: DictConfig, logger):
		set_seed(cfg.seed)

		# Configs
		self.cfg = cfg
		self.agent_cfg = cfg.curiosity_pre_training.curiosity_agent
		self.dpt_cfg = cfg.curiosity_pre_training.dp_transformer

		# Logger
		self.logger = logger

		# Bookkeeping
		self.timestep = 0
		self.collected_episodes = 0
		self.H_max = self.cfg.curiosity_pre_training.H_max
		self.warmup = max(self.dpt_cfg.training.batch_size, self.agent_cfg.batch_size) + self.H_max

		# Environments
		self.envs = [make_env(self.cfg.env.domain, task, self.cfg.env.max_episode_timesteps, seed=self.cfg.seed) for task in self.cfg.env.pretraining_tasks]
		assert all(gym.spaces.flatdim(env.observation_space) == gym.spaces.flatdim(self.envs[0].observation_space) for env in self.envs)
		assert all(gym.spaces.flatdim(env.action_space) == gym.spaces.flatdim(self.envs[0].action_space) for env in self.envs)
		obs_dim = gym.spaces.flatdim(self.envs[0].observation_space)
		act_dim = gym.spaces.flatdim(self.envs[0].action_space)
		self.eval_env = make_env(self.cfg.env.domain, self.cfg.env.downstream_task, self.cfg.env.max_episode_timesteps, seed=self.cfg.seed, render_mode="rgb_array")

		# Trainable Components
		self.agent = CuriosityAgent(self.agent_cfg, obs_dim, act_dim)
		self.dynamics_predictor = DynamicsPredictor(self.dpt_cfg, obs_dim, act_dim, H_max=self.H_max)

	def _eval(self) -> None:
		"""
			Run a single deterministic eval episode, log return and an RGB video.
		"""
		obs, _ = self.eval_env.reset(seed=self.cfg.seed)
		frames = [self.eval_env.render()]
		episode_return = 0.0
		done = False
		while not done:
			action = self.agent.act(obs, deterministic=True)
			if isinstance(action, torch.Tensor):
				action = action.detach().cpu().numpy()
			obs, reward, terminated, truncated, _ = self.eval_env.step(action)
			frames.append(self.eval_env.render())
			episode_return += float(reward)
			done = terminated or truncated

		video = np.stack(frames).transpose(0, 3, 1, 2)  # pyright: ignore[reportCallIssue, reportArgumentType]
		self.logger.log(
			{
				'Eval Return': episode_return,
				'Eval Video': wandb.Video(video, fps=30, format='mp4'),
			},
			step=self.timestep,
		)

	def _prep_agent_training_batch(self):
		"""
			Sample a length-L sequence batch from the agent's replay buffer, compute
			the per-step intrinsic reward from the dynamics predictor, flatten over
			the sequence dimension, and return as a flat-transition batch.
		"""
		# Cap at H_max - 1 so compute_intrinsic_reward can extend to L+1 without
		# overflowing the dynamics transformer's positional embedding (sized H_max*2).
		L = min(sample_seq_length(self.H_max), self.H_max - 1)

		obs, actions, _, next_obs, dones = self.agent.replay_buffer.sample(
			batch_size=self.agent_cfg.batch_size, L=L
		)

		intrinsic_reward = self.dynamics_predictor.compute_intrinsic_reward(obs, actions, next_obs)

		return (
			obs.flatten(0, 1),
			actions.flatten(0, 1),
			intrinsic_reward.flatten(0, 1),
			next_obs.flatten(0, 1),
			dones.flatten(0, 1),
		)

	def _prep_dynamics_predictor_training_batch(self):
		"""
			Sample a length-L sequence batch from the dynamics predictor's replay
			buffer. For pure MLM training only (obs, ac) sequences are needed —
			the masks and targets are derived inside update().
		"""
		L = sample_seq_length(self.H_max)

		obs, actions, _, _, _ = self.dynamics_predictor.replay_buffer.sample(
			batch_size=self.dpt_cfg.training.batch_size, L=L
		)

		return obs, actions

	def _collect_episodes(self, env: gym.Env) -> None:
		obs, info = env.reset(seed=self.cfg.seed)
		for _ in range(self.cfg.curiosity_pre_training.num_collections_per_loop):
			for _ in range(self.cfg.env.max_episode_timesteps):
				action = self.agent.act(obs).detach().cpu().numpy()
				next_obs, reward, terminated, truncated, info = env.step(action)

				self.agent.replay_buffer.add(obs, action, reward, next_obs, terminated | truncated)
				self.dynamics_predictor.replay_buffer.add(obs, action, reward, next_obs, terminated | truncated)

				obs = next_obs
				if terminated or truncated: obs, info = env.reset()

				self.timestep += 1

			self.collected_episodes += 1

	def _train_dynamics_predictor(self) -> None:
		for _ in range(self.cfg.curiosity_pre_training.dynamics_training_iterations):
			batch = self._prep_dynamics_predictor_training_batch()
			dynamics_metrics = self.dynamics_predictor.update(batch)
			self.logger.log(dynamics_metrics, step=self.timestep)

	def _train_agent(self) -> None:
		for _ in range(self.cfg.curiosity_pre_training.curiosity_training_iterations):
			batch = self._prep_agent_training_batch()
			train_metrics = self.agent.update(batch)
			self.logger.log(train_metrics, step=self.timestep)

	def train(self) -> None:
		"""Train an agent."""

		while self.collected_episodes < self.cfg.curiosity_pre_training.total_num_episodes:
			# Collect Episodes
			collect_start = time.perf_counter()
			self._collect_episodes(random.choice(self.envs))
			self.logger.log({'time/collect_s': time.perf_counter() - collect_start}, step=self.timestep)

			# Skip training until replay buffers have at least one batch's worth of data
			if self.dynamics_predictor.replay_buffer.size < self.warmup: continue

			# Train DynamicsTransfomer
			dpt_start = time.perf_counter()
			self._train_dynamics_predictor()
			self.logger.log({'time/dpt_train_s': time.perf_counter() - dpt_start}, step=self.timestep)

			# Train Agent
			agent_start = time.perf_counter()
			self._train_agent()
			self.logger.log({'time/agent_train_s': time.perf_counter() - agent_start}, step=self.timestep)

			# Eval
			if self.collected_episodes % self.cfg.curiosity_pre_training.eval_interval == 0:
				self._eval()
