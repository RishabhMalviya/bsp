from collections import deque

import gymnasium as gym
import torch
from omegaconf import DictConfig

from bsp.common.base_classes import BaseTrainer
from bsp.finetuning.agent import BSPAgent
from bsp.common.utils import get_device, make_env, sample_seq_length, set_seed


device = get_device()


class TaskSpecificTrainer(BaseTrainer):
	def _load_dpt_checkpoint(self, checkpoint_path: str | None) -> None:
		"""Warm-start the policy's DynamicsTransformer from a pretrained checkpoint.

		`checkpoint_path` points at a `dynamics_transformer.pth` produced by the
		pretraining trainer (a bare DynamicsTransformer state_dict). The weights
		are loaded into the BSPPolicyNet's `dynamics_transformer`; the policy's
		action-prediction head keeps its random initialization. A falsy path skips
		the warm start (train from scratch), which is useful for ablations.
		"""
		if not checkpoint_path:
			print("[finetuning] No dpt_checkpoint_path provided; training DynamicsTransformer from scratch.")
			return

		state_dict = torch.load(checkpoint_path, map_location=device)
		self.agent.actor.dynamics_transformer.load_state_dict(state_dict)
		print(f"[finetuning] Loaded DynamicsTransformer weights from {checkpoint_path}")

	def __init__(self, cfg, logger):
		self.cfg = cfg
		self.logger = logger

		set_seed(cfg.seed)

		self.timestep = 0
		self.collected_episodes = 0

		self.H_max = cfg.task_training.H_max
		# Need at least one batch's worth of length-H_max windows before training;
		# the +H_max headroom accounts for windows lost to episode boundaries.
		self.warmup = cfg.task_training.batch_size + self.H_max

		self.env = make_env(cfg.env.domain, cfg.env.downstream_task, cfg.env.max_episode_timesteps, seed=cfg.seed)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		ac_dim = gym.spaces.flatdim(self.env.action_space)

		self.agent = BSPAgent(cfg, obs_dim, ac_dim)
		self._load_dpt_checkpoint(cfg.task_training.dpt_checkpoint_path)
		self.obs_history = deque(maxlen=cfg.task_training.H_max)

	def _eval(self) -> None:
		"""Run deterministic eval episodes and log the average return."""
		eval_returns = []

		for episode in range(self.cfg.task_training.eval_num_episodes):
			obs_history = deque(maxlen=self.cfg.task_training.H_max)
			obs, _ = self.env.reset(seed=self.cfg.seed + episode)
			episode_return = 0.0
			done = False
			while not done:
				obs_history.append(obs)
				action = self.agent.act(obs_history, deterministic=True).detach().cpu().numpy()
				obs, reward, terminated, truncated, info = self.env.step(action)
				episode_return += float(reward)
				done = terminated or truncated
			eval_returns.append(episode_return)

		avg_return = sum(eval_returns) / len(eval_returns)
		self.logger.log({'Eval Average Return': avg_return}, step=self.timestep)

	def _collect_episodes(self) -> None:
		self.agent.to_cpu()  # Keep the agent on CPU during collection to avoid GPU-CPU data transfer overhead

		for _ in range(self.cfg.task_training.num_collections_per_loop):
			self.obs_history.clear()

			obs, _ = self.env.reset(seed=self.cfg.seed)
			for _ in range(self.cfg.task_training.max_episode_len):
				self.obs_history.append(obs)
				with torch.no_grad():
					action = self.agent.act(self.obs_history).detach().cpu().numpy()
				next_obs, reward, terminated, truncated, info = self.env.step(action)

				self.agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)

				obs = next_obs
				if terminated or truncated:
					break

				self.timestep += 1

			self.collected_episodes += 1

		self.agent.to_device()

	def _prep_agent_training_batch(self):
		"""Sample a length-L (L <= H_max) sequence batch and shape it for the agent.

		The actor conditions on the full observation history, so we hand it the
		sampled `obs` sequence (ending at s_t) and a one-step-shifted history
		`next_obs_seq` (ending at s_{t+1}). The critic only needs the final
		transition, so the action/reward/done are taken at the last step.
		"""
		L = sample_seq_length(self.H_max)

		obs, actions, rewards, next_obs, dones = self.agent.replay_buffer.sample(
			batch_size=self.cfg.task_training.batch_size, L=L
		)

		# History ending at s_{t+1}: drop the oldest state, append the final next state.
		next_obs_seq = torch.cat([obs[:, 1:], next_obs[:, -1:]], dim=1)

		return (
			obs,                # obs_seq      (B, L, obs_dim)
			actions[:, -1],     # action       (B, ac_dim)
			rewards[:, -1],     # reward       (B,)
			next_obs_seq,       # next_obs_seq (B, L, obs_dim)
			dones[:, -1],       # done         (B,)
		)

	def _train_agent(self) -> None:
		for _ in range(self.cfg.task_training.utd):
			batch = self._prep_agent_training_batch()
			train_metrics = self.agent.update(batch)
			self.logger.log(train_metrics, step=self.timestep)

	def train(self) -> None:
		while self.collected_episodes < self.cfg.task_training.total_num_episodes:
			# Collect Episodes
			with self.logger.timer('time/collect_s', step=lambda: self.timestep):
				self._collect_episodes()

			# Skip training until the replay buffer holds enough length-H_max windows
			if self.agent.replay_buffer.size < self.warmup:
				continue

			# Train Agent
			with self.logger.timer('time/agent_train_s', step=lambda: self.timestep):
				self._train_agent()

			# Eval
			if self.collected_episodes % self.cfg.task_training.eval_interval == 0:
				self._eval()
