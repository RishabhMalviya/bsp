from collections import deque
from pathlib import Path

import torch
import wandb
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig, open_dict

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

	def __init__(self, cfg, logger, downstream_task: str | None = None):
		self.cfg = cfg
		self.logger = logger

		set_seed(cfg.seed)

		self.downstream_task = cfg.env.downstream_task

		self.timestep = 0
		self.collected_episodes = 0

		self.H_max = cfg.task_training.H_max
		# Need at least one batch's worth of length-H_max windows before training;
		# the +H_max headroom accounts for windows lost to episode boundaries.
		self.warmup = cfg.task_training.batch_size + self.H_max

		self.env = make_env(cfg.env.domain, self.downstream_task, cfg.env.max_episode_timesteps, seed=cfg.seed)
		obs_dim = gym.spaces.flatdim(self.env.observation_space)
		ac_dim = gym.spaces.flatdim(self.env.action_space)
		self.eval_env = make_env(cfg.env.domain, self.downstream_task, cfg.env.max_episode_timesteps, seed=cfg.seed + 999, render_mode="rgb_array")

		self.agent = BSPAgent(cfg, obs_dim, ac_dim, downstream_task=self.downstream_task)
		self._load_dpt_checkpoint(cfg.task_training.dpt_checkpoint_path)

	def _eval(self) -> None:
		"""Run deterministic eval episodes and log the average return."""
		self.agent.to_cpu()  # Run the eval episode on CPU to avoid GPU-CPU data transfer overhead
		self.agent.actor.eval()  # Set the policy to eval mode (affects any dropout/batchnorm, though we don't use those in this implementation)

		eval_returns = []

		for episode in range(self.cfg.task_training.eval_num_episodes):
			obs_history = deque(maxlen=self.cfg.task_training.H_max)
			actions_history = deque(maxlen=self.cfg.task_training.H_max)

			obs, _ = self.env.reset(seed=self.cfg.seed + episode)
			episode_return = 0.0
			done = False
			while not done:
				obs_history.append(obs)
				with torch.no_grad():
					action = self.agent.act(obs_history, actions_history, deterministic=True).detach().cpu().numpy()
				actions_history.append(action)
				obs, reward, terminated, truncated, info = self.env.step(action)
				
				episode_return += float(reward)
				done = terminated or truncated
			eval_returns.append(episode_return)

		avg_return = sum(eval_returns) / len(eval_returns)
		self.logger.log({f'{self.downstream_task} Eval Average Return': avg_return}, step=self.timestep)

		self.agent.to_device()  # Move the agent back to the training device
		self.agent.actor.train()  # Set the policy back to train mode
	
	def _video(self) -> None:
		"""
			Run a single deterministic eval episode, and log RGB video.
		"""
		self.agent.to_cpu()  # Run the eval episode on CPU to avoid GPU-CPU data transfer overhead
		self.agent.actor.eval()  # Set the policy to eval mode (affects any dropout/batchnorm, though we don't use those in this implementation)

		obs_history = deque(maxlen=self.cfg.task_training.H_max)
		actions_history = deque(maxlen=self.cfg.task_training.H_max)

		obs, _ = self.eval_env.reset(seed=self.cfg.seed)
		frames = [self.eval_env.render()]
		done = False
		while not done:
			obs_history.append(obs)
			with torch.no_grad():
				action = self.agent.act(obs_history, actions_history, deterministic=True).detach().cpu().numpy()
			actions_history.append(action)
			obs, _, terminated, truncated, _ = self.eval_env.step(action)

			frames.append(self.eval_env.render())  # pyright: ignore[reportCallIssue]
			done = terminated or truncated
		video = np.stack(frames).transpose(0, 3, 1, 2)  # pyright: ignore[reportCallIssue, reportArgumentType]
		self.logger.log(
			{
				# 'Eval Return': episode_return,
				'Eval Video': wandb.Video(video, fps=30, format='mp4'),
			},
			step=self.timestep,
		)

		self.agent.to_device()  # Move the agent back to the training device
		self.agent.actor.train()  # Set the policy back to train mode

	def _collect_episodes(self) -> None:
		self.agent.to_cpu()  # Keep the agent on CPU during collection to avoid GPU-CPU data transfer overhead

		for _ in range(self.cfg.task_training.num_collections_per_loop):
			obs_history = deque(maxlen=self.cfg.task_training.H_max)
			actions_history = deque(maxlen=self.cfg.task_training.H_max)

			obs, _ = self.env.reset(seed=self.cfg.seed)
			for _ in range(self.cfg.task_training.max_episode_len):
				obs_history.append(obs)
				with torch.no_grad():
					action = self.agent.act(obs_history, actions_history).detach().cpu().numpy()
				actions_history.append(action)
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

		The actor conditions on the full observation history and action history, 
		so we output the full obs_seq, action_seq, and next_obs_seq.

		The reward and done are only needed for the update, so they are taken from the last step.
		"""
		L = sample_seq_length(self.H_max)

		obs, actions, rewards, next_obs, dones = self.agent.replay_buffer.sample(
			batch_size=self.cfg.task_training.batch_size, L=L
		)

		return (
			obs,                # obs_seq      (B, L, obs_dim)
			actions,            # action_seq   (B, L, ac_dim)
			rewards[:, -1],     # reward       (B,)
			next_obs,           # next_obs_seq (B, L, obs_dim)
			dones[:, -1],       # done         (B,)
		)

	def _train_agent(self) -> None:
		for _ in range(self.cfg.task_training.utd):
			batch = self._prep_agent_training_batch()
			train_metrics = self.agent.update(batch)
			self.logger.log(train_metrics, step=self.timestep)

	def _save_dpt_checkpoint(self) -> None:
		"""Persist the fine-tuned DynamicsTransformer weights.

		The filename carries the downstream task as a suffix so checkpoints from
		different finetuning tasks don't clobber each other. Mirrors the
		pretraining trainer: write locally every call and push a wandb artifact.
		"""
		dynamics_transformer = self.agent.actor.dynamics_transformer

		ckpt_dir = Path(self.cfg.log_dir) / 'checkpoints' / self.logger.run.id
		ckpt_dir.mkdir(parents=True, exist_ok=True)
		ckpt_path = ckpt_dir / f'dynamics_transformer_{self.downstream_task}.pth'
		torch.save(dynamics_transformer.state_dict(), ckpt_path)

	def _log_dpt_artifact_final(self):
		ckpt_dir = Path(self.cfg.log_dir) / 'checkpoints' / self.logger.run.id
		ckpt_path = ckpt_dir / f'dynamics_transformer_{self.downstream_task}.pth'

		self.logger.log_artifact(ckpt_path, name=f'dynamics_transformer_{self.downstream_task}', type='model')

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

			# Eval and Checkpointing
			if self.collected_episodes % self.cfg.task_training.eval_interval == 0:
				self._eval()
			if self.collected_episodes % self.cfg.task_training.video_interval == 0:
				self._video()
			if self.collected_episodes % self.cfg.task_training.ckpt_interval == 0:
				self._save_dpt_checkpoint()
		
		self._log_dpt_artifact_final()
