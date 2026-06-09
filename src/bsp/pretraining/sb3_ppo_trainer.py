"""Curiosity pretraining trainer backed by the vendored stable_baselines3 PPO.

This is a drop-in replacement for the hand-rolled
:class:`bsp.pretraining.trainer.BodySchemaTrainer`'s RL agent
(:class:`bsp.pretraining.agent.CuriosityAgent`): the on-policy PPO under
``bsp.stable_baselines3`` drives rollout collection and policy/value updates,
while the project-specific pretraining responsibilities are preserved:

  1. **DynamicsPredictor training** -- the masked-language-model update runs after
     every PPO rollout, on fresh on-policy data.
  2. **Intrinsic reward** -- :class:`IntrinsicRewardWrapper` replaces the env
     reward with the dynamics predictor's per-step prediction error, so the
     transitions PPO trains on carry intrinsic (not extrinsic) reward.
  3. **DynamicsTransformer checkpointing** -- weights are saved to
     ``dynamics_transformer.pth`` so downstream finetuning can warm-start.
  4. **wandb logging** -- DynamicsPredictor metrics, PPO's internal scalars, eval
     returns and rollout videos all flow to the same wandb run.

PPO drives the env-interaction loop through ``model.learn``; a callback trains
the dynamics predictor on ``on_rollout_end`` and fires eval/video/checkpoint
hooks on a timestep schedule. A custom SB3 log writer forwards PPO's internal
scalar metrics to wandb.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
import yaml

from bsp.common.base_classes import BaseTrainer
from bsp.common.utils import get_device, make_env, sample_seq_length, set_seed
from bsp.pretraining.dynamics_predictor import DynamicsPredictor
from bsp.pretraining.intrinsic_reward_env import IntrinsicRewardWrapper
from bsp.stable_baselines3.ppo.ppo import PPO
from bsp.stable_baselines3.common.callbacks import BaseCallback
from bsp.stable_baselines3.common.logger import Logger as SB3Logger
from bsp.stable_baselines3.common.monitor import Monitor
from bsp.stable_baselines3.common.vec_env import DummyVecEnv

# Reuse the rl-zoo hyperparameter plumbing and the wandb log writer from the
# finetuning PPO trainer; they are identical for pretraining.
from bsp.finetuning.sb3_ppo_trainer import (
    PPO_YAML_PATH,
    _PPO_KWARG_KEYS,
    _WandbOutputFormat,
    _coerce_value,
)


def _load_ppo_hyperparameters(sb3_cfg) -> tuple[dict, int]:
    """Load PPO kwargs + total timesteps from ppo.yaml for the configured block.

    Mirrors :func:`bsp.finetuning.sb3_ppo_trainer.load_ppo_hyperparameters` but
    reads the ``sb3`` block from the pretraining config. ``overrides`` win over
    the yaml so smoke tests / sweeps can shrink the run without editing ppo.yaml.
    """
    hparam_key = sb3_cfg.get("hparam_key", "default")

    with open(PPO_YAML_PATH) as f:
        all_hparams = yaml.unsafe_load(f)  # rl-zoo uses yaml anchors / python tags
    block = dict(all_hparams.get(hparam_key) or all_hparams["default"])

    overrides = sb3_cfg.get("overrides", None)
    if overrides is not None:
        block.update({k: v for k, v in dict(overrides).items()})

    total_timesteps = int(float(block.get("n_timesteps", 1_000)))

    kwargs = {}
    for key in _PPO_KWARG_KEYS:
        if key in block and block[key] is not None:
            kwargs[key] = _coerce_value(key, block[key])
    kwargs.setdefault("policy", "MlpPolicy")

    return kwargs, total_timesteps


class _PretrainCallback(BaseCallback):
    """Trains the dynamics predictor each rollout and fires periodic hooks."""

    def __init__(self, trainer: "SB3PPOPretrainer", eval_freq: int, video_freq: int, ckpt_freq: int):
        super().__init__()
        self.trainer = trainer
        self.eval_freq = eval_freq
        self.video_freq = video_freq
        self.ckpt_freq = ckpt_freq
        self._last_eval = 0
        self._last_video = 0
        self._last_ckpt = 0

    def _maybe(self, freq: int, last: int, fn) -> int:
        if freq and (self.num_timesteps - last) >= freq:
            fn()
            return self.num_timesteps
        return last

    def _on_step(self) -> bool:
        # Keep the trainer's notion of "current step" in sync for wandb logging.
        self.trainer.timestep = self.num_timesteps
        self._last_eval = self._maybe(self.eval_freq, self._last_eval, self.trainer._eval)
        self._last_video = self._maybe(self.video_freq, self._last_video, self.trainer._video)
        self._last_ckpt = self._maybe(self.ckpt_freq, self._last_ckpt, self.trainer._save_dpt_checkpoint)
        return True

    def _on_rollout_end(self) -> None:
        self.trainer.timestep = self.num_timesteps
        self.trainer._train_dynamics_predictor()
        self.trainer._save_dpt_checkpoint()


class SB3PPOPretrainer(BaseTrainer):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        set_seed(cfg.seed)

        # Configs
        self.pre_cfg = cfg.curiosity_pre_training
        self.dpt_cfg = cfg.curiosity_pre_training.dp_transformer  # Training parameters
        self.dpt_model_cfg = cfg.dp_transformer  # Architecture parameters

        self.timestep = 0
        self.H_max = self.pre_cfg.H_max
        self.dynamics_predictor_utd = self.pre_cfg.dynamics_predictor_utd
        # Need at least one batch's worth of transitions (plus room for the
        # longest sampled sequence) before the dynamics predictor can update.
        self.warmup = self.dpt_cfg.training.batch_size + self.H_max

        # --- Environments -----------------------------------------------------
        # One clean env per pretraining task; dimensions must agree across tasks.
        task_envs = [
            make_env(cfg.env.domain, task, cfg.env.max_episode_timesteps, seed=cfg.seed)
            for task in cfg.env.pretraining_tasks
        ]
        assert all(
            gym.spaces.flatdim(e.observation_space) == gym.spaces.flatdim(task_envs[0].observation_space)
            for e in task_envs
        )
        assert all(
            gym.spaces.flatdim(e.action_space) == gym.spaces.flatdim(task_envs[0].action_space)
            for e in task_envs
        )
        obs_dim = gym.spaces.flatdim(task_envs[0].observation_space)
        ac_dim = gym.spaces.flatdim(task_envs[0].action_space)

        # Eval / video env runs the downstream task (rgb_array for rendering).
        self.eval_env = make_env(
            cfg.env.domain, cfg.env.downstream_task, cfg.env.max_episode_timesteps,
            seed=cfg.seed + 999, render_mode="rgb_array",
        )

        # --- Dynamics predictor ----------------------------------------------
        self.dynamics_predictor = DynamicsPredictor(
            self.dpt_cfg, obs_dim, ac_dim, H_max=self.H_max, model_cfg=self.dpt_model_cfg
        )

        # Wrap each task env so PPO sees intrinsic reward and the predictor's
        # replay buffer is fed on every step; share one predictor across copies.
        def _wrap(env: gym.Env):
            wrapped = IntrinsicRewardWrapper(
                env, self.dynamics_predictor, self.dynamics_predictor.replay_buffer, self.H_max
            )
            return Monitor(wrapped)

        self.vec_env = DummyVecEnv([(lambda e=e: _wrap(e)) for e in task_envs])

        # --- PPO agent --------------------------------------------------------
        sb3_cfg = self.pre_cfg.get("sb3", {})
        ppo_kwargs, self.total_timesteps = _load_ppo_hyperparameters(sb3_cfg)
        self.model = PPO(
            env=self.vec_env,
            seed=cfg.seed,
            device=str(get_device()),
            verbose=0,
            **ppo_kwargs,
        )
        # Route PPO's internal scalar metrics into the same wandb run.
        self.model.set_logger(SB3Logger(folder=None, output_formats=[_WandbOutputFormat(self.logger)]))

        self.eval_freq = int(sb3_cfg.get("eval_freq", 25_000))
        self.video_freq = int(sb3_cfg.get("video_freq", 50_000))
        self.ckpt_freq = int(sb3_cfg.get("ckpt_freq", 25_000))
        self.eval_num_episodes = int(self.pre_cfg.get("eval_num_episodes", 2))

    # --- DynamicsPredictor training (preserved from BodySchemaTrainer) --------

    def _prep_dynamics_predictor_training_batch(self):
        """Sample a length-L sequence batch from the predictor's replay buffer.

        For pure MLM training only (obs, ac) sequences are needed -- the masks
        and targets are derived inside ``DynamicsPredictor.update``.
        """
        L = sample_seq_length(self.H_max)
        obs, actions, _, _, _ = self.dynamics_predictor.replay_buffer.sample(
            batch_size=self.dpt_cfg.training.batch_size, L=L
        )
        return obs, actions

    def _train_dynamics_predictor(self) -> None:
        if self.dynamics_predictor.replay_buffer.size < self.warmup:
            return
        for _ in range(self.dynamics_predictor_utd):
            try:
                batch = self._prep_dynamics_predictor_training_batch()
            except ValueError:
                # Not enough valid (episode-contiguous) sequences for the sampled
                # length yet; skip this update.
                continue
            dynamics_metrics = self.dynamics_predictor.update(batch)
            self.logger.log(dynamics_metrics, step=self.timestep)

    # --- DynamicsTransformer checkpointing (preserved) ------------------------

    def _save_dpt_checkpoint(self) -> None:
        """Persist the DynamicsTransformer weights so finetuning can warm-start."""
        dynamics_transformer = self.dynamics_predictor.dynamics_predictor_module.dynamics_transformer

        ckpt_dir = Path(self.cfg.log_dir) / "checkpoints" / self.logger.run.id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "dynamics_transformer.pth"
        torch.save(dynamics_transformer.state_dict(), ckpt_path)

    def _dpt_checkpoint_path(self) -> Path:
        return Path(self.cfg.log_dir) / "checkpoints" / self.logger.run.id / "dynamics_transformer.pth"

    def _log_dpt_artifact_final(self) -> None:
        ckpt_path = self._dpt_checkpoint_path()
        if ckpt_path.exists():
            self.logger.log_artifact(ckpt_path, name="dynamics_transformer", type="model")

    # --- Eval / video ---------------------------------------------------------

    def _eval(self) -> None:
        """Run deterministic eval episodes on the downstream task, log the return.

        Uses the *env* (extrinsic) reward to gauge downstream task performance --
        independent of the intrinsic reward PPO trains on.
        """
        eval_returns = []
        for episode in range(self.eval_num_episodes):
            obs, _ = self.eval_env.reset(seed=self.cfg.seed + episode)
            episode_return = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(np.asarray(obs), deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
            eval_returns.append(episode_return)

        avg_return = sum(eval_returns) / len(eval_returns)
        self.logger.log({"Eval Average Return": avg_return}, step=self.timestep)

    def _video(self) -> None:
        """Run a single deterministic eval episode and log the RGB video."""
        obs, _ = self.eval_env.reset(seed=self.cfg.seed)
        frames = [self.eval_env.render()]
        done = False
        while not done:
            action, _ = self.model.predict(np.asarray(obs), deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            frames.append(self.eval_env.render())
            done = terminated or truncated

        video = np.stack(frames).transpose(0, 3, 1, 2)  # pyright: ignore[reportCallIssue, reportArgumentType]
        self.logger.log({"Eval Video": wandb.Video(video, fps=30, format="mp4")}, step=self.timestep)

    # --- Training entry point -------------------------------------------------

    def train(self) -> None:
        callback = _PretrainCallback(self, self.eval_freq, self.video_freq, self.ckpt_freq)
        with self.logger.timer("time/train_s", step=lambda: self.timestep):
            self.model.learn(total_timesteps=self.total_timesteps, callback=callback, log_interval=1)

        # Final eval / video / checkpoint so short runs still produce all artifacts.
        self._eval()
        self._video()
        self._save_dpt_checkpoint()
        self._log_dpt_artifact_final()
