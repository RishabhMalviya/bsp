"""Helpers: seeding, device, etc."""

import random

import wandb
import torch
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from shimmy.registration import DM_CONTROL_SUITE_ENVS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_seq_length(H_max: int, bias_k: float = 4.0) -> int:
    """Sample L in [1, H_max] via L = ceil(H_max * U^(1/k)), U ~ Uniform(0,1).

    bias_k=1.0 is uniform; bias_k>1 biases toward H_max (k=2 mild, k=4 strong);
    bias_k<1 biases toward shorter lengths.
    """
    u = np.random.uniform()
    return int(np.ceil(H_max * u ** (1.0 / bias_k)))


class Logger:
    """Thin wrapper around wandb for logging scalar metrics."""

    def __init__(self, cfg: DictConfig):
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[reportArgumentType]
        )

    def log(self, metrics: dict, step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        wandb.finish()


def make_env(cfg: DictConfig, render_mode: str | None = None) -> gym.Env:
    """DM-Control environment construction via Shimmy + Gymnasium."""

    env = gym.make(f"dm_control/{cfg.env.domain}-{cfg.env.task}-v0", render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.env.max_episode_timesteps)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    return env
