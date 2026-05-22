"""DM-Control environment construction via Shimmy + Gymnasium."""

import gymnasium as gym
from omegaconf import DictConfig


def make_env(cfg: DictConfig) -> gym.Env:
    env = gym.make(f"dm_control/{cfg.env.domain}-{cfg.env.task}-v0")
    env = gym.wrappers.FlattenObservation(env)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    return env
