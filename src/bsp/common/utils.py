"""Helpers: seeding, device, etc."""

import contextlib
import random
import time

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

    @contextlib.contextmanager
    def timer(self, key: str, step=None):
        """Time the wrapped block and log the elapsed seconds under `key`.

        `step` may be an int or a zero-arg callable; callables are resolved
        when the block exits so the logged step reflects state changes made
        inside the block (e.g. self.timestep incrementing during collection).
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            resolved_step = step() if callable(step) else step
            self.log({key: time.perf_counter() - start}, step=resolved_step) # type: ignore

    def finish(self) -> None:
        wandb.finish()


def make_env(domain: str, task: str, max_timesteps: int, seed: int, render_mode: str | None = None) -> gym.Env:
    """DM-Control environment construction via Shimmy + Gymnasium."""

    env = gym.make(f"dm_control/{domain}-{task}-v0", render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_timesteps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class LinearSchedule:
	"""Linear ramp from `initial` to `final` over `ramp_steps` advances, then hold at `final`."""

	def __init__(self, initial: float, final: float, ramp_steps: int):
		self.initial = initial
		self.final = final
		self.ramp = max(1, ramp_steps)
		self._n = 0

	@property
	def value(self) -> float:
		frac = min(1.0, self._n / self.ramp)
		return self.initial + (self.final - self.initial) * frac

	def step(self, n: int = 1) -> None:
		self._n += n
