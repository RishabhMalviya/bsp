"""Helpers: seeding, device, etc."""

import contextlib
import json
import random
import subprocess
import time
from pathlib import Path

import wandb
import torch
import numpy as np
import gymnasium as gym
from hydra.core.hydra_config import HydraConfig
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


def _get_git_branch() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or 'no-branch'
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'no-git'


def _next_run_name(branch: str, counter_path: Path) -> str:
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    if counter_path.exists():
        try:
            counters = json.loads(counter_path.read_text())
        except json.JSONDecodeError:
            counters = {}
    idx = counters.get(branch, 0)
    counters[branch] = idx + 1
    counter_path.write_text(json.dumps(counters, indent=2))
    return f"{branch}-{idx}"


class Logger:
    """Thin wrapper around wandb for logging scalar metrics."""

    def __init__(self, cfg: DictConfig):
        name = cfg.wandb.name
        if name is None:
            name = _next_run_name(_get_git_branch(), Path(cfg.log_dir) / '.run_counter.json')

        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=name,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[reportArgumentType]
        )

        try:
            hydra_dir = Path(HydraConfig.get().runtime.output_dir) / '.hydra'
        except ValueError:
            hydra_dir = None
        if hydra_dir is not None and hydra_dir.is_dir():
            wandb.save(str(hydra_dir / '*.yaml'), base_path=str(hydra_dir.parent), policy='now')

    def log(self, metrics: dict, step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str, type: str = 'model') -> None:
        """Log a local file as a versioned wandb artifact tied to this run."""
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

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


def _safe_histogram(tensor: torch.Tensor, num_bins: int = 32, min_range: float = 1e-3) -> wandb.Histogram:
    """wandb.Histogram that doesn't crash when all values are (near-)identical."""
    data = tensor.detach().cpu().numpy()
    v_lo, v_hi = float(data.min()), float(data.max())
    half = max(0.5 * (v_hi - v_lo), 0.5 * min_range)
    mid = 0.5 * (v_lo + v_hi)
    counts, edges = np.histogram(data, bins=num_bins, range=(mid - half, mid + half))
    return wandb.Histogram(np_histogram=(counts, edges))
