"""Finetuning trainer backed by the vendored stable_baselines3 SAC.

This is a drop-in alternative to :class:`bsp.finetuning.trainer.TaskSpecificTrainer`
that delegates the RL algorithm (collection, replay, gradient steps) to the
stable_baselines3 ``SAC`` implementation under ``bsp.stable_baselines3`` while
preserving the parts of the hand-rolled trainer the project cares about:

  1. Logging to wandb (via :class:`bsp.common.utils.Logger`).
  2. The eval / video / checkpointing logic.

SAC drives the env-interaction loop through ``model.learn``; a callback fires our
eval/video/checkpoint hooks on a timestep schedule, and a custom SB3 log writer
forwards SAC's internal scalar metrics (losses, ent_coef, rollout returns, ...) to
the same wandb run.
"""

import ast
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from bsp.common.base_classes import BaseTrainer
from bsp.common.utils import get_device, make_env, set_seed
from bsp.stable_baselines3.sac.sac import SAC
from bsp.stable_baselines3.common.callbacks import BaseCallback
from bsp.stable_baselines3.common.logger import KVWriter, Logger as SB3Logger


# Constructor kwargs we forward from the rl-zoo style sac.yaml to SAC(...). Anything
# else in a hyperparameter block (n_timesteps, normalize, env_wrapper, ...) is either
# handled separately or not applicable to this single-env finetuning setup.
_SAC_KWARG_KEYS = (
    "policy",
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "tau",
    "gamma",
    "train_freq",
    "gradient_steps",
    "ent_coef",
    "target_update_interval",
    "target_entropy",
    "use_sde",
    "sde_sample_freq",
    "use_sde_at_warmup",
    "policy_kwargs",
)

SAC_YAML_PATH = Path(__file__).resolve().parent.parent / "stable_baselines3" / "sac" / "sac.yaml"


def _linear_schedule(initial_value: float):
    """rl-zoo ``lin_<x>`` schedule: linearly anneal from ``initial_value`` to 0."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def _coerce_value(key: str, value):
    """Coerce an rl-zoo yaml value into what the SAC constructor expects."""
    if key in ("learning_rate",) and isinstance(value, str) and value.startswith("lin_"):
        return _linear_schedule(float(value[len("lin_"):]))
    if key == "policy_kwargs" and isinstance(value, str):
        # rl-zoo stores these as e.g. "dict(net_arch=[400, 300], log_std_init=-3)".
        return eval(value, {"__builtins__": {}}, {"dict": dict})  # noqa: S307 - trusted local config
    if key == "train_freq" and isinstance(value, list):
        # yaml encodes tuples like [1, "episode"] as lists; SAC wants a tuple.
        return tuple(value)
    return value


def load_sac_hyperparameters(cfg) -> tuple[dict, int]:
    """Load SAC kwargs + total timesteps from sac.yaml for the configured env block.

    The block is selected by ``cfg.task_training.sb3.hparam_key`` (defaults to the
    ``default`` block, which mirrors stable_baselines3' own defaults). Values under
    ``cfg.task_training.sb3.overrides`` win over the yaml so smoke tests / sweeps can
    shrink the run without editing sac.yaml.
    """
    sb3_cfg = cfg.task_training.get("sb3", {})
    hparam_key = sb3_cfg.get("hparam_key", "default")

    with open(SAC_YAML_PATH) as f:
        all_hparams = yaml.unsafe_load(f)  # rl-zoo uses yaml anchors / python tags
    block = dict(all_hparams.get(hparam_key) or all_hparams["default"])

    # Config overrides take precedence over the yaml block.
    overrides = sb3_cfg.get("overrides", None)
    if overrides is not None:
        block.update({k: v for k, v in dict(overrides).items()})

    total_timesteps = int(float(block.get("n_timesteps", 1_000)))

    kwargs = {}
    for key in _SAC_KWARG_KEYS:
        if key in block and block[key] is not None:
            kwargs[key] = _coerce_value(key, block[key])
    kwargs.setdefault("policy", "MlpPolicy")

    return kwargs, total_timesteps


class _WandbOutputFormat(KVWriter):
    """SB3 log writer that forwards recorded scalar metrics to our wandb Logger."""

    def __init__(self, logger):
        self.logger = logger

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        scalars = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key) or ()
            if "wandb" in excluded:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars[key] = float(value)
        if scalars:
            self.logger.log(scalars, step=int(step))

    def close(self) -> None:
        pass


class _PeriodicCallback(BaseCallback):
    """Fires the trainer's eval / video / checkpoint hooks on a timestep schedule."""

    def __init__(self, trainer: "SB3SACTrainer", eval_freq: int, video_freq: int, ckpt_freq: int):
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
        self._last_ckpt = self._maybe(self.ckpt_freq, self._last_ckpt, self.trainer._save_checkpoint)
        return True


class SB3SACTrainer(BaseTrainer):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        set_seed(cfg.seed)

        self.downstream_task = cfg.env.downstream_task
        self.timestep = 0

        self.env = make_env(
            cfg.env.domain, self.downstream_task, cfg.env.max_episode_timesteps, seed=cfg.seed
        )
        self.eval_env = make_env(
            cfg.env.domain, self.downstream_task, cfg.env.max_episode_timesteps,
            seed=cfg.seed + 999, render_mode="rgb_array",
        )

        sac_kwargs, self.total_timesteps = load_sac_hyperparameters(cfg)
        self.model = SAC(
            env=self.env,
            seed=cfg.seed,
            device=str(get_device()),
            verbose=0,
            **sac_kwargs,
        )

        # Route SAC's internal scalar metrics into the same wandb run.
        self.model.set_logger(SB3Logger(folder=None, output_formats=[_WandbOutputFormat(self.logger)]))

        sb3_cfg = cfg.task_training.get("sb3", {})
        self.eval_freq = int(sb3_cfg.get("eval_freq", 5_000))
        self.video_freq = int(sb3_cfg.get("video_freq", 25_000))
        self.ckpt_freq = int(sb3_cfg.get("ckpt_freq", 25_000))
        self.eval_num_episodes = int(cfg.task_training.eval_num_episodes)

    # --- preserved trainer logic (adapted to SB3's Markov policy) --------------

    def _eval(self) -> None:
        """Run deterministic eval episodes and log the average return."""
        eval_returns = []
        for episode in range(self.eval_num_episodes):
            obs, _ = self.env.reset(seed=self.cfg.seed + episode)
            episode_return = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(np.asarray(obs), deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
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

        video = np.stack(frames).transpose(0, 3, 1, 2) # pyright: ignore[reportCallIssue, reportArgumentType]
        self.logger.log({"Eval Video": wandb.Video(video, fps=30, format="mp4")}, step=self.timestep)

    def _checkpoint_path(self) -> Path:
        ckpt_dir = Path(self.cfg.log_dir) / "checkpoints" / self.logger.run.id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"sac_{self.downstream_task}.zip"

    def _save_checkpoint(self) -> None:
        """Persist the SAC model (policy + optimizers) to disk."""
        self.model.save(self._checkpoint_path())

    def _log_checkpoint_artifact_final(self) -> None:
        ckpt_path = self._checkpoint_path()
        if ckpt_path.exists():
            self.logger.log_artifact(ckpt_path, name=f"sac_{self.downstream_task}", type="model")

    # --- training entry point --------------------------------------------------

    def train(self) -> None:
        callback = _PeriodicCallback(self, self.eval_freq, self.video_freq, self.ckpt_freq)
        with self.logger.timer("time/train_s", step=lambda: self.timestep):
            self.model.learn(total_timesteps=self.total_timesteps, callback=callback, log_interval=1)

        # Final eval + checkpoint so short runs still produce all artifacts.
        self._eval()
        self._save_checkpoint()
        self._log_checkpoint_artifact_final()
