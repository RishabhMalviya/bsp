"""Finetuning trainer backed by the vendored stable_baselines3 PPO.

This is a drop-in alternative to :class:`bsp.finetuning.trainer.TaskSpecificTrainer`
(and a sibling of :class:`bsp.finetuning.sb3_trainer.SB3SACTrainer`) that delegates
the RL algorithm (rollout collection, advantage estimation, gradient steps) to the
stable_baselines3 ``PPO`` implementation under ``bsp.stable_baselines3`` while
preserving the parts of the hand-rolled trainer the project cares about:

  1. Logging to wandb (via :class:`bsp.common.utils.Logger`).
  2. The eval / video / checkpointing logic.

PPO drives the env-interaction loop through ``model.learn``; a callback fires our
eval/video/checkpoint hooks on a timestep schedule, and a custom SB3 log writer
forwards PPO's internal scalar metrics (losses, clip_fraction, entropy, rollout
returns, ...) to the same wandb run.
"""

from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from gymnasium import spaces
from omegaconf import OmegaConf

from bsp.common.base_classes import BaseTrainer
from bsp.common.utils import get_device, make_env, set_seed
from bsp.finetuning.nn_modules import BSPPolicyNet
from bsp.stable_baselines3.ppo.ppo import PPO
from bsp.stable_baselines3.common.callbacks import BaseCallback
from bsp.stable_baselines3.common.logger import KVWriter, Logger as SB3Logger
from bsp.stable_baselines3.common.policies import ActorCriticPolicy


# Constructor kwargs we forward from the rl-zoo style ppo.yaml to PPO(...). Anything
# else in a hyperparameter block (n_timesteps, normalize, n_envs, env_wrapper, ...) is
# either handled separately or not applicable to this single-env finetuning setup.
_PPO_KWARG_KEYS = (
    "policy",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "use_sde",
    "sde_sample_freq",
    "target_kl",
    "policy_kwargs",
)

PPO_YAML_PATH = Path(__file__).resolve().parent.parent / "stable_baselines3" / "ppo" / "ppo.yaml"


def _linear_schedule(initial_value: float):
    """rl-zoo ``lin_<x>`` schedule: linearly anneal from ``initial_value`` to 0."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def _coerce_value(key: str, value):
    """Coerce an rl-zoo yaml value into what the PPO constructor expects."""
    if key in ("learning_rate", "clip_range", "clip_range_vf") and isinstance(value, str) and value.startswith("lin_"):
        return _linear_schedule(float(value[len("lin_"):]))
    if key == "policy_kwargs" and isinstance(value, str):
        # rl-zoo stores these as e.g. "dict(net_arch=[400, 300], log_std_init=-2)".
        return eval(value, {"__builtins__": {}}, {"dict": dict})  # noqa: S307 - trusted local config
    return value


def load_ppo_hyperparameters(cfg) -> tuple[dict, int]:
    """Load PPO kwargs + total timesteps from ppo.yaml for the configured env block.

    The block is selected by ``cfg.task_training.sb3.hparam_key`` (defaults to the
    ``default`` block, which mirrors stable_baselines3' own defaults). Values under
    ``cfg.task_training.sb3.overrides`` win over the yaml so smoke tests / sweeps can
    shrink the run without editing ppo.yaml.
    """
    sb3_cfg = cfg.task_training.get("sb3", {})
    hparam_key = sb3_cfg.get("hparam_key", "default")

    with open(PPO_YAML_PATH) as f:
        all_hparams = yaml.unsafe_load(f)  # rl-zoo uses yaml anchors / python tags
    block = dict(all_hparams.get(hparam_key) or all_hparams["default"])

    # Config overrides take precedence over the yaml block.
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


class BSPActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy whose action-mean network is a :class:`BSPPolicyNet`.

    PPO drives a Markovian (single-observation) interface, whereas ``BSPPolicyNet``
    wraps a ``DynamicsTransformer`` that consumes an (obs, action) *sequence*. We
    bridge the two by treating each observation as a length-1 history: the current
    observation is the only visible state and the (missing) action slot is masked
    -- the transformer fills it with its learned action-mask token. This mirrors
    exactly how :meth:`bsp.finetuning.agent.BSPAgent.act` selects an action. The
    transformer aggregates the singleton sequence into a ``(B, ac_dim)`` action
    mean that parameterises a diagonal Gaussian (with a learnable per-dim
    log-std). The critic keeps SB3's default MLP value head.

    The ``DynamicsTransformer`` weights can be warm-started from a pretraining
    checkpoint via ``dynamics_transformer_ckpt`` (a ``dynamics_transformer.pth``
    produced by the curiosity-pretraining trainer).

    Extra constructor kwargs (forwarded through PPO's ``policy_kwargs``):
      * ``bsp_net_kwargs`` -- the shared ``dp_transformer`` architecture dims
        (``d_model``, ``embedder_hidden_dim``, ``simnorm_dim``, ``num_heads``,
        ``num_layers``, ``dim_feedforward``).
      * ``H_max`` -- max history length (sizes the positional embedding); must
        match the checkpoint.
      * ``dynamics_transformer_ckpt`` -- optional path to warm-start weights.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        bsp_net_kwargs=None,
        H_max: int = 16,
        dynamics_transformer_ckpt=None,
        **kwargs,
    ):
        # Stash BSP-specific config; consumed in `_build`, which the parent
        # constructor calls before wiring up the optimizer.
        self._bsp_net_kwargs = dict(bsp_net_kwargs or {})
        self._bsp_H_max = int(H_max)
        self._dynamics_transformer_ckpt = dynamics_transformer_ckpt
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # Build the BSP actor *before* the parent builds the optimizer over
        # ``self.parameters()`` so the transformer / action-head params are
        # picked up and trained by PPO.
        obs_dim = int(spaces.flatdim(self.observation_space))
        self._action_dim = int(spaces.flatdim(self.action_space))
        self.bsp_actor = BSPPolicyNet(
            obs_dim=obs_dim,
            ac_dim=self._action_dim,
            H_max=self._bsp_H_max,
            **self._bsp_net_kwargs,
        )
        if self._dynamics_transformer_ckpt is not None:
            state_dict = torch.load(self._dynamics_transformer_ckpt, map_location=self.device)
            self.bsp_actor.dynamics_transformer.load_state_dict(state_dict)

        # Parent builds mlp_extractor, value_net, log_std (and an unused
        # action_net) and the optimizer over all params, now including bsp_actor.
        super()._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(
            bsp_net_kwargs=self._bsp_net_kwargs,
            H_max=self._bsp_H_max,
            dynamics_transformer_ckpt=self._dynamics_transformer_ckpt,
        )
        return data

    def _bsp_distribution(self, pi_features: torch.Tensor):
        """Action distribution from the BSP actor given (preprocessed) obs features.

        ``pi_features`` is the flattened ``(B, obs_dim)`` observation. We lift it to
        a length-1 ``(B, 1, obs_dim)`` history with an empty action sequence; the
        transformer masks the missing action and returns the ``(B, ac_dim)`` mean.
        """
        obs_seq = pi_features.unsqueeze(1)  # (B, 1, obs_dim)
        empty_actions = pi_features.new_zeros((pi_features.shape[0], 0, self._action_dim))
        mean_actions = self.bsp_actor(obs_seq, empty_actions)  # (B, ac_dim)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
            pi_features = features
        else:
            pi_features, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._bsp_distribution(pi_features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
            pi_features = features
        else:
            pi_features, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._bsp_distribution(pi_features)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        pi_features = super().extract_features(obs, self.pi_features_extractor)
        return self._bsp_distribution(pi_features)


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

    def __init__(self, trainer: "SB3PPOTrainer", eval_freq: int, video_freq: int, ckpt_freq: int):
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


class SB3PPOTrainer(BaseTrainer):
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

        ppo_kwargs, self.total_timesteps = load_ppo_hyperparameters(cfg)

        # Optionally swap PPO's default MlpPolicy actor for a BSPPolicyNet
        # (DynamicsTransformer + action head), warm-started from a pretraining
        # checkpoint. The transformer architecture comes from the shared
        # `dp_transformer` config so it matches the checkpoint exactly.
        bsp_cfg = cfg.task_training.get("bsp_actor", None)
        if bsp_cfg is not None and bsp_cfg.get("enabled", True):
            policy_kwargs = dict(ppo_kwargs.get("policy_kwargs") or {})
            policy_kwargs["bsp_net_kwargs"] = OmegaConf.to_container(cfg.dp_transformer, resolve=True)
            policy_kwargs["H_max"] = int(cfg.task_training.H_max)
            # Warm-start the DynamicsTransformer from the pretraining checkpoint.
            # An explicit `bsp_actor.dynamics_transformer_ckpt` wins; otherwise
            # fall back to `dpt_checkpoint_path` (set by finetune.py from the
            # preceding pretrain run). null on both trains the actor from scratch.
            ckpt = bsp_cfg.get("dynamics_transformer_ckpt", None)
            if ckpt is None:
                ckpt = cfg.task_training.get("dpt_checkpoint_path", None)
            if ckpt is not None:
                policy_kwargs["dynamics_transformer_ckpt"] = str(ckpt)
            ppo_kwargs["policy"] = BSPActorCriticPolicy
            ppo_kwargs["policy_kwargs"] = policy_kwargs

        self.model = PPO(
            env=self.env,
            seed=cfg.seed,
            device=str(get_device()),
            verbose=0,
            **ppo_kwargs,
        )

        # Route PPO's internal scalar metrics into the same wandb run.
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

        video = np.stack(frames).transpose(0, 3, 1, 2)  # pyright: ignore[reportCallIssue, reportArgumentType]
        self.logger.log({"Eval Video": wandb.Video(video, fps=30, format="mp4")}, step=self.timestep)

    def _checkpoint_path(self) -> Path:
        ckpt_dir = Path(self.cfg.log_dir) / "checkpoints" / self.logger.run.id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"ppo_{self.downstream_task}.zip"

    def _save_checkpoint(self) -> None:
        """Persist the PPO model (policy + optimizer) to disk."""
        self.model.save(self._checkpoint_path())

    def _log_checkpoint_artifact_final(self) -> None:
        ckpt_path = self._checkpoint_path()
        if ckpt_path.exists():
            self.logger.log_artifact(ckpt_path, name=f"ppo_{self.downstream_task}", type="model")

    # --- training entry point --------------------------------------------------

    def train(self) -> None:
        callback = _PeriodicCallback(self, self.eval_freq, self.video_freq, self.ckpt_freq)
        with self.logger.timer("time/train_s", step=lambda: self.timestep):
            self.model.learn(total_timesteps=self.total_timesteps, callback=callback, log_interval=1)

        # Final eval + checkpoint so short runs still produce all artifacts.
        self._eval()
        self._save_checkpoint()
        self._log_checkpoint_artifact_final()
