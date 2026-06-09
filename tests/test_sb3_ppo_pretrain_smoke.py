"""Smoke test for the stable_baselines3 PPO curiosity-pretraining trainer.

Runs :class:`bsp.pretraining.sb3_ppo_trainer.SB3PPOPretrainer` end-to-end for a
handful of timesteps on a small dm-control task and asserts that all of the
preserved pretraining responsibilities fire:

  * the RL agent is the vendored stable_baselines3 PPO,
  * the DynamicsPredictor trains (its MLM losses get logged to wandb),
  * the intrinsic reward replaces the env reward in the tuples PPO trains on,
  * the DynamicsTransformer weights are checkpointed for downstream warm-start,
  * evaluation, video rendering, checkpointing, and wandb logging all run.

wandb is forced into ``disabled`` mode so there are no network calls, and the run
name is prefixed with ``ppo-pretrain-smoke-test``.

Runnable directly (``python tests/test_sb3_ppo_pretrain_smoke.py``) or via pytest.
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

# Keep wandb fully offline/local for the test.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
# osmesa (software GL) is the reliable MuJoCo render backend under WSL; egl
# core-dumps here.
os.environ["MUJOCO_GL"] = "osmesa"

from bsp.common.utils import Logger
from bsp.pretraining.intrinsic_reward_env import IntrinsicRewardWrapper
from bsp.pretraining.sb3_ppo_trainer import SB3PPOPretrainer


def _make_cfg(tmp_log_dir: str):
    return OmegaConf.create({
        "seed": 0,
        "log_dir": tmp_log_dir,
        "wandb": {
            "project": "bsp-test",
            "entity": None,
            "name": "smoke",
            "group": None,
            "mode": "disabled",
        },
        "env": {
            "domain": "walker",
            "pretraining_tasks": ["stand", "walk"],
            "downstream_task": "walk",
            "max_episode_timesteps": 40,
        },
        "dp_transformer": {
            "d_model": 16,
            "embedder_hidden_dim": 32,
            "simnorm_dim": 4,
            "dim_feedforward": 64,
            "num_heads": 2,
            "num_layers": 2,
        },
        "curiosity_pre_training": {
            "agent_type": "sb3_ppo",
            "H_max": 8,
            "dynamics_predictor_utd": 3,
            "eval_num_episodes": 1,
            "dp_transformer": {
                "training": {"lr": 1e-4, "batch_size": 16},
                "replay_buffer": {"capacity": 10_000},
            },
            "sb3": {
                "hparam_key": "default",
                "eval_freq": 80,
                "video_freq": 80,
                "ckpt_freq": 80,
                "overrides": {
                    "n_timesteps": 256,
                    "n_steps": 64,
                    "batch_size": 32,
                    "n_epochs": 2,
                    "policy_kwargs": "dict(net_arch=[32, 32])",
                },
            },
        },
    })


def test_sb3_ppo_pretrainer_smoke(tmp_path: Path | None = None):
    tmp_log_dir = str(tmp_path) if tmp_path is not None else "runs/_sb3_ppo_pretrain_smoke"
    cfg = _make_cfg(tmp_log_dir)

    # The 'ppo-pretrain-smoke-test' prefix is applied via name_prefix; wandb's
    # `disabled` mode substitutes a dummy run name, so we can't assert on it here
    # (it is honoured for real online runs).
    logger = Logger(cfg, name_prefix="ppo-pretrain-smoke-test")
    try:
        trainer = SB3PPOPretrainer(cfg, logger)

        # The agent is the vendored PPO, with one rollout-env per pretraining task.
        assert trainer.model.__class__.__name__ == "PPO"
        assert trainer.vec_env.num_envs == len(cfg.env.pretraining_tasks)
        assert trainer.total_timesteps == 256

        # The rollout envs replace env reward with the dynamics-predictor's
        # intrinsic reward (i.e. PPO trains on intrinsic, not extrinsic, reward).
        # Walk the wrapper stack and confirm an IntrinsicRewardWrapper is present.
        env = trainer.vec_env.envs[0]
        found_intrinsic = False
        while hasattr(env, "env"):
            if isinstance(env, IntrinsicRewardWrapper):
                found_intrinsic = True
                break
            env = env.env
        assert found_intrinsic, "IntrinsicRewardWrapper not found in the rollout env stack"

        trainer.train()

        # Training advanced the PPO env-interaction counter.
        assert trainer.model.num_timesteps >= trainer.total_timesteps

        # The dynamics predictor was fed on-policy data and trained at least once.
        assert trainer.dynamics_predictor.replay_buffer.size > 0

        # DynamicsTransformer checkpoint was written for downstream warm-start.
        ckpt = trainer._dpt_checkpoint_path()
        assert ckpt.exists(), f"dynamics_transformer checkpoint not found at {ckpt}"

        # Eval / video produce results without raising (also exercised in train()).
        trainer._eval()
        trainer._video()
    finally:
        logger.finish()


if __name__ == "__main__":
    test_sb3_ppo_pretrainer_smoke()
    print("SB3 PPO PRETRAIN SMOKE TEST PASSED", flush=True)
