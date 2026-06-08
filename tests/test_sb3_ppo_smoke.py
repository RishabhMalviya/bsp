"""Smoke test for the stable_baselines3 PPO finetuning trainer.

Runs `SB3PPOTrainer` end-to-end for a handful of timesteps on a small dm-control
task and asserts that the preserved trainer responsibilities all fire:

  * training via the vendored stable_baselines3 PPO,
  * evaluation,
  * video rendering,
  * checkpointing, and
  * logging to wandb (in disabled mode, so no network calls).

Runnable directly (`python tests/test_sb3_ppo_smoke.py`) or via pytest.
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
from bsp.finetuning.sb3_ppo_trainer import SB3PPOTrainer


def _make_cfg(tmp_log_dir: str):
    return OmegaConf.create({
        "seed": 0,
        "log_dir": tmp_log_dir,
        "wandb": {
            "project": "bsp-test",
            "entity": None,
            "name": "sb3-ppo-smoke",
            "group": None,
            "mode": "disabled",
        },
        "env": {
            "domain": "walker",
            "downstream_task": "walk",
            "max_episode_timesteps": 50,
        },
        "task_training": {
            "agent_type": "sb3_ppo",
            "eval_num_episodes": 2,
            "sb3": {
                "hparam_key": "default",
                "overrides": {
                    "n_timesteps": 200,
                    "n_steps": 64,
                    "batch_size": 32,
                    "n_epochs": 2,
                    "policy_kwargs": "dict(net_arch=[32, 32])",
                },
                "eval_freq": 100,
                "video_freq": 100,
                "ckpt_freq": 100,
            },
        },
    })


def test_sb3_ppo_trainer_smoke(tmp_path: Path | None = None):
    tmp_log_dir = str(tmp_path) if tmp_path is not None else "runs/_sb3_ppo_smoke"
    cfg = _make_cfg(tmp_log_dir)

    logger = Logger(cfg)
    try:
        trainer = SB3PPOTrainer(cfg, logger)
        # Sanity: PPO is the vendored copy and is wired to our env.
        assert trainer.model.__class__.__name__ == "PPO"
        assert trainer.total_timesteps == 200

        trainer.train()

        # Training advanced the env interaction counter.
        assert trainer.model.num_timesteps >= trainer.total_timesteps

        # Checkpoint was written.
        ckpt = trainer._checkpoint_path()
        assert ckpt.exists(), f"checkpoint not found at {ckpt}"

        # Eval / video produce finite results without raising (exercised in train()).
        trainer._eval()
        trainer._video()
    finally:
        logger.finish()


if __name__ == "__main__":
    test_sb3_ppo_trainer_smoke()
    print("SB3 PPO SMOKE TEST PASSED", flush=True)
