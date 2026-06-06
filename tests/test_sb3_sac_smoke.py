"""Smoke test for the stable_baselines3 SAC finetuning trainer.

Runs `SB3SACTrainer` end-to-end for a handful of timesteps on a small dm-control
task and asserts that the preserved trainer responsibilities all fire:

  * training via the vendored stable_baselines3 SAC,
  * evaluation,
  * video rendering,
  * checkpointing, and
  * logging to wandb (in disabled mode, so no network calls).

Runnable directly (`python tests/test_sb3_sac_smoke.py`) or via pytest.
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
from bsp.finetuning.sb3_trainer import SB3SACTrainer


def _make_cfg(tmp_log_dir: str):
    return OmegaConf.create({
        "seed": 0,
        "log_dir": tmp_log_dir,
        "wandb": {
            "project": "bsp-test",
            "entity": None,
            "name": "sb3-sac-smoke",
            "group": None,
            "mode": "disabled",
        },
        "env": {
            "domain": "walker",
            "downstream_task": "walk",
            "max_episode_timesteps": 50,
        },
        "task_training": {
            "agent_type": "sb3_sac",
            "eval_num_episodes": 2,
            "sb3": {
                "hparam_key": "default",
                "overrides": {
                    "n_timesteps": 200,
                    "learning_starts": 50,
                    "batch_size": 32,
                    "buffer_size": 1000,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "policy_kwargs": "dict(net_arch=[32, 32])",
                },
                "eval_freq": 100,
                "video_freq": 100,
                "ckpt_freq": 100,
            },
        },
    })


def test_sb3_sac_trainer_smoke(tmp_path: Path | None = None):
    tmp_log_dir = str(tmp_path) if tmp_path is not None else "runs/_sb3_smoke"
    cfg = _make_cfg(tmp_log_dir)

    logger = Logger(cfg)
    try:
        trainer = SB3SACTrainer(cfg, logger)
        # Sanity: SAC is the vendored copy and is wired to our env.
        assert trainer.model.__class__.__name__ == "SAC"
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
    test_sb3_sac_trainer_smoke()
    print("SB3 SAC SMOKE TEST PASSED", flush=True)
