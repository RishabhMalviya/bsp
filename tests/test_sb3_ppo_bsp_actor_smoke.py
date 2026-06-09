"""Smoke test for PPO finetuning with a BSPPolicyNet actor.

Runs :class:`bsp.finetuning.sb3_ppo_trainer.SB3PPOTrainer` end-to-end for a
handful of timesteps on a small dm-control task, but with the PPO actor swapped
for a :class:`bsp.finetuning.nn_modules.BSPPolicyNet` whose ``DynamicsTransformer``
is warm-started from a checkpoint. It asserts that:

  * PPO's policy is the custom :class:`BSPActorCriticPolicy` wrapping a
    ``BSPPolicyNet`` actor,
  * the ``DynamicsTransformer`` weights are actually loaded from the checkpoint,
  * the BSP actor params are registered with PPO's optimizer (i.e. trained),
  * training, eval, video, and checkpointing all run without raising.

wandb is forced into ``disabled`` mode so there are no network calls.

Runnable directly (``python tests/test_sb3_ppo_bsp_actor_smoke.py``) or via pytest.
"""

import os
from pathlib import Path

import torch
from gymnasium import spaces
from omegaconf import OmegaConf

# Keep wandb fully offline/local for the test.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
# osmesa (software GL) is the reliable MuJoCo render backend under WSL; egl
# core-dumps here.
os.environ["MUJOCO_GL"] = "osmesa"

from bsp.common.nn_modules import DynamicsTransformer
from bsp.common.utils import Logger, make_env
from bsp.finetuning.nn_modules import BSPPolicyNet
from bsp.finetuning.sb3_ppo_trainer import BSPActorCriticPolicy, SB3PPOTrainer


DP_TRANSFORMER = {
    "d_model": 16,
    "embedder_hidden_dim": 32,
    "simnorm_dim": 4,
    "dim_feedforward": 64,
    "num_heads": 2,
    "num_layers": 2,
}
H_MAX = 8


def _write_dynamics_transformer_checkpoint(ckpt_path: Path) -> dict:
    """Build a DynamicsTransformer with the test dims and save its state_dict.

    Returns the saved state_dict so the test can verify the policy loaded it.
    """
    env = make_env("walker", "walk", max_timesteps=50, seed=0)
    obs_dim = int(spaces.flatdim(env.observation_space))
    ac_dim = int(spaces.flatdim(env.action_space))
    env.close()

    dynamics_transformer = DynamicsTransformer(
        ac_dim=ac_dim, obs_dim=obs_dim, H_max=H_MAX, **DP_TRANSFORMER
    )
    state_dict = dynamics_transformer.state_dict()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, ckpt_path)
    return state_dict


def _make_cfg(tmp_log_dir: str, ckpt_path: Path):
    return OmegaConf.create({
        "seed": 0,
        "log_dir": tmp_log_dir,
        "wandb": {
            "project": "bsp-test",
            "entity": None,
            "name": "sb3-ppo-bsp-actor-smoke",
            "group": None,
            "mode": "disabled",
        },
        "env": {
            "domain": "walker",
            "downstream_task": "walk",
            "max_episode_timesteps": 50,
        },
        "dp_transformer": DP_TRANSFORMER,
        "task_training": {
            "agent_type": "sb3_ppo",
            "eval_num_episodes": 2,
            "H_max": H_MAX,
            "bsp_actor": {
                "enabled": True,
                "dynamics_transformer_ckpt": str(ckpt_path),
            },
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


def test_sb3_ppo_bsp_actor_smoke(tmp_path: Path | None = None):
    base = Path(tmp_path) if tmp_path is not None else Path("runs/_sb3_ppo_bsp_actor_smoke")
    ckpt_path = base / "dynamics_transformer.pth"
    saved_state = _write_dynamics_transformer_checkpoint(ckpt_path)

    cfg = _make_cfg(str(base), ckpt_path)

    logger = Logger(cfg)
    try:
        trainer = SB3PPOTrainer(cfg, logger)

        # The PPO actor is the custom BSP policy wrapping a BSPPolicyNet.
        assert trainer.model.__class__.__name__ == "PPO"
        policy = trainer.model.policy
        assert isinstance(policy, BSPActorCriticPolicy)
        assert isinstance(policy.bsp_actor, BSPPolicyNet)
        assert trainer.total_timesteps == 200

        # The DynamicsTransformer was warm-started from the checkpoint: its
        # weights must equal the saved state_dict *before* any training step.
        loaded_state = policy.bsp_actor.dynamics_transformer.state_dict()
        assert loaded_state.keys() == saved_state.keys()
        for key, saved in saved_state.items():
            assert torch.allclose(loaded_state[key].cpu(), saved.cpu()), f"mismatch in {key}"

        # The BSP actor params are registered with PPO's optimizer (will train).
        optimized_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
        bsp_params = list(policy.bsp_actor.parameters())
        assert bsp_params, "BSPPolicyNet has no parameters"
        assert all(id(p) in optimized_params for p in bsp_params), \
            "BSPPolicyNet params are not in the PPO optimizer"

        # Snapshot the transformer weights to confirm training updates them. Some
        # params (e.g. the state mask token, unused when state_mask is None) get
        # no gradient, so we require that *some* transformer weight changed.
        before = {k: v.clone() for k, v in policy.bsp_actor.dynamics_transformer.state_dict().items()}

        trainer.train()

        # Training advanced the env interaction counter.
        assert trainer.model.num_timesteps >= trainer.total_timesteps

        # The BSP actor (DynamicsTransformer) was actually optimized by PPO.
        after = policy.bsp_actor.dynamics_transformer.state_dict()
        changed = [k for k in before if not torch.allclose(before[k].cpu(), after[k].cpu())]
        assert changed, "no DynamicsTransformer weights changed during training"

        # Checkpoint was written.
        ckpt = trainer._checkpoint_path()
        assert ckpt.exists(), f"checkpoint not found at {ckpt}"

        # Eval / video produce results without raising (also exercised in train()).
        trainer._eval()
        trainer._video()
    finally:
        logger.finish()


if __name__ == "__main__":
    test_sb3_ppo_bsp_actor_smoke()
    print("SB3 PPO BSP ACTOR SMOKE TEST PASSED", flush=True)
