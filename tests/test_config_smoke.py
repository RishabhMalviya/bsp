"""Smoke tests for the config consolidation (step 5) and pretraining rewiring.

Covers:
  * The shared `dp_transformer` config group composes at the top level.
  * The model block was removed from `curiosity_pre_training`.
  * `task_training` exposes every key the finetuning code reads.
  * H_max matches across pretraining/finetuning (checkpoint compatibility).
  * DynamicsPredictor accepts the model config via its new `model_cfg` arg.

Runnable directly (`python tests/test_config_smoke.py`) or via pytest.
"""

from pathlib import Path

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from bsp.pretraining.dynamics_predictor import DynamicsPredictor


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'configs')

EXPECTED_MODEL_KEYS = {
    'd_model', 'embedder_hidden_dim', 'simnorm_dim',
    'dim_feedforward', 'num_heads', 'num_layers',
}


def _compose():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name='config')


def test_shared_dp_transformer_present():
    cfg = _compose()
    assert set(cfg.dp_transformer.keys()) == EXPECTED_MODEL_KEYS, set(cfg.dp_transformer.keys())


def test_model_block_removed_from_pretraining():
    cfg = _compose()
    assert 'model' not in cfg.curiosity_pre_training.dp_transformer
    # The remaining training/replay_buffer sections survive.
    assert cfg.curiosity_pre_training.dp_transformer.training.lr is not None
    assert cfg.curiosity_pre_training.dp_transformer.replay_buffer.capacity is not None


def test_task_training_keys_present():
    cfg = _compose()
    tt = cfg.task_training
    for key in (
        'H_max', 'batch_size', 'num_collections_per_loop', 'utd',
        'total_num_episodes', 'max_episode_len', 'eval_interval', 'eval_num_episodes',
        'dpt_checkpoint_path',
    ):
        assert key in tt, key

    assert tt.replay_buffer.capacity is not None
    assert tt.agent.gamma is not None
    for key in ('actor_lr', 'entropy_coef', 'smoothness_coef'):
        assert tt.actor[key] is not None, key
    for key in ('hidden', 'depth', 'critic_lr', 'tau'):
        assert tt.critic[key] is not None, key


def test_h_max_consistent_across_phases():
    cfg = _compose()
    assert cfg.curiosity_pre_training.H_max == cfg.task_training.H_max


def test_dynamics_predictor_accepts_model_cfg():
    model_cfg = OmegaConf.create(dict(
        d_model=16, embedder_hidden_dim=32, simnorm_dim=4,
        dim_feedforward=64, num_heads=2, num_layers=2,
    ))
    cfg = OmegaConf.create({'replay_buffer': {'capacity': 1000}, 'training': {'lr': 1e-4}})
    DynamicsPredictor(cfg, obs_dim=5, ac_dim=3, H_max=16, model_cfg=model_cfg)


def _run_all():
    tests = [
        test_shared_dp_transformer_present,
        test_model_block_removed_from_pretraining,
        test_task_training_keys_present,
        test_h_max_consistent_across_phases,
        test_dynamics_predictor_accepts_model_cfg,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}", flush=True)
    print("ALL CONFIG SMOKE TESTS PASSED", flush=True)


if __name__ == '__main__':
    _run_all()
