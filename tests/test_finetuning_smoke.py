"""Smoke tests for the finetuning agent (steps 3-6 of this session's work).

Covers:
  * BSPAgent.act over variable-length observation histories (step 3)
  * BSPAgent.update end-to-end + emitted metrics (step 4)
  * Warm-starting the policy from a pretrained DynamicsTransformer checkpoint
    (the whole point of the config consolidation in step 5)
  * to_cpu / to_device round-trip (keeps the logstd Parameter optimizer-bound)

Runnable directly (`python tests/test_finetuning_smoke.py`) or via pytest.
"""

import numpy as np
import torch
from collections import deque
from omegaconf import OmegaConf

from bsp.finetuning.agent import BSPAgent
from bsp.pretraining.nn_modules import DynamicsPredictorModule


OBS_DIM, AC_DIM, H_MAX = 5, 3, 16

DP_TRANSFORMER = dict(
    d_model=16, embedder_hidden_dim=32, simnorm_dim=4,
    dim_feedforward=64, num_heads=2, num_layers=2,
)


def _make_cfg():
    return OmegaConf.create({
        'dp_transformer': DP_TRANSFORMER,
        'task_training': {
            'H_max': H_MAX,
            'replay_buffer': {'capacity': 1000},
            'agent': {'gamma': 0.99},
            'actor': {'actor_lr': 1e-4, 'entropy_coef': 1e-2, 'smoothness_coef': 1e-2},
            'critic': {'hidden': 64, 'depth': 2, 'critic_lr': 1e-4, 'tau': 1e-3},
        },
    })


def _make_agent():
    return BSPAgent(_make_cfg(), OBS_DIM, AC_DIM)


def _history(length):
    h = deque(maxlen=H_MAX)
    for _ in range(length):
        h.append(np.random.randn(OBS_DIM).astype(np.float32))
    return h


def test_act_variable_history_lengths():
    agent = _make_agent()
    for length in (1, 4, H_MAX):
        hist = _history(length)

        action = agent.act(hist)
        assert action.shape == (AC_DIM,), action.shape

        det = agent.act(hist, deterministic=True)
        assert det.shape == (AC_DIM,), det.shape
        # Actions are tanh-squashed, so the mean lives in [-1, 1].
        assert torch.all(det.abs() <= 1.0 + 1e-5)


def _prepped_batch(agent, B=32, L=4):
    """A batch shaped like the trainer's `_prep_agent_training_batch` output:
    (obs_seq, action, reward, next_obs_seq, done)."""
    dev = agent.device
    return (
        torch.randn(B, L, OBS_DIM, device=dev),          # obs_seq
        torch.randn(B, AC_DIM, device=dev).clamp(-1, 1),  # action
        torch.randn(B, device=dev),                       # reward
        torch.randn(B, L, OBS_DIM, device=dev),          # next_obs_seq
        torch.zeros(B, device=dev),                       # done
    )


def test_update_returns_expected_metrics():
    agent = _make_agent()
    # The actor must accept obs sequences of any length up to H_max; the critic
    # only ever sees the final transition.
    for L in (1, 4, H_MAX):
        metrics = agent.update(_prepped_batch(agent, B=32, L=L))
        expected = {'critic_loss', 'actor_loss'}
        assert expected.issubset(metrics.keys()), metrics.keys()
        assert all(np.isfinite(v) for v in metrics.values()), metrics


def test_update_consumes_trainer_prepped_sequence_batch():
    """End-to-end of the trainer's prep contract: sample length-L sequences from
    the replay buffer, build the shifted next-obs history, and feed update()."""
    agent = _make_agent()
    for _ in range(128):  # one long episode (no dones) -> valid length-L windows
        agent.replay_buffer.add(
            np.random.randn(OBS_DIM).astype(np.float32),
            np.random.randn(AC_DIM).astype(np.float32),
            np.float32(0.5),
            np.random.randn(OBS_DIM).astype(np.float32),
            False,
        )

    L = 4
    obs, actions, rewards, next_obs, dones = agent.replay_buffer.sample(32, L=L)
    next_obs_seq = torch.cat([obs[:, 1:], next_obs[:, -1:]], dim=1)
    assert next_obs_seq.shape == obs.shape

    batch = (obs, actions[:, -1], rewards[:, -1], next_obs_seq, dones[:, -1])
    metrics = agent.update(batch)
    assert {'critic_loss', 'actor_loss'}.issubset(metrics.keys()), metrics.keys()
    assert all(np.isfinite(v) for v in metrics.values()), metrics


def test_policy_loads_pretrained_dpt_checkpoint():
    """The policy's DynamicsTransformer must accept a pretraining checkpoint
    verbatim -- this is what makes the shared dp_transformer config necessary."""
    agent = _make_agent()
    pretrained = DynamicsPredictorModule(
        ac_dim=AC_DIM, obs_dim=OBS_DIM, H_max=H_MAX, **DP_TRANSFORMER,
    )
    state_dict = pretrained.dynamics_transformer.state_dict()

    # Strict load: identical architecture, no missing/unexpected keys.
    agent.actor.dynamics_transformer.load_state_dict(state_dict, strict=True)


def test_to_cpu_to_device_roundtrip():
    agent = _make_agent()
    agent.to_cpu()
    assert agent.logstd.device.type == 'cpu'
    # Optimizer must still reference the same logstd Parameter after the move.
    opt_params = {id(p) for group in agent.actor_optimizer.param_groups for p in group['params']}
    assert id(agent.logstd) in opt_params

    agent.to_device()
    # act still works after moving back.
    assert agent.act(_history(3)).shape == (AC_DIM,)


def _run_all():
    tests = [
        test_act_variable_history_lengths,
        test_update_returns_expected_metrics,
        test_update_consumes_trainer_prepped_sequence_batch,
        test_policy_loads_pretrained_dpt_checkpoint,
        test_to_cpu_to_device_roundtrip,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}", flush=True)
    print("ALL FINETUNING SMOKE TESTS PASSED", flush=True)


if __name__ == '__main__':
    _run_all()
