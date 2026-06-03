"""Shape-tracing tests for the replay buffer and the trainer's batch prep.

A debugging aid: print (and assert) the shapes of the tensors flowing out of
`ReplayBuffer.sample` and `TaskSpecificTrainer._prep_agent_training_batch`.

Runnable directly (`python tests/test_shape_trace.py`) or via pytest. Use
`pytest tests/test_shape_trace.py -s` to actually see the printed shapes.
"""

import types

import numpy as np
import torch
from omegaconf import OmegaConf

from bsp.common.replay_buffer import ReplayBuffer
from bsp.finetuning.trainer import TaskSpecificTrainer


OBS_DIM, AC_DIM, CAPACITY = 5, 3, 256
H_MAX, BATCH_SIZE = 16, 32


def _filled_buffer(num_transitions: int = 128) -> ReplayBuffer:
    """A buffer holding one long episode (no dones) so every window is valid."""
    rb = ReplayBuffer(OBS_DIM, AC_DIM, CAPACITY)
    for _ in range(num_transitions):
        rb.add(
            np.random.randn(OBS_DIM).astype(np.float32),
            np.random.randn(AC_DIM).astype(np.float32),
            np.float32(0.5),
            np.random.randn(OBS_DIM).astype(np.float32),
            False,
        )
    return rb


def test_replay_buffer_sample_shapes():
    rb = _filled_buffer()
    L = 4

    obs, action, reward, next_obs, dones = rb.sample(batch_size=BATCH_SIZE, L=L)

    print(f"\n[ReplayBuffer.sample] batch_size={BATCH_SIZE}, L={L}")
    print(f"  obs:      {tuple(obs.shape)}")
    print(f"  action:   {tuple(action.shape)}")
    print(f"  reward:   {tuple(reward.shape)}")
    print(f"  next_obs: {tuple(next_obs.shape)}")
    print(f"  dones:    {tuple(dones.shape)}")

    assert obs.shape == (BATCH_SIZE, L, OBS_DIM), obs.shape
    assert action.shape == (BATCH_SIZE, L, AC_DIM), action.shape
    assert reward.shape == (BATCH_SIZE, L), reward.shape
    assert next_obs.shape == (BATCH_SIZE, L, OBS_DIM), next_obs.shape
    assert dones.shape == (BATCH_SIZE, L), dones.shape


def trace_prep_agent_training_batch_shapes():
    """Trace the shapes returned by `_prep_agent_training_batch` without paying
    for the full `TaskSpecificTrainer.__init__` (envs, agent, checkpoint load).

    We hand the unbound method a lightweight stand-in exposing only the
    attributes it touches: `.H_max`, `.cfg.task_training.batch_size`, and
    `.agent.replay_buffer`.
    """
    fake = types.SimpleNamespace(
        H_max=H_MAX,
        cfg=OmegaConf.create({'task_training': {'batch_size': BATCH_SIZE}}),
        agent=types.SimpleNamespace(replay_buffer=_filled_buffer()),
    )

    obs_seq, action_seq, reward, next_obs_seq, done = (
        TaskSpecificTrainer._prep_agent_training_batch(fake)  # pyright: ignore[reportArgumentType]
    )

    L = obs_seq.shape[1]  # L is random (sample_seq_length), so read it back.
    print(f"\n[_prep_agent_training_batch] batch_size={BATCH_SIZE}, L={L}")
    print(f"  obs_seq:      {tuple(obs_seq.shape)}")
    print(f"  action_seq:   {tuple(action_seq.shape)}")
    print(f"  reward:       {tuple(reward.shape)}")
    print(f"  next_obs_seq: {tuple(next_obs_seq.shape)}")
    print(f"  done:         {tuple(done.shape)}")

    assert obs_seq.shape == (BATCH_SIZE, L, OBS_DIM), obs_seq.shape
    assert action_seq.shape == (BATCH_SIZE, L, AC_DIM), action_seq.shape
    assert reward.shape == (BATCH_SIZE,), reward.shape
    assert next_obs_seq.shape == (BATCH_SIZE, L, OBS_DIM), next_obs_seq.shape
    assert done.shape == (BATCH_SIZE,), done.shape


def test_prep_agent_training_batch_shapes():
    trace_prep_agent_training_batch_shapes()


def _run_all():
    tests = [
        test_replay_buffer_sample_shapes,
        test_prep_agent_training_batch_shapes,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}", flush=True)
    print("ALL SHAPE TRACE TESTS PASSED", flush=True)


if __name__ == '__main__':
    _run_all()
