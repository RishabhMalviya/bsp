"""Intrinsic-reward env wrapper for curiosity pretraining with on-policy RL.

The hand-rolled curiosity trainer replaced the env reward with a
dynamics-predictor intrinsic reward at *training* time, when sampling sequences
from an off-policy replay buffer. On-policy algorithms (PPO) instead need the
reward assigned at *collection* time, transition by transition, because returns
and advantages are computed from the rollout as it is gathered.

:class:`IntrinsicRewardWrapper` does exactly that: it sits between the dm-control
env and stable_baselines3, returning the dynamics predictor's per-step intrinsic
reward in place of the env reward, and tee-ing every transition into the
dynamics predictor's replay buffer so the predictor keeps learning from fresh
on-policy data.
"""

import gymnasium as gym
import numpy as np


class IntrinsicRewardWrapper(gym.Wrapper):
    """Replace env reward with a dynamics-predictor intrinsic reward.

    For transition ``(s_l, a_l, s_{l+1})`` the reward is the prediction error of
    ``s_{l+1}`` given the causal in-episode prefix ``s_0..s_l, a_0..a_l`` (capped
    to the most recent ``H_max - 1`` steps so the positional embeddings fit). The
    original transition (with the *env* reward, which the predictor ignores) is
    also pushed to ``dpt_replay_buffer`` for masked-language-model training.

    The wrapper is per-env, so it composes with a vectorized env: each copy keeps
    its own in-episode window while sharing one dynamics predictor / buffer.
    """

    def __init__(self, env: gym.Env, dynamics_predictor, dpt_replay_buffer, H_max: int):
        super().__init__(env)
        self.dynamics_predictor = dynamics_predictor
        self.dpt_replay_buffer = dpt_replay_buffer
        self.H_max = H_max

        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._states = []
        self._actions = []
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        action = np.asarray(action, dtype=np.float32)
        next_obs_np = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated or truncated)

        # Tee the transition (with the original env reward) into the dynamics
        # predictor's replay buffer for MLM training; the predictor ignores the
        # reward field.
        self.dpt_replay_buffer.add(self._last_obs, action, float(reward), next_obs_np, float(done))

        # Extend the in-episode prefix with the current (s_l, a_l) and cap its
        # length so the predicted sequence length (T + 1) stays within H_max.
        self._states.append(self._last_obs)
        self._actions.append(action)
        if len(self._states) > self.H_max - 1:
            self._states.pop(0)
            self._actions.pop(0)

        intrinsic_reward = self.dynamics_predictor.compute_intrinsic_reward_step(
            np.stack(self._states), np.stack(self._actions), next_obs_np
        )

        self._last_obs = next_obs_np

        return next_obs, intrinsic_reward, terminated, truncated, info
