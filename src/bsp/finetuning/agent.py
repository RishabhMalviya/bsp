"""Policy and value networks."""

import math
import itertools
from collections import deque

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions
from omegaconf import DictConfig

from bsp.common.utils import get_device, _safe_histogram, Normalizer
from bsp.common.base_classes import BaseAgent
from bsp.common.replay_buffer import ReplayBuffer
from bsp.finetuning.nn_modules import BSPPolicyNet, TaskValueNet


device = get_device()



class BSPAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, obs_dim: int, ac_dim: int, downstream_task: str | None = None):
        self.cfg = cfg
        self.device = device
        self.downstream_task = downstream_task

        self.obs_dim = obs_dim
        self.ac_dim = ac_dim

        self.tt = cfg.task_training

        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, self.tt.replay_buffer.capacity)

        # Running observation normalizer. The replay buffer stores raw observations;
        # normalization is applied at network-input time (actor and critic) using the
        # current running statistics, so the stats can keep adapting during training.
        self.normalize_obs = self.tt.normalize_obs
        self.normalizer = Normalizer(obs_dim)

        # Actor: a BSPPolicyNet wrapping the (pretrained) DynamicsTransformer. The
        # transformer dims come from the shared `dp_transformer` config so the
        # architecture matches the pretraining checkpoint exactly.
        self.actor = BSPPolicyNet(
            obs_dim=obs_dim,
            ac_dim=ac_dim,
            H_max=self.tt.H_max,
            **cfg.dp_transformer,
        ).to(device)

        self.logstd = nn.Parameter(torch.ones(ac_dim, dtype=torch.float32, device=device))
        nn.init.normal_(self.logstd, mean=-0.5, std=0.1)  # Initialize logstd to have a mean of -0.5 (std of ~0.6 in action space)
        self.LOGSTD_MIN = math.log(1e-5)
        self.LOGSTD_MAX = math.log(2.0)

        self.actor_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.actor.parameters()),
            lr=self.tt.actor.actor_lr
         )

        # Critic
        self.critic_local_1 = TaskValueNet(
            obs_dim, ac_dim, hidden=self.tt.critic.hidden, depth=self.tt.critic.depth
        ).to(device)
        self.critic_target_1 = TaskValueNet(
            obs_dim, ac_dim, hidden=self.tt.critic.hidden, depth=self.tt.critic.depth
        ).to(device)

        self.critic_target_1.load_state_dict(self.critic_local_1.state_dict())
        self.critic_target_1.eval()


        self.critic_local_2 = TaskValueNet(
            obs_dim, ac_dim, hidden=self.tt.critic.hidden, depth=self.tt.critic.depth
        ).to(device)
        self.critic_target_2 = TaskValueNet(
            obs_dim, ac_dim, hidden=self.tt.critic.hidden, depth=self.tt.critic.depth
        ).to(device)

        self.critic_target_2.load_state_dict(self.critic_local_2.state_dict())
        self.critic_target_2.eval()


        self.critic_optimizer = optim.Adam(
            itertools.chain(self.critic_local_1.parameters(), self.critic_local_2.parameters()),
            lr=self.tt.critic.critic_lr
        )

    def to_cpu(self):
        self.critic_local_1.to('cpu')
        self.critic_local_2.to('cpu')
        self.critic_target_1.to('cpu')
        self.critic_target_2.to('cpu')

        self.actor.to('cpu')
        self.logstd.data = self.logstd.data.to('cpu')  # In-place so the optimizer keeps referencing this Parameter

        self.device = 'cpu'

    def to_device(self):
        self.critic_local_1.to(device)
        self.critic_local_2.to(device)
        self.critic_target_1.to(device)
        self.critic_target_2.to(device)

        self.actor.to(device)
        self.logstd.data = self.logstd.data.to(device)

        self.device = device

    def observe(self, obs) -> None:
        """Update the running observation statistics with a freshly collected obs.

        Called during data collection only (not during eval), so the normalization
        statistics reflect the on-policy data distribution.
        """
        if self.normalize_obs:
            self.normalizer.observe(np.asarray(obs, dtype=np.float64))

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations with the running mean/std before feeding a network.

        Broadcasts over the trailing obs_dim axis, so it works for both
        (B, obs_dim) critic inputs and (B, L, obs_dim) actor sequences.
        """
        if not self.normalize_obs:
            return obs
        mean = torch.as_tensor(self.normalizer.mean, dtype=obs.dtype, device=obs.device)
        std = torch.as_tensor(np.sqrt(self.normalizer.var), dtype=obs.dtype, device=obs.device)
        return (obs - mean) / std

    def _history_deque_to_tensor(self, seq : deque, is_action_seq: bool = False) -> torch.Tensor:
        """Stack a deque of observations into a (1, L, obs_dim) sequence tensor."""
        if is_action_seq and len(seq) == 0:
            return torch.empty((1, 0, self.ac_dim), dtype=torch.float32, device=self.device)
        else:
            return torch.as_tensor(np.stack(list(seq), axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_action_distribution(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor, temperature: float = 1.0) -> distributions.Normal:
        """Run the policy over an observation sequence and return the action distribution.

        `obs_seq` is (B, L, obs_dim). Actions are unknown at action-selection time,
        so every action slot is masked (replaced by the transformer's learned action
        mask token) and the state slots are all left visible. The policy aggregates
        the sequence into a single (B, ac_dim) action mean.
        """
        action_mean = self.actor(self._normalize_obs(obs_seq), actions_seq)

        action_logstd = self.logstd.expand_as(action_mean)
        clipped_logstd = self.LOGSTD_MIN + 0.5 * (self.LOGSTD_MAX - self.LOGSTD_MIN) * (torch.tanh(action_logstd) + 1)
        action_std = torch.exp(clipped_logstd) * float(temperature)

        return distributions.Normal(action_mean, action_std)

    def _sample_action_from(self, action_distribution: distributions.Distribution) -> torch.Tensor:
        return torch.clamp(input=action_distribution.rsample(), min=-1.0, max=1.0)

    def act(self, obs_seq: deque | torch.Tensor, action_seq: deque | torch.Tensor, deterministic: bool = False, temperature: float = 1.0) -> torch.Tensor:
        """
        Select an action given the recent observation history.
        """
        if isinstance(obs_seq, deque):
            if len(obs_seq) == 0:
                raise ValueError("Observation history is empty. Cannot select action without any observations.")
            obs_seq = self._history_deque_to_tensor(obs_seq)
        if isinstance(action_seq, deque):
            action_seq = self._history_deque_to_tensor(action_seq, is_action_seq=True)
        
        action_dist = self._get_action_distribution(obs_seq, action_seq, temperature)  # pyright: ignore[reportArgumentType]

        action = action_dist.mean if deterministic else self._sample_action_from(action_dist)
        return action.squeeze(0)  # If dim 0 (batch dim) is 1, this will remove it. Otherwise, it will do nothing.

    def update(self, batch) -> dict[str, float]:
        """Update the actor and critic from a batch of transitions.

        The batch is produced by the trainer's prep step:
          obs_seq      (B, L, obs_dim)  observation history ending at s_t
          action_seq   (B, L, ac_dim)   action taken at s_t
          reward       (B,)             reward for (s_t, a_t)
          next_obs_seq (B, L, obs_dim)  observation history ending at s_{t+1}
          done         (B,)             episode-termination flag for the step

        The actor conditions on the full observation history (obs_seq / action_seq, length L <= H_max).
        The critic operates on the single current/next transition, taken as the last element of each sequence. 
        
        The critic is trained with a standard one-step TD target.

        The actor is trained to maximize the critic's value while encouraging exploration (entropy bonus)
        and smooth control signals (smoothness penalty), mirroring the pretraining curiosity agent.
        """
        def _soft_update(local_model, target_model, tau):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        # Extract the relevant tensors from the batch and shape them for the actor and critic updates.
        obs_seq, action_seq, reward, next_obs_seq, done = batch

        action = action_seq[:, -1, :]      # action taken at s_t (B, ac_dim)
        # Critic inputs are normalized with the running stats. The actor normalizes
        # its full obs_seq / next_obs_seq internally (see _get_action_distribution),
        # so the whole agent operates in a consistent normalized observation space.
        obs = self._normalize_obs(obs_seq[:, -1, :])            # current state s_{t} (B, obs_dim)
        next_obs = self._normalize_obs(next_obs_seq[:, -1, :])  # next state s_{t+1}  (B, obs_dim)


        metrics = {}


        # Train Critic
        with torch.no_grad():
            next_action_mean = self.act(obs_seq = next_obs_seq, action_seq = action_seq[:, 1:, :], deterministic=True)
            target_noise = (torch.randn_like(next_action_mean) * self.tt.critic.target_noise).clamp(-self.tt.critic.target_noise_clip, self.tt.critic.target_noise_clip)
            next_action = (next_action_mean + target_noise).clamp(-1.0, 1.0)

            next_q = torch.min(self.critic_target_1(torch.cat([next_obs, next_action], dim=-1)), self.critic_target_2(torch.cat([next_obs, next_action], dim=-1))).squeeze(-1)

            td_target = reward + self.tt.agent.gamma * (1 - done) * next_q

        critic_pred_1 = self.critic_local_1(torch.cat([obs, action], dim=-1)).squeeze(-1)
        critic_pred_2 = self.critic_local_2(torch.cat([obs, action], dim=-1)).squeeze(-1)
        critic_loss = F.mse_loss(critic_pred_1, td_target) + F.mse_loss(critic_pred_2, td_target)
        metrics[f'critic_loss'] = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(itertools.chain(self.critic_local_1.parameters(), self.critic_local_2.parameters()), self.tt.critic.max_grad_norm)
        metrics['critic_grad_norm'] = critic_grad_norm.item()
        self.critic_optimizer.step()


        # Train Actor
        actions_dist = self._get_action_distribution(obs_seq, action_seq[:, :-1, :])
        actions_sample = self._sample_action_from(actions_dist)

        next_actions_dist = self._get_action_distribution(next_obs_seq, action_seq[:, 1:, :])
        next_actions_sample = self._sample_action_from(next_actions_dist)

        actions_value_loss = -torch.min(self.critic_local_1(torch.cat([obs, actions_sample], dim=-1)), self.critic_local_2(torch.cat([obs, actions_sample], dim=-1))).mean()
        entropy_loss = -actions_dist.entropy().sum(-1).mean()
        smoothness_loss = (actions_sample - next_actions_sample).pow(2).sum(-1).mean()  # L2 norm difference between consecutive actions

        actor_loss = (
            actions_value_loss
            + (self.tt.actor.entropy_coef * entropy_loss)
            + (self.tt.actor.smoothness_coef * smoothness_loss)
        )


        metrics['actor_loss'] = actor_loss.item()
        metrics['actions_value_loss'] = actions_value_loss.item()
        metrics['actions_entropy'] = -entropy_loss.item()
        metrics['actions_jitteriness'] = smoothness_loss.item()
        metrics['actions_dist'] = _safe_histogram(actions_sample, num_bins=128)
        metrics['logstd'] = _safe_histogram(self.logstd, num_bins=128, min_range=1e-4)


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            itertools.chain([self.logstd], self.actor.parameters()), self.tt.actor.max_grad_norm
        )
        metrics['actor_grad_norm'] = actor_grad_norm.item()
        self.actor_optimizer.step()

        # Soft Update Target Network
        _soft_update(self.critic_local_1, self.critic_target_1, self.tt.critic.tau)
        _soft_update(self.critic_local_2, self.critic_target_2, self.tt.critic.tau)

        return metrics
