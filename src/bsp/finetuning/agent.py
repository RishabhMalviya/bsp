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

from bsp.common.utils import get_device
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

        tt = cfg.task_training

        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, tt.replay_buffer.capacity)

        # Actor: a BSPPolicyNet wrapping the (pretrained) DynamicsTransformer. The
        # transformer dims come from the shared `dp_transformer` config so the
        # architecture matches the pretraining checkpoint exactly.
        self.actor = BSPPolicyNet(
            obs_dim=obs_dim,
            ac_dim=ac_dim,
            H_max=tt.H_max,
            **cfg.dp_transformer,
        ).to(device)

        self.logstd = nn.Parameter(torch.ones(ac_dim, dtype=torch.float32, device=device))
        nn.init.normal_(self.logstd, mean=-0.5, std=0.1)  # Initialize logstd to have a mean of -0.5 (std of ~0.6 in action space)

        self.actor_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.actor.parameters()),
            lr=tt.actor.actor_lr
         )

        # Critic
        self.critic_local = TaskValueNet(
            obs_dim, ac_dim, hidden=tt.critic.hidden, depth=tt.critic.depth
        ).to(device)

        self.critic_target = TaskValueNet(
            obs_dim, ac_dim, hidden=tt.critic.hidden, depth=tt.critic.depth
        ).to(device)

        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target.eval()

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=tt.critic.critic_lr
        )

    def to_cpu(self):
        self.actor.to('cpu')
        self.critic_local.to('cpu')
        self.critic_target.to('cpu')
        self.logstd.data = self.logstd.data.to('cpu')  # In-place so the optimizer keeps referencing this Parameter

        self.device = 'cpu'

    def to_device(self):
        self.actor.to(device)
        self.critic_local.to(device)
        self.critic_target.to(device)
        self.logstd.data = self.logstd.data.to(device)

        self.device = device

    def _obs_seq_to_tensor(self, obs_seq : deque) -> torch.Tensor:
        """Stack a deque of observations into a (1, L, obs_dim) sequence tensor."""
        obs = np.stack(list(obs_seq), axis=0)  # (L, obs_dim)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_action_distribution(self, obs_seq: torch.Tensor, temperature: float = 1.0) -> distributions.Normal:
        """Run the policy over an observation sequence and return the action distribution.

        `obs_seq` is (B, L, obs_dim). Actions are unknown at action-selection time,
        so every action slot is masked (replaced by the transformer's learned action
        mask token) and the state slots are all left visible. The policy aggregates
        the sequence into a single (B, ac_dim) action mean.
        """
        B, L, _ = obs_seq.shape
        actions = torch.zeros(B, L, self.ac_dim, dtype=torch.float32, device=obs_seq.device)
        state_mask = torch.zeros(B, L, dtype=torch.bool, device=obs_seq.device)
        action_mask = torch.ones(B, L, dtype=torch.bool, device=obs_seq.device)

        action_mean = self.actor(obs_seq, actions, state_mask, action_mask)

        action_logstd = self.logstd.expand_as(action_mean)
        clipped_logstd = torch.clamp(action_logstd, min=math.log(1e-5), max=math.log(2.0))
        action_std = torch.exp(clipped_logstd) * float(temperature)

        return distributions.Normal(action_mean, action_std)

    def _sample_from(self, action_distribution: distributions.Distribution) -> torch.Tensor:
        return torch.clamp(input=action_distribution.rsample(), min=-1.0, max=1.0)

    def act(self, obs_history: deque, deterministic: bool = False, temperature: float = 1.0) -> torch.Tensor:
        """Select an action given the recent observation history.

        `obs_history` is a deque of numpy observations (most recent last). It is
        converted to a length-L sequence and run through the policy; the resulting
        (ac_dim,) action is returned (sampled, or the mean when `deterministic`).
        """
        obs_seq = self._obs_seq_to_tensor(obs_history)
        action_dist = self._get_action_distribution(obs_seq, temperature)

        action = action_dist.mean if deterministic else self._sample_from(action_dist)
        return action.squeeze(0)

    def update(self, batch) -> dict[str, float]:
        """Update the actor and critic from a batch of transitions.

        The batch is produced by the trainer's prep step:
          obs_seq      (B, L, obs_dim)  observation history ending at s_t
          action       (B, ac_dim)      action taken at s_t
          reward       (B,)             reward for (s_t, a_t)
          next_obs_seq (B, L, obs_dim)  observation history ending at s_{t+1}
          done         (B,)             episode-termination flag for the step

        The actor conditions on the full observation history (obs_seq / next_obs_seq,
        length L <= H_max); the critic operates on the single current/next
        transition, taken as the last element of each sequence. The critic is
        trained with a standard one-step TD target; the actor is trained to
        maximize the critic's value while encouraging exploration (entropy bonus)
        and smooth control signals (smoothness penalty), mirroring the pretraining
        curiosity agent.
        """
        def _soft_update(local_model, target_model, tau):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        tt = self.cfg.task_training

        obs_seq, action, reward, next_obs_seq, done = batch

        obs = obs_seq[:, -1]            # current state s_t
        next_obs = next_obs_seq[:, -1]  # next state s_{t+1}

        metrics = {}

        # Train Critic
        with torch.no_grad():
            next_actions = self._get_action_distribution(next_obs_seq).mean
            next_q = self.critic_target(torch.cat([next_obs, next_actions], dim=-1)).squeeze(-1)
            td_target = reward + tt.agent.gamma * (1 - done) * next_q

        critic_pred = self.critic_local(torch.cat([obs, action], dim=-1)).squeeze(-1)
        critic_loss = F.mse_loss(critic_pred, td_target)
        metrics[f'{self.downstream_task}_critic_loss'] = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train Actor
        actions_dist = self._get_action_distribution(obs_seq)
        actions_sample = self._sample_from(actions_dist)

        next_actions_dist = self._get_action_distribution(next_obs_seq)
        next_actions_sample = self._sample_from(next_actions_dist)

        actions_value_loss = -self.critic_target(torch.cat([obs, actions_sample], dim=-1)).mean()
        entropy_loss = -actions_dist.entropy().sum(-1).mean()
        smoothness_loss = (actions_sample - next_actions_sample).pow(2).sum(-1).mean()  # L2 norm difference between consecutive actions

        actor_loss = (
            actions_value_loss
            + (tt.actor.entropy_coef * entropy_loss)
            + (tt.actor.smoothness_coef * smoothness_loss)
        )

        metrics[f'{self.downstream_task}_actor_loss'] = actor_loss.item()
        # metrics[f'{self.downstream_task}_actions_value_loss'] = actions_value_loss.item()
        # metrics[f'{self.downstream_task}_actor_entropy'] = entropy_loss.item()
        # metrics[f'{self.downstream_task}_actor_smoothness_loss'] = smoothness_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update Target Network
        _soft_update(self.critic_local, self.critic_target, tt.critic.tau)

        return metrics
