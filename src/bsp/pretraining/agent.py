import math
import itertools

import torch
import numpy as np
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F
from torch import distributions

from bsp.common.utils import get_device
from bsp.common.base_classes import BaseAgent
from bsp.common.replay_buffer import ReplayBuffer
from bsp.common.utils import _safe_histogram
from bsp.pretraining.nn_modules import CuriosityPolicyNet, CuriosityValueNet


device = get_device()


class CuriosityAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, obs_dim: int, ac_dim: int):
        self.cfg = cfg
        self.device = device

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, cfg.replay_buffer.capacity)

        # Actor
        self.actor = CuriosityPolicyNet(
            obs_dim, ac_dim, hidden=self.cfg.actor.hidden, depth=self.cfg.actor.depth
        ).to(device)

        self.logstd = nn.Parameter(torch.ones(ac_dim, dtype=torch.float32, device=device)*math.log(0.5))  # High initial exploration noise, annealed by temperature in act()

        self.actor_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.actor.parameters()),
            lr=self.cfg.actor.lr
        )

        # Critic
        self.critic_local = CuriosityValueNet(
            obs_dim, ac_dim, hidden=self.cfg.critic.hidden, depth=self.cfg.critic.depth
        ).to(device)

        self.critic_target = CuriosityValueNet(
            obs_dim, ac_dim, hidden=self.cfg.critic.hidden, depth=self.cfg.critic.depth
        ).to(device)

        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target.eval()
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.cfg.critic.lr)

    def to_cpu(self):
        self.actor.to('cpu')
        self.critic_local.to('cpu')
        self.critic_target.to('cpu')
        self.logstd.data = self.logstd.data.to('cpu')

        self.device = 'cpu'

    def to_device(self):
        self.actor.to(device)
        self.critic_local.to(device)
        self.critic_target.to(device)
        self.logstd.data = self.logstd.data.to(device)

        self.device = device

    def _get_action_distribution(self, obs: torch.Tensor, temperature: float = 1.0) -> distributions.Normal:
        action_mean = self.actor(obs)

        action_logstd = self.logstd.expand_as(action_mean)
        clipped_logstd = torch.clamp(action_logstd, min=math.log(1e-5), max=math.log(2.0))
        action_std = torch.exp(clipped_logstd) * float(temperature)

        return distributions.Normal(action_mean, action_std)


    def act(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_dist = self._get_action_distribution(obs, temperature)

        if deterministic:
            return action_dist.mean
        else:
            return action_dist.rsample()


    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """
            Update the agent's actor and critic networks using a batch of experiences.
            Expects batch = (obs, actions, rewards, next_obs, dones) where rewards
            are intrinsic rewards from the dynamics predictor and dones are
            (terminated | truncated) as floats.
        """
        def _soft_update(local_model, target_model, tau):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        obs, actions, rewards, next_obs, dones = batch
        metrics = {}

        # Train Critic
        with torch.no_grad():
            next_actions = self.act(next_obs, deterministic=True)
            next_q = self.critic_target(torch.cat([next_obs, next_actions], dim=-1)).squeeze(-1)
            td_target = rewards + self.cfg.gamma * (1 - dones) * next_q

        critic_pred = self.critic_local(torch.cat([obs, actions], dim=-1)).squeeze(-1)
        critic_loss = F.mse_loss(critic_pred, td_target)
        metrics['critic_loss'] = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train Actor
        pi_dist = self._get_action_distribution(obs)
        pi_actions = pi_dist.rsample()
        actions_value = self.critic_target(torch.cat([obs, pi_actions], dim=-1))
        entropy = pi_dist.entropy().sum(-1).mean()
        actor_loss = -actions_value.mean()  # - self.cfg.actor.entropy_coef*entropy
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_entropy'] = entropy.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.logstd.grad is not None:
            metrics['logstd_grad'] = _safe_histogram(self.logstd.grad)

        self.actor_optimizer.step()

        with torch.no_grad():
            metrics['logstd'] = _safe_histogram(self.logstd, num_bins=128, min_range=1e-4)

        # Soft Update Target Networks
        _soft_update(self.critic_local, self.critic_target, self.cfg.critic.tau)

        return metrics

