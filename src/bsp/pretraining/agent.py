import math

import torch
import numpy as np
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F
from torch import distributions

from bsp.pretraining.dynamics_predictor import DynamicsPredictor
from bsp.utils import get_device
from bsp.pretraining.nn_modules import MLP
from bsp.common.base_classes import BaseAgent


device = get_device()

# TODO: Try copying HW2 code here.

class CuriosityAgent(BaseAgent):
    def __init__(self, agent_cfg: DictConfig, obs_dim: int, ac_dim: int):
        super().__init__(agent_cfg, obs_dim, ac_dim)  # Intialize ReplayBuffer and cfg

        # Actor
        self.actor = MLP(in_dim=obs_dim, out_dim=ac_dim, hidden=agent_cfg.actor.hidden, depth=agent_cfg.actor.depth)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=agent_cfg.actor.lr)
        self.logstd = nn.Parameter(
            torch.zeros(ac_dim, dtype=torch.float32, device=device)
        )

        # Critic
        self.critic_local = MLP(in_dim=obs_dim+ac_dim, out_dim=1, hidden=agent_cfg.critic.hidden, depth=agent_cfg.critic.depth)
        self.critic_target = MLP(in_dim=obs_dim+ac_dim, out_dim=1, hidden=agent_cfg.critic.hidden, depth=agent_cfg.critic.depth)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target.eval()
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=agent_cfg.critic.lr)


    def _get_action_distribution(self, obs: torch.Tensor, temperature: float = 1.0) -> distributions.Normal:
        action_mean = torch.tanh(self.actor(obs))

        action_logstd = self.logstd.expand_as(action_mean)
        clipped_logstd = torch.clamp(action_logstd, min=math.log(1e-5), max=math.log(2.0))
        action_std = torch.exp(clipped_logstd) * float(temperature)

        return distributions.Normal(action_mean, action_std)

    def act(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        action_dist = self._get_action_distribution(obs, temperature)

        if deterministic:
            return action_dist.mean
        else:
            return action_dist.sample()
            
    def update(self, dynamics_predictor: DynamicsPredictor) -> dict[str, float]:
        def _soft_update(local_model, target_model, tau):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        training_metrics = {}

        # ------ Sample Experiences ------ #
        obs, actions, rewards, next_obs, truncated, terminated = self.replay_buffer.sample(batch_size=self.cfg.batch_size)
        # TODO: Change rewards to the values of the loss from dynamics prediction
        rewards = DynamicsPredictor.compute_intrinsic_reward(dynamics_predictor, obs, actions, next_obs).unsqueeze(-1)
        dones = torch.logical_or(truncated, terminated)

        # ------ Train Local Critic ------ #
        with torch.no_grad():
            td_target = rewards + (self.cfg.gamma * (1 - dones) * self.critic_target(next_obs, self.act(next_obs, deterministic=True)))

        critic_loss = F.mse_loss(self.critic_local(obs, actions), td_target)
        training_metrics['critic_loss'] = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------ Train Actor ------ #
        actor_loss = -self.critic_target(obs, self.act(obs, deterministic=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------ Soft Update Target Networks ------ #
        _soft_update(self.critic_local, self.critic_target, self.cfg.critic.tau)


        return {}

