"""Policy and value networks."""

import itertools

import torch
from torch import optim
import torch.nn as nn
from omegaconf import DictConfig

from bsp.common.utils import get_device
from bsp.common.base_classes import BaseAgent
from bsp.common.replay_buffer import ReplayBuffer
from bsp.finetuning.nn_modules import BSPPolicyNet, TaskValueNet


device = get_device()



class BSPAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, obs_dim: int, ac_dim: int):
        self.cfg = cfg
        self.device = device

        self.replay_buffer = ReplayBuffer(obs_dim, ac_dim, cfg.replay_buffer.capacity)

        # Actor
        self.actor = BSPPolicyNet(
            obs_dim=obs_dim,
            ac_dim=ac_dim,
            d_model=cfg.agent.d_model,
            embedder_hidden_dim=cfg.agent.embedder_hidden_dim,
            simnorm_dim=cfg.agent.simnorm_dim,
            num_heads=cfg.agent.num_heads,
            num_layers=cfg.agent.num_layers,
            dim_feedforward=cfg.agent.dim_feedforward,
            H_max=cfg.agent.H_max
        ).to(device)

        self.logstd = nn.Parameter(torch.ones(ac_dim, dtype=torch.float32, device=device))
        nn.init.normal_(self.logstd, mean=-0.5, std=0.1)  # Initialize logstd to have a mean of -0.5 (std of ~0.6 in action space)

        self.actor_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.actor.parameters()),
            lr=cfg.actor.actor_lr
         )

        # Critic
        self.critic_local = TaskValueNet(
            obs_dim, ac_dim, hidden=cfg.critic.hidden, depth=cfg.critic.depth
        ).to(device)

        self.critic_target = TaskValueNet(
            obs_dim, ac_dim, hidden=cfg.critic.hidden, depth=cfg.critic.depth
        ).to(device)

        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target.eval()

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), 
            lr=cfg.critic.critic_lr
        )

    def to_cpu(self):
        self.actor.to('cpu')
        self.critic_local.to('cpu')
        self.critic_target.to('cpu')
        self.logstd = nn.Parameter(self.logstd.data.to('cpu'))

        self.device = 'cpu'

    def to_device(self):
        self.actor.to(device)
        self.critic_local.to(device)
        self.critic_target.to(device)
        self.logstd = nn.Parameter(self.logstd.data.to(device))

        self.device = device


    def act(self, obs) -> torch.Tensor:
        raise NotImplementedError
    
    def update(self, batch) -> dict[str, float]:
        raise NotImplementedError