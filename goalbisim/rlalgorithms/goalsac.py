import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from rlkit.core import logger
import wandb

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.sac import Actor, Critic, QFunction, gaussian_logprob, squash, weight_init
#from encoder import make_encoder

LOG_FREQ = 10000

class GoalActor(Actor):
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max, concat_goal_with_encoder = False):
        self.concat_goal_with_encoder = concat_goal_with_encoder
        
        if not concat_goal_with_encoder:
            scale = 2
        else:
            scale = 1
        super().__init__(obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max)

        self.trunk = nn.Sequential(
            nn.Linear(scale * self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

    def forward(self, obs, goal, compute_pi=True, compute_log_pi=True, detach_encoder=False, detach_all=False):
        
        #import pdb; pdb.set_trace()
        if self.concat_goal_with_encoder:
            obs_goal = torch.cat([obs, goal], dim = 1) #Cat in color channel
            policy_input = self.encoder(obs_goal, detach=detach_encoder, detach_all=detach_all)
        else:
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
            policy_input = torch.cat([obs, goal], dim = 1)

        mu, log_std = self.trunk(policy_input).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

class GoalCritic(Critic):
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder, concat_goal_with_encoder = False, discrete = False):
        super().__init__(obs_shape, action_shape, hidden_dim, encoder)

        self.concat_goal_with_encoder = concat_goal_with_encoder

        if not concat_goal_with_encoder:
            scale = 2
        else:
            scale = 1

        self.Q1 = QFunction(
            scale * self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            scale * self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        

    def act(self, obs, goal, detach_encoder=True, detach_all=True):
        if self.concat_goal_with_encoder:
            obs_goal = torch.cat([obs, goal], dim = 1) #Cat in color channel
            q_input = self.encoder(obs_goal, detach=detach_encoder, detach_all=detach_all)
        else:
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
            q_input = torch.cat([obs, goal], dim = 1)

        #all_actions =

    def forward(self, obs, goal, action, detach_encoder=False, detach_all=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if self.concat_goal_with_encoder:
            obs_goal = torch.cat([obs, goal], dim = 1) #Cat in color channel
            q_input = self.encoder(obs_goal, detach=detach_encoder, detach_all=detach_all)
        else:
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
            q_input = torch.cat([obs, goal], dim = 1)

        q1 = self.Q1(q_input, action)
        q2 = self.Q2(q_input, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2