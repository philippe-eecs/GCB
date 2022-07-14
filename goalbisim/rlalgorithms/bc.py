import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from rlkit.core import logger
import wandb

from torch.distributions.multivariate_normal import MultivariateNormal

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.sac import Actor, Critic, QFunction, VFunction, gaussian_logprob, squash, weight_init



class GoalActorBC(Actor): #Same exact actor as SAC
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max, concat_goal_with_encoder = False):
        super().__init__(obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max)
        self.concat_goal_with_encoder = concat_goal_with_encoder
        
        if not concat_goal_with_encoder:
            scale = 2
        else:
            scale = 1
        self.encoder = encoder

        self.trunk = nn.Sequential(
            nn.Linear(scale * self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
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

        mu = self.trunk(policy_input)

        # constrain log_std inside [log_std_min, log_std_max]

        self.outputs['mu'] = mu
        #self.outputs['std'] = log_std.exp()

        std = torch.ones(mu.shape) * .1

        distribution = MultivariateNormal(mu, torch.stack([torch.diag(std[i]) for i in range(std.shape[0])], dim = 0))

        return mu, std, distribution