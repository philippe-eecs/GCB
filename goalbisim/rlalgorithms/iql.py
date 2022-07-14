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



class GoalActorIQL(Actor): #Same exact actor as SAC
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max, analogy_goal = False, phi_config = 'psi', phi_encoder = None, cpv_encoder = None):

        if phi_config == 'psi':
            scale = 2
            if analogy_goal:
                scale += 1
        elif phi_config == 'psi_phi':
            scale = 2

        elif phi_config == 'phi':
            scale = 1
            if analogy_goal:
                raise NotImplementedError
        
        elif phi_config == 'cpv':
            scale = 2

        else:
            raise NotImplementedError
        
        super().__init__(obs_shape, action_shape, hidden_dim, encoder, log_std_min, log_std_max)

        if phi_config == 'cpv':
            self.cpv_encoder = cpv_encoder

        self.trunk = nn.Sequential(
            nn.Linear(scale * self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.analogy_goal = analogy_goal
        self.phi_config = phi_config
        self.phi_encoder = phi_encoder

    def forward(self, obs, goal, compute_pi=True, compute_log_pi=True, detach_encoder=False, detach_all=False, init_obs=None):
        
        #import pdb; pdb.set_trace()
        if self.phi_config == 'psi':
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                context = self.encoder(goal[0], detach=detach_encoder, detach_all=detach_all)
                goal = self.encoder(goal[1], detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([obs, context, goal], dim = 1)
            else:
                goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([obs, goal], dim = 1)
        elif self.phi_config == 'psi_phi':
            psi = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([psi, phi], dim = 1)
            else:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([psi, phi], dim = 1)
        elif self.phi_config == 'phi':
            phi_input = torch.cat([obs, goal], dim = 1)
            policy_input = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
        elif self.phi_config == 'cpv':
            if self.analogy_goal:
                o_0T = torch.cat([goal[0], goal[1]], dim=1)
                o_0t = torch.cat([init_obs, obs], dim=1)
                v_oT = self.cpv_encoder(o_0T, detach=detach_encoder, detach_all=detach_all)
                v_ot = self.cpv_encoder(o_0t, detach=detach_encoder, detach_all=detach_all)
                v_tT = v_oT - v_ot

                obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([obs, v_tT], dim=1)

            else:
                assert init_obs is not None

                o_0T = torch.cat([init_obs, goal], dim=1)
                o_0t = torch.cat([init_obs, obs], dim=1)
                v_oT = self.cpv_encoder(o_0T, detach=detach_encoder, detach_all=detach_all)
                v_ot = self.cpv_encoder(o_0t, detach=detach_encoder, detach_all=detach_all)
                v_tT = v_oT - v_ot

                obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
                policy_input = torch.cat([obs, v_tT], dim=1)
        else:
            raise NotImplementedError
            

        mu, log_std = self.trunk(policy_input).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        std = log_std.exp()

        distribution = MultivariateNormal(mu, torch.stack([torch.diag(std[i]) for i in range(std.shape[0])], dim = 0))

        return mu, std, distribution

class GoalCriticIQL(Critic): #Basically same critic as SAC, but has a value function as well
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder, analogy_goal = False, phi_config = 'psi', phi_encoder = None):
        super().__init__(obs_shape, action_shape, hidden_dim, encoder)

        # Only use CPV architecture in Actor network, not Critic network.
        if phi_config == 'cpv':
            phi_config = 'psi'

        if phi_config == 'psi':
            scale = 2
            if analogy_goal:
                scale += 1
        elif phi_config == 'psi_phi':
            scale = 2

        elif phi_config == 'phi':
            scale = 1
            if analogy_goal:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.Q1 = QFunction(
            scale * self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            scale * self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.V = VFunction(scale * self.encoder.feature_dim, hidden_dim)

        self.analogy_goal = analogy_goal

        self.phi_encoder = phi_encoder

        self.phi_config = phi_config

    def forward_v(self, obs, goal, detach_encoder=False, detach_all=False):
        if self.phi_config == 'psi':
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                context = self.encoder(goal[0], detach=detach_encoder, detach_all=detach_all)
                goal = self.encoder(goal[1], detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([obs, context, goal], dim = 1)
            else:
                goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([obs, goal], dim = 1)
        elif self.phi_config == 'psi_phi':
            psi = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([psi, phi], dim = 1)
            else:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([psi, phi], dim = 1)
        elif self.phi_config == 'phi':
            phi_input = torch.cat([obs, goal], dim = 1)
            q_input = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
        else:
            raise NotImplementedError

            #q_input = torch.cat([obs, goal], dim = 1)

        v = self.V(q_input)

        return v

    def forward(self, obs, goal, action, detach_encoder=False, detach_all=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if self.phi_config == 'psi':
            obs = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                context = self.encoder(goal[0], detach=detach_encoder, detach_all=detach_all)
                goal = self.encoder(goal[1], detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([obs, context, goal], dim = 1)
            else:
                goal = self.encoder(goal, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([obs, goal], dim = 1)
        elif self.phi_config == 'psi_phi':
            psi = self.encoder(obs, detach=detach_encoder, detach_all=detach_all)
            if self.analogy_goal:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([psi, phi], dim = 1)
            else:
                phi_input = torch.cat([obs, goal], dim = 1)
                phi = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
                q_input = torch.cat([psi, phi], dim = 1)
        elif self.phi_config == 'phi':
            phi_input = torch.cat([obs, goal], dim = 1)
            q_input = self.phi_encoder(phi_input, detach=detach_encoder, detach_all=detach_all)
        else:
            raise NotImplementedError


        q1 = self.Q1(q_input, action)
        q2 = self.Q2(q_input, action)
        

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        #self.outputs['v'] = v

        return q1, q2
