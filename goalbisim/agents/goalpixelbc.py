import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.bc import GoalActorBC
from goalbisim.utils.misc_utils import soft_update_params
from rlkit.core import logger
from goalbisim.agents.pixelsac import PixelSACAgent
import wandb
#from transition_model import make_transition_model
#from decoder import make_decoder


class GoalPixelBCAgent(nn.Module):
    """Basic IQL Agent with Encoder Attached (Could be representation algorithm attached)."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        eval_transforms,
        actor_representation,
        reward_function,
        policy_hidden_dim = 256,
        deterministic = False,
        discount=0.99,
        actor_lr=1e-3,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        detach_encoder = False,
        detach_conv = False,
        concat_goal_with_encoder = False
    ):
        super().__init__()
        self.device = device
        self.discount = discount
        self.eval_transforms = eval_transforms
        self.concat_goal_with_encoder = concat_goal_with_encoder
        self.detach_encoder = detach_encoder
        self.detach_conv = detach_conv
        self.deterministic = deterministic

        #generalized class of encoder
        self.actor_representation = actor_representation
        self.reward_function = reward_function #Intrinisc or direct..., possibly truth rewards...


        #Ask if different encoders are used here...
        self.actor = GoalActorBC(
            obs_shape, action_shape, policy_hidden_dim, actor_representation.encoder, actor_log_std_min, actor_log_std_max
        ,concat_goal_with_encoder = concat_goal_with_encoder).to(device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999), weight_decay = 5e-4
        )


        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        #self.critic.train(training)

    def select_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                goal = self.eval_transforms(goal, self.device)
                #obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                goal = goal.unsqueeze(0)
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False)
                return mu.cpu().numpy().flatten()
            else:
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False)
                return mu


    def sample_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                goal = self.eval_transforms(goal, self.device)
                #obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                goal = goal.unsqueeze(0)
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False)
                return dist.sample().cpu().numpy().flatten()
            else:
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False)
                return dist.sample()

    def get_action_distribution(self, obs, goal): 
        with torch.no_grad():
            #obs = self.eval_transforms(obs, self.device)
            #goal = self.eval_transforms(goal, self.device)
            #obs = torch.FloatTensor(obs).to(self.device)
            #obs = obs.unsqueeze(0)
            #goal = goal.unsqueeze(0)
            mu, std, dist = self.actor(
                obs, goal, compute_pi=False, compute_log_pi=False
            )
            return mu, std, dist        

    def BC_update(self, obs, goals, action, reward, next_obs, not_done, step, critic_gradients_allowed = True):

        #IQL Q Update

        mu, std, dist = self.actor(obs, goals, detach_encoder=self.detach_conv, detach_all = self.detach_encoder) #Detach for actor, just use critic for advice

        if self.deterministic:
            policy_logpp = (action - dist.mu)**2
        else:
            policy_logpp = dist.log_prob(action)
            policy_loss = (-policy_logpp).mean()


        #L.log('train_critic/loss', critic_loss, step)

        stats = {'train_step' : step,
        'train/critic/BC_loss' : policy_loss.item()}

        logger.logging_tool.log(stats)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()         

    def update(self, replay_buffer, step):
        obs, action, rtg, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        #L.log('train/batch_reward', reward.mean(), step)

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}

        logger.logging_tool.log(stats)

        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['rtg'] = rtg
        kwargs['reward'] = reward
        kwargs['goal'] = goals

        self.BC_update(obs, goals, action, reward, next_obs, not_done, step)
        self.actor_representation.update(replay_buffer, self, kwargs, step) #Important to pass in policy!

            

    def update_representation(self, replay_buffer, step):
        obs, action, rtg, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = rtg
        kwargs['goal'] = goals

        self.actor_representation.update(replay_buffer, self, kwargs, step)


    def update_policy(self, replay_buffer, step, critic_gradients_allowed = False):
        obs, action, rtg, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = rtg
        kwargs['goal'] = goals

        self.BC_update(obs, goals, action, reward, next_obs, not_done, step, critic_gradients_allowed = critic_gradients_allowed)


    def test_representation(self, replay_buffer, step):
        obs, action, rtg, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = rtg
        kwargs['goal'] = goals

        self.actor_representation.eval_loss(replay_buffer, self, kwargs, step)

    def save(self, save_loc, name):
        save_dir = save_loc + '/agents/'
        try:
            torch.save(self.actor.state_dict(), save_dir + 'actor' + name + '.pt')
            #torch.save(self.critic.state_dict(), save_dir + 'critic' + name + '.pt')
        except:
            import os
            os.makedirs(save_dir)
            torch.save(self.actor.state_dict(), save_dir + 'actor' + name + '.pt')
            #torch.save(self.critic.state_dict(), save_dir + 'critic' + name + '.pt')
        

    def load(self, loc_dir, name):
        self.actor.load_state_dict(torch.load(loc_dir + 'actor' + name + '.pt'))
        #self.critic.load_state_dict(torch.load(loc_dir + 'critic' + name + '.pt'))
        