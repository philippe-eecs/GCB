import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.goalsac import  GoalActor, GoalCritic, weight_init, LOG_FREQ
from goalbisim.utils.misc_utils import soft_update_params
from rlkit.core import logger
from goalbisim.agents.pixelsac import PixelSACAgent
import wandb
#from transition_model import make_transition_model
#from decoder import make_decoder


class GoalPixelSACAgent(nn.Module):
    """Basic SAC Agent with Encoder Attached (Could be representation algorithm attached)."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        eval_transforms,
        actor_representation,
        critic_representation,
        target_critic_representation,
        reward_function,
        policy_hidden_dim = 256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        encoder_tau=0.01,
        give_policy = False,
        detach_encoder = False,
        detach_conv = False,
        discrete = False,
        critic_target_update_freq=2,
        representation_update_freq=1,
        concat_goal_with_encoder = False
    ):
        super().__init__()
        self.device = device
        self.discount = discount
        self.eval_transforms = eval_transforms
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.representation_update_freq = representation_update_freq
        self.concat_goal_with_encoder = concat_goal_with_encoder
        self.hinge = 1.
        self.sigma = 0.5
        self.give_policy = give_policy

        self.detach_encoder = detach_encoder
        self.detach_conv = detach_conv

        #generalized class of encoder
        self.actor_representation = actor_representation
        self.critic_representation = critic_representation
        self.target_critic_representation = target_critic_representation
        self.reward_function = reward_function #Intrinisc or direct..., possibly truth rewards...


        #Ask if different encoders are used here...
        self.actor = GoalActor(
            obs_shape, action_shape, policy_hidden_dim, actor_representation.encoder, actor_log_std_min, actor_log_std_max
        ,concat_goal_with_encoder = concat_goal_with_encoder).to(device)

        self.critic = GoalCritic(
            obs_shape, action_shape, policy_hidden_dim, critic_representation.encoder
        ,concat_goal_with_encoder = concat_goal_with_encoder).to(device)

        self.critic_target = GoalCritic(
            obs_shape, action_shape, policy_hidden_dim, target_critic_representation.encoder
        ,concat_goal_with_encoder = concat_goal_with_encoder).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)


        # optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999), weight_decay = 5e-4
        )
        #Should include encoder... make sure to detach when neccessary

        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999), weight_decay = 5e-4
        )

        self.log_alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999), weight_decay = 5e-4
        )


        #self.discrete = discrete
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                goal = self.eval_transforms(goal, self.device)
                #obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                goal = goal.unsqueeze(0)
                mu, _, _, _ = self.actor(
                    obs, goal, compute_pi=False, compute_log_pi=False
                )
                return mu.cpu().data.numpy().flatten()
            else:
                mu, _, _, _ = self.actor(
                    obs, goal, compute_pi=False, compute_log_pi=False
                )
                return mu

    def sample_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                goal = self.eval_transforms(goal, self.device)
                #obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                goal = goal.unsqueeze(0)
                mu, pi, _, _ = self.actor(obs, goal, compute_log_pi=False)
                return pi.cpu().data.numpy().flatten()
            else:
                mu, pi, _, _ = self.actor(obs, goal, compute_log_pi=False)
                return pi

    def get_action_distribution(self, obs, goal): 
        with torch.no_grad():
            #obs = self.eval_transforms(obs, self.device)
            #goal = self.eval_transforms(goal, self.device)
            #obs = torch.FloatTensor(obs).to(self.device)
            #obs = obs.unsqueeze(0)
            #goal = goal.unsqueeze(0)
            mu, _, log_pi, log_std = self.actor(
                obs, goal, compute_pi=False, compute_log_pi=False
            )
            return mu, log_std        
    
    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        #L.log('train/batch_reward', reward.mean(), step)

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}

        logger.logging_tool.log(stats)

        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = None
        kwargs['goal'] = goals

        self.update_critic(obs, goals, action, reward, next_obs, not_done, step)

        self.critic_representation.update(replay_buffer, self, kwargs, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, goals, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def update_critic(self, obs, goals, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, goals)
            target_Q1, target_Q2 = self.critic_target(next_obs, goals, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, goals, action, detach_encoder=self.detach_conv, detach_all = self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        #L.log('train_critic/loss', critic_loss, step)

        stats = {'train_step' : step,
        'train/critic/loss' : critic_loss}

        logger.logging_tool.log(stats)



        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(step)



    def update_actor_and_alpha(self, obs, goals, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, goals, detach_encoder=True, detach_all=True)
        actor_Q1, actor_Q2 = self.critic(obs, goals, pi, detach_encoder=True, detach_all=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        #L.log('train_actor/loss', actor_loss, step)
        #L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        #L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        #L.log('train_alpha/loss', alpha_loss, step)
        #L.log('train_alpha/value', self.alpha, step)

        stats = {'train_step' : step,
        'train/alpha/loss' : alpha_loss,
        'train/alpha/value' : self.alpha,
        'train/actor/loss' : actor_loss,
        'train/actor/target_entropy' : self.target_entropy,
        'train/actor/entropy' : entropy.mean()}

        logger.logging_tool.log(stats)

        alpha_loss.backward()

        self.log_alpha_optimizer.step()  

    def test_representation(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = None
        kwargs['goal'] = goals

        self.critic_representation.eval_loss(replay_buffer, self, kwargs, step)

    def step_all(self):
        self.step_lr()
        self.critic_representation.step_lr()

    def step_lr(self):
        self.actor_optimizer.step()
        self.critic_optimizer.step()       

    def save(self, save_loc, name):
        save_dir = save_loc + '/agents/'
        try:
            torch.save(self.actor.state_dict(), save_dir + 'actor' + name + '.pt')
            torch.save(self.critic.state_dict(), save_dir + 'critic' + name + '.pt')
            torch.save(self.critic_target.state_dict(), save_dir + 'target_critic' + name + '.pt')
        except:
            import os
            os.makedirs(save_dir)
            torch.save(self.actor.state_dict(), save_dir + 'actor' + name + '.pt')
            torch.save(self.critic.state_dict(), save_dir + 'critic' + name + '.pt')
            torch.save(self.critic_target.state_dict(), save_dir + 'target_critic' + name + '.pt')
        

    def load(self, loc_dir, name):
        loc_dir = loc_dir + '/agents/'
        self.actor.load_state_dict(torch.load(loc_dir + 'actor' + name + '.pt'))
        self.critic.load_state_dict(torch.load(loc_dir + 'critic' + name + '.pt'))
        self.critic_target.load_state_dict(torch.load(loc_dir + 'target_critic' + name + '.pt'))  

    
        