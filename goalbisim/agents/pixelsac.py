import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.sac import  Actor, Critic, weight_init, LOG_FREQ
from goalbisim.utils.misc_utils import soft_update_params
from goalbisim.logging.logging import logger
import wandb
#from transition_model import make_transition_model
#from decoder import make_decoder


class PixelSACAgent(nn.Module):
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
        detach_encoder = False,
        critic_target_update_freq=2,
        representation_update_freq=1,
    ):
        super().__init__()
        self.device = device
        self.eval_transforms = eval_transforms
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.representation_update_freq = representation_update_freq
        self.hinge = 1.
        self.sigma = 0.5

        self.detach_encoder = detach_encoder

        #generalized class of encoder
        self.actor_representation = actor_representation
        self.critic_representation = critic_representation
        self.target_critic_representation = target_critic_representation
        self.reward_function = reward_function #Intrinisc or direct..., possibly truth rewards...


        #Ask if different encoders are used here...
        self.actor = Actor(
            obs_shape, action_shape, policy_hidden_dim, actor_representation.encoder, actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, policy_hidden_dim, critic_representation.encoder
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, policy_hidden_dim, target_critic_representation.encoder
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)


        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        #Should include encoder... make sure to detach when neccessary

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = self.eval_transforms(obs, self.device)
            #obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = self.eval_transforms(obs, self.device)
            #obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        #L.log('train_critic/loss', critic_loss, step)

        stats = {'train_step' : step,
        'train_critic/loss' : critic_loss}

        logger.record_dict(stats)
        wandb.log(stats)



        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

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

        #self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        #L.log('train_alpha/loss', alpha_loss, step)
        #L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, _, true_reward, next_obs, not_done, kwargs = replay_buffer.sample() #Work On

        #L.log('train/batch_reward', reward.mean(), step)
        reward = true_reward

        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = true_reward
        kwargs['curr_rewards'] = None


        self.update_critic(obs, action, reward, next_obs, not_done, step)
        self.critic_representation.update(replay_buffer, kwargs, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

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

            



    def update_representation(self, obs, action, next_obs):

        # Sample negative state across episodes at random

        loss = self.critic.representation.update(obs, action, next_obs) #Might need to change...
        return loss

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )