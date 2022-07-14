import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import goalbisim.utils.misc_utils
from goalbisim.rlalgorithms.iql import  GoalActorIQL, GoalCriticIQL
from goalbisim.utils.misc_utils import soft_update_params
from rlkit.core import logger
from goalbisim.agents.pixelsac import PixelSACAgent
import wandb
#from transition_model import make_transition_model
#from decoder import make_decoder


class GoalPixelIQLAgent(nn.Module):
    """Basic IQL Agent with Encoder Attached (Could be representation algorithm attached)."""
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
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        encoder_tau=0.01,
        quantile=0.5,
        policy_update_period=1,
        q_update_period=1,
        target_update_period=1,
        clip_score=None,
        soft_target_tau=1e-2,
        beta=1.0,
        detach_encoder = False,
        detach_conv = False,
        analogy_goal = False,
        use_adamw = False,
        phi_config = 'psi'
    ):
        super().__init__()
        self.device = device
        self.discount = discount
        self.eval_transforms = eval_transforms
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        #self.concat_goal_with_encoder = concat_goal_with_encoder
        self.hinge = 1.
        self.sigma = 0.5
        self.beta = beta
        self.clip_score = clip_score
        self.analogy_goal = analogy_goal
        self.phi_config = phi_config

        self.q_update_period = q_update_period
        self.target_update_period = target_update_period
        self.policy_update_period = policy_update_period

        self.detach_encoder = detach_encoder
        self.detach_conv = detach_conv

        #generalized class of encoder
        self.actor_representation = actor_representation
        self.critic_representation = critic_representation
        self.target_critic_representation = target_critic_representation
        self.reward_function = reward_function #Intrinisc or direct..., possibly truth rewards...


        #Ask if different encoders are used here...
        cpv_encoder, place = None, None
        if 'phi' in phi_config:
            place = actor_representation.phi.encoder 
        elif 'cpv' in phi_config:
            from goalbisim.representation.encoders.RADencoder import PixelEncoder
            cpv_obs_shape = actor_representation.encoder.obs_shape
            cpv_obs_shape = (cpv_obs_shape[0] * 2,) + cpv_obs_shape[1:]
            cpv_encoder = PixelEncoder(
                cpv_obs_shape, 
                actor_representation.encoder.feature_dim, 
                num_layers=actor_representation.encoder.num_layers, 
                num_filters=actor_representation.encoder.num_filters,
                output_logits=actor_representation.encoder.output_logits,
                tanh_scale=actor_representation.encoder.tanh_scale,
                goal_flag=actor_representation.encoder.goal_flag,
            )

        self.actor = GoalActorIQL(
            obs_shape, action_shape, policy_hidden_dim, actor_representation.encoder, actor_log_std_min, actor_log_std_max
        ,analogy_goal = analogy_goal, phi_config = phi_config, phi_encoder = place, cpv_encoder = cpv_encoder).to(device)

        if 'phi' in phi_config:
            place = critic_representation.phi.encoder 
        else:
            place = None

        self.critic = GoalCriticIQL(
            obs_shape, action_shape, policy_hidden_dim, critic_representation.encoder
        ,analogy_goal = analogy_goal, phi_config = phi_config, phi_encoder = place).to(device)

        if 'phi' in phi_config:
            place = target_critic_representation.phi.encoder 
        else:
            place = None

        self.critic_target = GoalCriticIQL(
            obs_shape, action_shape, policy_hidden_dim, target_critic_representation.encoder
        ,analogy_goal = analogy_goal, phi_config = phi_config, phi_encoder = place).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        if 'phi' in phi_config:
            self.actor.phi_encoder.copy_conv_weights_from(self.critic.phi_encoder)


        # optimizers
        if use_adamw:
            self.actor_optimizer = torch.optim.AdamW(
                self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999), weight_decay = 5e-4
            )
            #Should include encoder... make sure to detach when neccessary
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999), weight_decay = 5e-4
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999), weight_decay = 5e-4
            )
            #Should include encoder... make sure to detach when neccessary
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999), weight_decay = 5e-4
            )

        self.actor_optimizer_step = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, 2, gamma=0.9, last_epoch= -1)
        self.critic_optimizer_step = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, 2, gamma=0.9, last_epoch= -1)
        #TODO: Might need reward transforms, but I don't think it's neccessary...

        self.quantile = quantile

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                obs = obs.unsqueeze(0)
                if self.analogy_goal:
                    assert isinstance(goal, list)
                    goal1 = self.eval_transforms(goal[0], self.device)
                    goal2 = self.eval_transforms(goal[1], self.device)

                    goal1 = goal1.unsqueeze(0)
                    goal2 = goal2.unsqueeze(0)
                    goal = [goal1, goal2]
                    #goal[0] = goal[0].unsqueeze(0)
                    #goal[1] = goal[1].unsqueeze(0)
                else:
                    goal = self.eval_transforms(goal, self.device)
                    goal = goal.unsqueeze(0)
                
                #goal = goal.unsqueeze(0)
                if init_obs is not None:
                    init_obs = self.eval_transforms(init_obs, self.device)
                    init_obs = init_obs.unsqueeze(0)    
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False, init_obs=init_obs)
                return mu.cpu().numpy().flatten()
            else:
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False, init_obs=init_obs)
                return mu


    def sample_action(self, obs, goal, batched = False, init_obs=None):
        with torch.no_grad():
            if not batched:
                obs = self.eval_transforms(obs, self.device)
                obs = obs.unsqueeze(0)
                if self.analogy_goal:
                    assert isinstance(goal, list)
                    goal1 = self.eval_transforms(goal[0], self.device)
                    goal2 = self.eval_transforms(goal[1], self.device)

                    goal1 = goal1.unsqueeze(0)
                    goal2 = goal2.unsqueeze(0)
                    goal = [goal1, goal2]
                    #goal[0] = goal[0].unsqueeze(0)
                    #goal[1] = goal[1].unsqueeze(0)
                else:
                    goal = self.eval_transforms(goal, self.device)
                    goal = goal.unsqueeze(0)
                
                if init_obs is not None:
                    init_obs = self.eval_transforms(init_obs, self.device)
                    init_obs = init_obs.unsqueeze(0)    
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False, init_obs=init_obs)
                return dist.sample().cpu().numpy().flatten()
            else:
                mu, std, dist = self.actor(obs, goal, compute_log_pi=False, init_obs=init_obs)
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

    def IQL_update(self, obs, goals, action, reward, next_obs, not_done, step, critic_gradients_allowed = True, init_obs=None):

        #IQL Q Update

        mu, std, dist = self.actor(obs, goals, detach_encoder=True, detach_all = True, init_obs=init_obs) #Detach for actor, just use critic for advice
        if critic_gradients_allowed:
            Q1_pred, Q2_pred = self.critic(obs, goals, action, detach_encoder = self.detach_conv, detach_all = self.detach_encoder)
        else:
            Q1_pred, Q2_pred = self.critic(obs, goals, action, detach_encoder=True, detach_all = True)

        V_target = self.critic.forward_v(next_obs, goals).detach()

        target_Q = (reward + not_done * self.discount * V_target).detach()

        #qf1_loss = nn.MSELoss(Q1_pred, target_Q)
        #qf2_loss = nn.MSELoss(Q2_pred, target_Q)

        Q_critic_loss = F.mse_loss(Q1_pred, target_Q) + F.mse_loss(Q2_pred, target_Q)


        #IQL V Update

        Q1_pred_target, Q2_pred_target = self.critic_target(obs, goals, action, detach_encoder=True, detach_all = True)

        min_q_update = torch.min(Q1_pred_target, Q2_pred_target).detach()
        V_pred = self.critic.forward_v(obs, goals)
        Vf_error = V_pred - min_q_update
        Vf_sign = (Vf_error > 0).float()
        vf_weight = (1 - Vf_sign) * self.quantile + Vf_sign * (1 - self.quantile)
        Vf_loss = (vf_weight * (Vf_error ** 2)).mean()

        critic_loss = Q_critic_loss + Vf_loss

        policy_logpp = dist.log_prob(action)

        advantage = min_q_update - V_pred

        exp_adv = torch.exp(advantage / self.beta)

        weights = exp_adv[:, 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        #L.log('train_critic/loss', critic_loss, step)

        stats = {'train_step' : step,
        'train/critic/Q_loss' : Q_critic_loss.item(),
        'train/critic/V_loss' : Vf_loss.item(),
        'train/actor/pi_loss' : policy_loss.item()}

        logger.logging_tool.log(stats)

        if step % self.q_update_period == 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        if step % self.policy_update_period == 0:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()         

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
        kwargs['td'] = None
        kwargs['goal'] = goals

        if self.analogy_goal:
            kwargs['analogy_obs'] = kwargs['analogy_obses']
            kwargs['analogy_goals'] = kwargs['analogy_goals']
            self.IQL_update(obs, [kwargs['analogy_obses'], kwargs['analogy_goals']], action, reward, next_obs, not_done, step, init_obs=kwargs.get('init_obs', None))
        else:
            self.IQL_update(obs, goals, action, reward, next_obs, not_done, step, init_obs=kwargs.get('init_obs', None))
        self.critic_representation.update(replay_buffer, self, kwargs, step) #Important to pass in policy!

        if step % self.target_update_period == 0:
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

            if 'phi' in self.phi_config:
                soft_update_params(
                self.critic.phi_encoder, self.critic_target.phi_encoder,
                self.encoder_tau
                )
        #if step % self.representation_update_freq == 0:
            

    def update_representation(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = None
        kwargs['td'] = None
        kwargs['goal'] = goals

        self.critic_representation.update(replay_buffer, self, kwargs, step)

        if step % self.target_update_period == 0:
            #Might not be neccessary, but encoders are being updated nonetheless...
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def update_policy(self, replay_buffer, step, critic_gradients_allowed = False):
        obs, action, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample() #Work On

        stats = {'train_step' : step,
        'train/reward_sampled_mean' : reward.mean()}


        logger.logging_tool.log(stats)


        kwargs['obs'] = obs
        kwargs['next_obs'] = next_obs
        kwargs['action'] = action
        kwargs['reward'] = reward
        kwargs['rtg'] = None
        kwargs['td'] = None
        kwargs['goal'] = goals

        self.IQL_update(obs, goals, action, reward, next_obs, not_done, step, critic_gradients_allowed = critic_gradients_allowed, init_obs=kwargs.get('init_obs', None))

        if step % self.target_update_period == 0:
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
        kwargs['td'] = None
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
        