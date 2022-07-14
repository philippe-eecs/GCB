import torch
import torch.nn as nn
from goalbisim.representation.encoders.RADencoder import PixelEncoder
from rlkit.core import logger
import torch.nn.functional as F
from goalbisim.dynamics.dynamics_models import make_transition_model
import numpy as np
import wandb
import math


class PairedStateGoal(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type = 'ensemble',
            metric_loss = 'l1',
            metric_distance = 'reward',
            decoder_type = 'reward',
            dynamics_loss = 'direct',
            dual_optimization = False,
            action_weight = 1,
            on_policy_dynamics = False, #Might be needed when performing offline RL
            decode_both = False,
            disconnect_implict_policy = True,
            train_iters_per_update = 1,
            action_shape = (5, 1), #We will need to fix at somepoint....
            action_scale = 1, #to clip actions properly...
            discount = 0.99,
            steps_till_on_policy = 3000,
            encoder_weight = 1,
            transition_weight = 1,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            output_logits = True,
            lr=1e-3,
            weight_decay = 0):
        super().__init__()

        self.device = device
        self.encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits = output_logits, goal_flag = True).to(self.device)
        self.phi = self
        
        self.action_scale = action_scale
        self.decode_both = decode_both
        self.metric_loss = metric_loss
        self.psi = self
        self.action_weight = action_weight
        self.metric_distance = metric_distance
        self.decoder_type = decoder_type
        self.on_policy_dynamics = on_policy_dynamics
        self.feature_dim = feature_dim
        self.transition_model_type = transition_model_type
        self.train_iters_per_update = train_iters_per_update
        self.encoder_weight = encoder_weight
        self.transition_weight = transition_weight
        self.disconnect_implict_policy = disconnect_implict_policy
        self.steps_till_on_policy = steps_till_on_policy
        self.dynamics_loss = dynamics_loss
        self.dual_optimization = dual_optimization
        self.cross_entropy = nn.CrossEntropyLoss()

        if self.decode_both:
            scale = 2
        else:
            scale = 1

        if self.decoder_type == 'reward' or self.decoder_type == 'rtg' or self.decoder_type == 'temporal' or self.decoder_type == 'none':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(self.device)
        elif self.decoder_type == 'rtg_reward':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 2)).to(self.device)
        elif self.decoder_type == 'temporal_action':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1 + action_shape[0])).to(self.device)           
        elif self.decoder_type == 'action':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(self.device)
        else:
            raise NotImplementedError



        if self.transition_model_type != 'next_observation' and self.transition_model_type != 'next_observation_l1':
            if self.transition_model_type == 'ensemble_proper':
                input_text = 'ensemble'
            elif self.transition_model_type == 'next_observation_ensemble':
                input_text = 'ensemble'
            else:
                input_text = self.transition_model_type

            self.dynamics_model = make_transition_model(input_text, feature_dim, action_shape).to(self.device)
        else:
            self.dynamics_model = make_transition_model('', feature_dim, action_shape).to(self.device)

        
        try:
            self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            self.decoder_optimizer = torch.optim.AdamW(list(self.decoder.parameters()) + list(self.dynamics_model.parameters()), lr=lr, weight_decay=weight_decay)
        except:
            raise NotImplementedError
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            self.decoder_optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.dynamics_model.parameters()), lr=lr, weight_decay=weight_decay)

        self.encoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, 1, gamma=0.9, last_epoch= -1, verbose=False)
        self.decoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, 1, gamma=0.9, last_epoch= -1, verbose=False)

        self.discount = discount
        self.mse = nn.MSELoss()



    def forward(self, obs, goal = None, detach=False):
        if goal is not None:
            obs = torch.cat([obs, goal], dim = 1) 
        return self.encode(obs, detach = detach)

    def encode(self, obs, goal = None, detach=False):
        if goal is not None:
            obs = torch.cat([obs, goal], dim = 1) 

        z_out = self.encoder(obs, detach = detach)

        return z_out

    def encoder_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, log = True, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal)
        perm = np.random.permutation(obs.shape[0])
        z_pair = z[perm]
        reward_pair = reward[perm]

        norms = torch.norm(z.detach(), p=1, dim = 1)
        output_norm = torch.nn.functional.normalize(z.detach(), p = 1, dim = 1)
        output_std = torch.std(output_norm, 0).mean().item()
        collapse_level = max(0., 1 - math.sqrt(self.feature_dim) * output_std)
        std_norm = torch.std(norms).detach().item()


        if self.transition_model_type != 'next_observation' and self.transition_model_type != 'next_observation_l1':
            with torch.no_grad():
                pred_next_latent_mu1, pred_next_latent_sigma1 = self.dynamics_model(torch.cat([z, action], dim=1))
                if self.dynamics_loss == 'delta':
                    pred_next_latent_mu1 += z
            if pred_next_latent_sigma1 is None:
                pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
            if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
            elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
            else:
                raise NotImplementedError

        if self.metric_loss == 'l1':
            z_dist = torch.norm(z - z_pair, p = 1, dim = 1)
        elif self.metric_loss == 'l2':
            z_dist = torch.norm(z - z_pair, dim = 1)
        else:
            raise NotImplementedError


        if self.metric_distance == 'reward':
            metric = F.smooth_l1_loss(reward, reward_pair, reduction='none').squeeze() * self.action_weight
        elif self.metric_distance == 'action':
            metric = (torch.norm(action - action[perm], dim = 1) * self.action_weight).squeeze()
        elif self.metric_distance == 'temporal':
            td_pair = td[perm]
            metric = torch.norm(td - td_pair, dim = 1).squeeze() * self.action_weight
        elif self.metric_distance == 'advantage_target':
            q1, q2 = policy.critic_target(obs, goal, action)
            vs = policy.critic_target.forward_v(obs, goal).detach()
            adv = q1.detach() - vs
            adv_pair = adv[perm]
            metric = torch.norm(adv - adv_pair, dim = 1).squeeze() * self.action_weight
        elif self.metric_distance == 'advantage':
            q1, q2 = policy.critic(obs, goal, action)
            vs = policy.critic.forward_v(obs, goal).detach()
            adv = q1.detach() - vs
            adv_pair = adv[perm]
            metric = torch.norm(adv - adv_pair, dim = 1).squeeze() * self.action_weight
        else:
            raise NotImplementedError

        if self.transition_model_type == 'next_observation':
            z_next = self.encode(next_obs, goal)
            z_next_pair = z_next[perm]
            transition_dist = torch.norm(z_next - z_next_pair, dim = 1)
        elif self.transition_model_type == 'next_observation_l1':
            z_next = self.encode(next_obs, goal)
            z_next_pair = z_next[perm]
            transition_dist = torch.norm(z_next - z_next_pair, p = 1, dim = 1)
        elif self.transition_model_type == 'deterministic':
            transition_dist = torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 1)
        elif self.transition_model_type == 'probabilistic':
            transition_dist = torch.sqrt(torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 1).pow(2) + torch.norm(pred_next_latent_sigma1 - pred_next_latent_sigma2, dim = 1).pow(2))
        elif self.transition_model_type == 'ensemble':
            transition_dist = torch.sqrt(torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 2).pow(2) + torch.norm(pred_next_latent_sigma1 - pred_next_latent_sigma2, dim = 2).pow(2))
            #transition_dist = transition_dist.unsqueeze(2)
        else:
            raise NotImplementedError

        bisimilarity = metric + self.discount * transition_dist
        if not self.dual_optimization:
            bisimilarity = bisimilarity.detach()

        loss = (z_dist - bisimilarity).pow(2).mean()

        return loss, std_norm, collapse_level

    def policy_decoder(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal).detach()
        pred_action = self.policy_decoder(z) #Inverse Model
        policy_decoder_loss = F.mse_loss(pred_action.squeeze(), action.squeeze())

        return policy_decoder_loss

    def transition_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.transition_model_type == 'next_observation' or self.transition_model_type == 'next_observation_l1':
            return torch.Tensor([0]).to(self.device)

        z = self.encode(obs, goal)
        dyn_input = z

        pred_next_latent_mu, pred_next_latent_sigma = self.dynamics_model(torch.cat([dyn_input, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        pred_next_latent_sigma_inv = torch.exp(-pred_next_latent_sigma)

        next_z = self.encode(next_obs, goal)
        if self.dynamics_loss == 'direct':
            diff = ((pred_next_latent_mu - next_z.detach()) ** 2 * pred_next_latent_sigma_inv) + pred_next_latent_sigma
        elif self.dynamics_loss == 'delta':
            diff = ((pred_next_latent_mu - (next_z.detach() - z.detach())) ** 2 * pred_next_latent_sigma_inv) + pred_next_latent_sigma
        else:
            raise NotImplementedError

        loss = torch.mean(diff)
 
        total_loss = loss
        if self.dynamics_loss == 'direct':
            stats = {'step' : step,
                beginning + '/phi/model_error' : torch.mean(torch.norm(pred_next_latent_mu - next_z.detach(), dim = 1)).item(),
                }
        elif self.dynamics_loss == 'delta':
            stats = {'step' : step,
                beginning + '/phi/model_error' : torch.mean(torch.norm(pred_next_latent_mu - (next_z.detach() - z.detach()), dim = 1)).item(),
                }
        else:
            raise NotImplementedError

        
        logger.logging_tool.log(stats)

        return total_loss 

    def decoder_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy and not self.decode_both:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy and not self.decode_both:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal)
        next_z = self.encode(next_obs, goal)
        if self.transition_model_type != 'next_observation':
            decodee = self.dynamics_model.sample_prediction(torch.cat([z, action], dim=1))
        else:
            decodee = next_z #Try not to use...

        if self.decode_both:
            decodee = torch.cat([z, decodee], dim = 1)

        if self.decoder_type == 'reward':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward.squeeze(), reward.squeeze())
        elif self.decoder_type == 'rtg':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward.squeeze(), rtg.squeeze())
        elif self.decoder_type == 'rtg_reward':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward[:,0], rtg.squeeze()) + F.mse_loss(pred_next_reward[:,1], reward.squeeze())
        elif self.decoder_type == 'action':
            pred_next_action = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_next_action.squeeze(), action.squeeze())
        elif self.decoder_type == 'temporal':
            pred_td = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_td.squeeze(), td.squeeze())
        elif self.decoder_type == 'temporal_action':
            pred_action_td = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_action_td[:,0], td.squeeze()) + F.mse_loss(pred_action_td[:,1:].squeeze(), action.squeeze())
        elif self.decoder_type =='none':
            decoder_loss = torch.Tensor([0]).to(self.device)
        else:
            raise NotImplementedError

        return decoder_loss

    def train_batch(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, log = True, take_step = True, beginning = 'train'):

        action = torch.clip(action, min = -1, max = 1) * self.action_scale

        #self.encoder.train()
        encoder_loss, std_norm, collapse_level = self.encoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, log = log, beginning = beginning)
        transition_loss = self.transition_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)
        decoder_loss = self.decoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)

        #policy_decoder_loss = self.policy_decoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)

        total_loss = self.encoder_weight * encoder_loss + self.transition_weight * (transition_loss + decoder_loss)

        if log:
            stats = {'step' : step,
            beginning + '/phi/loss' : total_loss.item(),
            beginning + '/phi/encoder_loss' : encoder_loss.item(),
            beginning + '/phi/transition_loss' : transition_loss.item(),
            beginning + '/phi/decoder_loss' : decoder_loss.item(),
            beginning + '/phi/std_norm' : std_norm,
            beginning + '/phi/collapse_level' : collapse_level
            }

            logger.logging_tool.log(stats)
        else:
            print("Loss_PHI: " ,total_loss.item())

        if take_step:
            assert beginning == 'train'

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            total_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        else:
            val_loss = total_loss.detach()

    def train_batches(self, dataset, batches=100):

        for b in range(batches):
            idxs = np.random.permutation(dataset.shape[0], self.train_batch_size)
            self.train_batch(dataset[idxs])

    def train_epoch(self, dataset):
        order = np.random.permutation(dataset.shape[0])
        iterations = dataset.shape[0] // self.train_batch_size

        for itr in range(iterations):
            self.train_batch(dataset[itr * self.train_batch_size: (itr + 1) * self.train_batch_size])

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_np(self, inputs, cont=True):
        assert False, "No decoder avaliable"

    def step_lr(self):
        self.encoder_optimizer_step.step()
        self.decoder_optimizer_step.step()

    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):
        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], kwargs['reward'], \
            kwargs['rtg'], kwargs['td'], policy, step, log = log, take_step = False, beginning = 'eval')

    def update(self, replay_buffer, policy, kwargs, step, log = True):
        #Will run through dataset...

         #Does it matter if not same batch....?

        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], \
            kwargs['reward'], kwargs['rtg'], kwargs['td'], policy, step, log = log)

        for _ in range(self.train_iters_per_update - 1):
            obs, action, _, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample()
            self.train_batch(obs, action, next_obs, goals, reward, kwargs['rtg'], kwargs['td'], policy, step, log = log)



