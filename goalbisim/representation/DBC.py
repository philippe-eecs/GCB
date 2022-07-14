import torch
import torch.nn as nn
from goalbisim.representation.encoders.RADencoder import PixelEncoder
from goalbisim.logging.logging import logger
import numpy as np
import wandb
import math
import torch.nn.functional as F
from goalbisim.dynamics.dynamics_models import make_transition_model
#from wandb



class DBC(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type = 'ensemble',
            discount = 0.99,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            lr=1e-3,
            weight_decay = 0,
            train_batch_size = 256):
        super().__init__()

        self.device = device

        self.encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters).to(device)
        #self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        #We should probably standarize the name for dimension...
        self.feature_dim = feature_dim
        #self.preprocess = preprocess

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.discount = discount

        self.transition_model_type = transition_model_type

        self.reward_decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        self.dynamics_model = make_transition_model(transition_model_type, feature_dim, (2, 1)).to(device)

        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.dynamics_model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        #Should Include W weights

        #self.cross_entropy = nn.CrossEntropyLoss()

        self.mse = nn.MSELoss()

        #self.kl_loss = nn.KLDivLoss()


        self.train_batch_size = train_batch_size

    def forward(self, obs, detach=False):
        return self.encode(obs, detach = detach)

    def encode(self, obs, detach=False):

        z_out = self.encoder(obs, detach = detach)

        return z_out

    def encoder_loss(self, obs, action, next_obs, reward, curr_reward, step, log = True):
        #import pdb; pdb.set_trace()
        z = self.encode(obs)
        perm = np.random.permutation(obs.shape[0])
        z2 = z[perm]


        norms = torch.norm(z.detach(), dim = 1)
        output_norm = torch.nn.functional.normalize(z.detach(), dim = 1)
        output_std = torch.std(output_norm, 0).mean().item()
        collapse_level = max(0., 1 - math.sqrt(self.feature_dim) * output_std)
        std_norm = torch.std(norms)

        if log:
            stats = {'step' : step,
            'train/phi/norm_std' : std_norm,
            'train/phi/collapse_level' : collapse_level
            }

            logger.record_dict(stats)
            wandb.log(stats)
        #print("Collapse_Level_Phi: ", collapse_level, std_norm )
        #print("Norm_STD_PHI: ", std_norm)

        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.dynamics_model(torch.cat([z, action], dim=1))
            reward2 = reward[perm]

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


        z_dist = F.smooth_l1_loss(z, z2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')

        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            transition_dist = torch.sqrt(
                (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
            )

        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()

        return loss

    def transition_loss(self, obs, action, next_obs, reward, step):
        h = self.encode(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.dynamics_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.encode(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        pred_next_latent = self.dynamics_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return total_loss      
    

    def train_batch(self, obs, action, next_obs, reward, curr_rewards, step, log = True):

        encoder_loss = self.encoder_loss(obs, action, next_obs, reward, curr_rewards, step, log = log)
        transition_loss = self.transition_loss(obs, action, next_obs, reward, step)

        total_loss = 0.5 * encoder_loss + transition_loss

        if log:
            stats = {'step' : step,
            'train/phi/loss' : total_loss.item(),
            'train/phi/encoder_loss' : encoder_loss.item(),
            'train/phi/transition_loss' : transition_loss.item()
            }

            logger.record_dict(stats)
            wandb.log(stats)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


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

    def update(self, replay_buffer, kwargs, step, log = True):
        #Will run through dataset...

        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], \
         kwargs['reward'], kwargs['curr_rewards'], step, log = log)



