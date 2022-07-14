import torch
import torch.nn as nn
import torch.nn.functional as F
from goalbisim.representation.encoders.resnet import Encoder, VAEEncoder
from goalbisim.representation.decoders.resnet import Decoder
from goalbisim.representation.building_blocks.vq import VectorQuantizer, VectorQuantizerEMA
from rlkit.core import logger

import abc
import math
from collections import deque
import numpy as np

class VAE(nn.Module):
    def __init__(
            self,
            obs_shape, # (9, 64, 64)
            device,
            embedding_dim=1,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            min_variance=1e-3,

            lr=1e-3,
            weight_decay = 1e-3,
            train_iters_per_update = 1,
            ):
        super(VAE, self).__init__()
        self.imsize = obs_shape[1]
        self.embedding_dim = embedding_dim
        self.input_channels = obs_shape[0]
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.log_min_variance = float(np.log(min_variance))

        self.device = device
        
        self._encoder = Encoder(self.input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens).to(device)
        
        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1).to(device)

        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1).to(device)
        
        self._decoder = Decoder(self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=obs_shape[0]).to(device)

        # Calculate latent sizes
        if self.imsize == 32:
            self.root_len = 8
        elif self.imsize == 36:
            self.root_len = 9
        elif self.imsize == 48:
            self.root_len = 12
        elif self.imsize == 64:
            self.root_len = 16
        elif self.imsize == 84:
            self.root_len = 21
        elif self.imsize == 100:
            self.root_len = 30
        else:
            raise ValueError(self.imsize)

        self.representation_size = self.root_len * self.root_len * self.embedding_dim

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_iters_per_update = train_iters_per_update
        # Calculate latent sizes

        self.beta_schedule = PiecewiseLinearSchedule(x_values=(0, 750000), y_values=(0, 50))
        self.encoder = VAEEncoder(
            self._encoder, self.f_mu, self.f_logvar, 
            self.input_channels, self.imsize, self.representation_size, self.log_min_variance
        ).to(device)

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z_s, kle = self.encode(inputs, computing_loss=True)
        recon = self.decode(z_s)

        recon_error = F.mse_loss(recon, inputs, reduction='sum')
        return recon, recon_error, kle

    def compute_loss_trainer(self, obs, step, log=True, beginning='train'):
        prefix = beginning + "/vae/"
        beta = float(self.beta_schedule.get_value(step))

        recon, error, kle = self.compute_loss(obs)
        loss = error + beta * kle

        if log:
            stats = {
                'step' : step,
                'beta' : beta,
                prefix + 'loss' : loss.item(),
                prefix + 'kle': kle.item(),
                prefix + 'Obs Recon Error': error.item(),
            }
            logger.logging_tool.log(stats)

        return loss

    def forward(self, obs, detach=False, detach_all=False):
        return self.encoder.encode(obs, detach=detach, detach_all=detach_all)

    def encode(self, inputs, detach=False, detach_all=False, computing_loss=False):
        return self.encoder.encode(inputs, detach=detach, detach_all=detach_all, computing_loss=computing_loss)

    def decode(self, latents):
        z_s = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        return self._decoder(z_s)

    def encode_one_np(self, inputs):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs)))[0]

    def encode_np(self, inputs):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs)))

    def decode_one_np(self, inputs):
        return np.clip(ptu.get_numpy(
            self.decode(ptu.from_numpy(inputs).reshape(1, -1)))[0],
            0, 1)

    def decode_np(self, inputs):
        return np.clip(
            ptu.get_numpy(self.decode(ptu.from_numpy(inputs))), 0, 1)

    def train_batch(self, obs, step, log = True, take_step = True, beginning = 'train'):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss_trainer(obs, step, log=log, beginning=beginning)

        if take_step:
            assert beginning == 'train'
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            val_loss = loss.detach()
    
    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):
         self.train_batch(kwargs['obs'], step, log = log, take_step = False, beginning = 'eval')

    def update(self, replay_buffer, policy, kwargs, step, log = True):
        #Will run through dataset...
        self.train_batch(kwargs['obs'], step, log = log)

        for _ in range(self.train_iters_per_update - 1):
            obs, _, _, _, _, _, _, _ = replay_buffer.sample()
            self.train_batch(obs, step, log = log)

class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass

class PiecewiseLinearSchedule(ScalarSchedule):
    """
    Given a list of (x, t) value-time pairs, return value x at time t,
    and linearly interpolate between the two
    """
    def __init__(
            self,
            x_values,
            y_values,
    ):
        self._x_values = x_values
        self._y_values = y_values

    def get_value(self, t):
        return np.interp(t, self._x_values, self._y_values)