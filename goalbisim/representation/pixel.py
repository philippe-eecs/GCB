import torch
import torch.nn as nn
import numpy as np
from goalbisim.representation.encoders.RADencoder import PixelEncoder

class PixelRepresentation(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            obs_proper = 54):
        super().__init__()

        obs_input = (obs_shape[0], obs_proper, obs_proper)
        self.encoder = PixelEncoder(obs_input, feature_dim, num_layers, num_filters).to(device)
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, obs, detach=False):
        return self.encode(obs, detach = detach)

    def encode(self, obs, detach=False):
        z_out = self.encoder(obs, detach = detach)

        return z_out

    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):
        return

    def update(self, replay_buffer, policy, kwargs, step, log = True):
        return