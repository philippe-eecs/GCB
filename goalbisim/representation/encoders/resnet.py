import torch
import torch.nn as nn
import torch.nn.functional as F
from goalbisim.representation.building_blocks.resnetblocks import ResidualStack

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)
        
        self.convs = [self._conv_1, self._conv_2, self._conv_3] + [layer._block[1] for layer in list(self._residual_stack._layers)] \
        + [layer._block[3] for layer in list(self._residual_stack._layers)]

    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)

class VQVAEEncoder(nn.Module):
    def __init__(self, _encoder, _pre_vq_conv, _vq_vae, input_channels, imsize, representation_size, discrete_size):
        super(VQVAEEncoder, self).__init__()

        self._encoder = _encoder
        self._pre_vq_conv = _pre_vq_conv
        self._vq_vae = _vq_vae

        self.feature_dim = representation_size
        self.conv_layers = self._encoder.convs + [self._pre_vq_conv]

        self.input_channels = input_channels
        self.imsize = imsize
        self.representation_size = representation_size
        self.discrete_size = discrete_size
    
    def forward(self, inputs, detach=False, detach_all=False):
        return self.encode(inputs, detach=detach, detach_all=detach_all)

    def quantize_image(self, inputs, detach=False, detach_all=False):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        if detach or detach_all:
            z = z.detach()

        out = self._vq_vae(z)
        if detach_all:
            out_detach = []
            for i in range(len(list(out))):
                out_detach.append(out[i].detach())
            out = tuple(out_detach)
        return out

    def encode(self, inputs, detach=False, detach_all=False, cont=True):
        _, quantized, _, encodings = self.quantize_image(inputs, detach=detach, detach_all=detach_all)

        if cont:
            return quantized.reshape(-1, self.representation_size)
        return encodings.reshape(-1, self.discrete_size)

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(len(self.conv_layers)):
            tie_weights(src=source.conv_layers[i], trg=self.conv_layers[i])

class VAEEncoder(nn.Module):
    def __init__(self, _encoder, f_mu, f_logvar, input_channels, imsize, representation_size, log_min_variance):
        super(VAEEncoder, self).__init__()

        self._encoder = _encoder
        self.f_mu = f_mu
        self.f_logvar = f_logvar

        self.feature_dim = representation_size
        self.conv_layers = self._encoder.convs + [self.f_mu] + [self.f_logvar]

        self.input_channels = input_channels
        self.imsize = imsize
        self.representation_size = representation_size
        self.log_min_variance = log_min_variance
    
    def forward(self, inputs, detach=False, detach_all=False):
        return self.encode(inputs, detach=detach, detach_all=detach_all)

    def encode(self, inputs, detach=False, detach_all=False, computing_loss=False):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z_conv = self._encoder(inputs)
        if detach or detach_all:
            z_conv = z_conv.detach()
        
        mu = self.f_mu(z_conv).reshape(-1, self.representation_size)
        unclipped_logvar = self.f_logvar(z_conv).reshape(-1, self.representation_size)
        logvar = self.log_min_variance + torch.abs(unclipped_logvar)
        
        if self.training:
            z_s = self.rsample(mu, logvar)
        else:
            z_s = mu
        
        if detach_all:
            z_s = z_s.detach()

        if computing_loss:
            return z_s, self.kl_divergence(mu, logvar)
        return z_s

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(len(self.conv_layers)):
            tie_weights(src=source.conv_layers[i], trg=self.conv_layers[i])

class CCVAEEncoder(nn.Module):
    def __init__(self, _encoder, _cond_encoder, f_mu, f_logvar, _conv_cond, input_channels, imsize, latent_size, representation_size, log_min_variance):
        super(CCVAEEncoder, self).__init__()

        self._encoder = _encoder
        self._cond_encoder = _cond_encoder
        self.f_mu = f_mu
        self.f_logvar = f_logvar
        self._conv_cond = _conv_cond

        self.feature_dim = representation_size
        self.conv_layers = self._encoder.convs + self._cond_encoder.convs + [self.f_mu, self.f_logvar, self._conv_cond]

        self.input_channels = input_channels
        self.imsize = imsize
        self.latent_size = latent_size
        self.representation_size = representation_size
        self.log_min_variance = log_min_variance
    
    def forward(self, inputs, detach=False, detach_all=False):
        return self.encode(inputs, detach=detach, detach_all=detach_all)

    def encode(self, obs, detach=False, detach_all=False, computing_loss=False):
        x_delta = obs[:,:self.input_channels]
        x_cond = obs[:,self.input_channels:]

        z_conv = self._encoder(obs)
        cond_conv = self._cond_encoder(x_cond)

        if detach or detach_all:
            z_conv = z_conv.detach()
            cond_conv = cond_conv.detach()

        mu = self.f_mu(z_conv).reshape(-1, self.latent_size)
        unclipped_logvar = self.f_logvar(z_conv).reshape(-1, self.latent_size)
        z_cond = self._conv_cond(cond_conv).reshape(-1, self.latent_size)
        logvar = self.log_min_variance + torch.abs(unclipped_logvar)
        
        if self.training:
            z_s = self.rsample(mu, logvar)
        else:
            z_s = mu

        z_cat = torch.cat([z_s, z_cond], dim=1)

        if detach_all:
            z_cat = z_cat.detach()

        if computing_loss:
            return z_cat, z_cond, self.kl_divergence(mu, logvar)
        return z_cat

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(len(self.conv_layers)):
            tie_weights(src=source.conv_layers[i], trg=self.conv_layers[i])