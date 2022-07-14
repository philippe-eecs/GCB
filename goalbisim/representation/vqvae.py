import torch
import torch.nn as nn
import torch.nn.functional as F
from goalbisim.representation.encoders.resnet import Encoder, VQVAEEncoder
from goalbisim.representation.decoders.resnet import Decoder
from goalbisim.representation.building_blocks.vq import VectorQuantizer, VectorQuantizerEMA
from rlkit.core import logger

class VQVAE(nn.Module):
    def __init__(
            self,
            obs_shape, # (9, 64, 64)
            device,
            embedding_dim=5,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            num_embeddings=512,
            
            commitment_cost=0.25,
            decay=0.0,
            lr=1e-3,
            weight_decay = 1e-3,
            train_iters_per_update = 1,
        ):
        super(VQVAE, self).__init__()
        self.imsize = obs_shape[1]
        self.embedding_dim = embedding_dim
        self.input_channels = obs_shape[0]
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.num_embeddings = num_embeddings

        self._encoder = Encoder(self.input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens).to(device)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1).to(device)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings,
                self.embedding_dim,
                commitment_cost, decay).to(device)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                commitment_cost).to(device)

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

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_iters_per_update = train_iters_per_update
        # Calculate latent sizes

        self.encoder = VQVAEEncoder(
            self._encoder, self._pre_vq_conv, self._vq_vae, 
            self.input_channels, self.imsize, self.representation_size, self.discrete_size
        ).to(device)

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        vq_loss, quantized, perplexity, _ = self.quantize_image(inputs)
        recon = self.decode(quantized)

        recon_error = F.mse_loss(recon, inputs)
        return vq_loss, recon, perplexity, recon_error

    def compute_loss_trainer(self, obs, step, log=True, beginning='train'):
        prefix = beginning + "/vqvae/"

        vq_loss, data_recon, perplexity, recon_error = self.compute_loss(obs)
        loss = vq_loss + recon_error

        if log:
            stats = {
                'step' : step,
                prefix + 'loss' : loss.item(),
                prefix + 'recon error': recon_error.item(),
                prefix + 'vq loss': vq_loss.item(),
                prefix + 'perplexity': perplexity.item(),
            }
            logger.logging_tool.log(stats)

        return loss

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
            out = out.detach()
        return out

    def forward(self, obs, detach=False, detach_all=False):
        return self.encoder.encode(obs, detach=detach, detach_all=detach_all)

    def encode(self, inputs, detach=False, detach_all=False, cont=True):
        return self.encoder.encode(inputs, detach=detach, detach_all=detach_all)

    def latent_to_square(self, latents):
        return latents.reshape(-1, self.root_len, self.root_len)

    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings,
            device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def decode(self, latents, cont=True):
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len,
                self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)

    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(
            self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0],
            0, 1)

    def decode_np(self, inputs, cont=True):
        return np.clip(
            ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)

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