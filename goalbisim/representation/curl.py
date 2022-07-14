import torch
import torch.nn as nn
from goalbisim.representation.encoders.RADencoder import PixelEncoder
from rlkit.core import logger
#from wandb


class CURL(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            lr=1e-3,
            weight_decay = 1e-3,
            obs_proper = 54,
            temporal = False):
        super().__init__()

        obs_input = (obs_shape[0], obs_proper, obs_proper)
        self.encoder = PixelEncoder(obs_input, feature_dim, num_layers, num_filters).to(device)
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim)).to(device)
        self.feature_dim = feature_dim

        self.device = device

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temporal = temporal

    def forward(self, obs, detach=False):
        return self.encode(obs, detach = detach)

    def encode(self, obs, detach=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        z_out = self.encoder(obs, detach = detach)

        return z_out

    def loss(self, obs_anchor, obs_positive):
        z_a = self.encode(obs_anchor)
        z_pos = self.encode(obs_positive)
        
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy(logits, labels)
        
        return loss
    
    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.t())  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def train_batch(self, obs_anchor, obs_positive, step, log = True, take_step = True, beginning = 'train'):

        loss = self.loss(obs_anchor, obs_positive)

        if log:
            stats = {'step' : step,
            'train/encoder/loss' : loss.item()
            }


            logger.logging_tool.log(stats)


        if take_step:
            assert beginning == 'train'
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss = loss.detach()
            stats = {'step' : step,
            'eval/encoder/loss' : loss.item()
            }

    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):

        if self.temporal:
            self.train_batch(kwargs['obs'], kwargs['next_obs'], step, log = log, take_step = False, beginning = 'eval')
        else:
            self.train_batch(kwargs['obs'], kwargs['pos'], step, log = log, take_step = False, beginning = 'eval')

    def update(self, replay_buffer, policy, kwargs, step, log = True):

        if self.temporal:
            self.train_batch(kwargs['obs'], kwargs['next_obs'], step, log = log)
        else:
            self.train_batch(kwargs['obs'], kwargs['pos'], step, log = log)
