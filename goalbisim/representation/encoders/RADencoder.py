import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations for RL."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=True, tanh_scale = 1, goal_flag = False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.tanh_scale = tanh_scale
        self.num_filters = num_filters
        self.goal_flag = goal_flag

        if goal_flag:
            scale = 2
        else:
            scale = 1
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0] * scale, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        #out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 

        out_dim = 39200

        if num_layers == 4 and num_filters == 32 and obs_shape[1] == 50:
            out_dim = 10368
        elif num_layers == 6 and num_filters == 64 and obs_shape[1] == 100:
            out_dim = 61504
        elif num_layers == 4 and num_filters == 32 and obs_shape[1] == 48:
            out_dim = 9248
        elif num_layers == 4 and num_filters == 32 and obs_shape[1] == 64:
            out_dim = 20000
        elif num_layers == 4 and num_filters == 64 and obs_shape[1] == 100:
            out_dim = 78400
        elif num_layers == 6 and num_filters == 32 and obs_shape[1] == 100:
            out_dim = 48672
        elif num_layers == 4 and num_filters == 32 and obs_shape[1] == 100:
            out_dim = 59168
        elif num_layers == 5 and num_filters == 32 and obs_shape[1] == 64:
            out_dim = 20000
        elif num_layers == 6 and num_filters == 32 and obs_shape[1] == 64:
            out_dim = 14112
        elif num_layers == 6 and num_filters == 32 and obs_shape[1] == 54:
            out_dim = 8192
        elif num_layers == 6 and num_filters == 32 and obs_shape[1] == 70:
            out_dim = 18432
        else:
            raise NotImplementedError
           # out_dim = 39200
        #if goal_flag:
            #out_dim *= 2

        self.fc = nn.Linear(out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.W_contrast = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.W_map = nn.Parameter(torch.rand(feature_dim, feature_dim))

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False, detach_all=False):
        assert obs.max() <=1 and obs.min() >= 0, "Something wrong with encoder input"
        h = self.forward_conv(obs)

        if detach or detach_all:
            h = h.detach()

        #print(h.shape)
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm) * self.tanh_scale
            self.outputs['tanh'] = out

        if detach_all:
            out = out.detach()

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )