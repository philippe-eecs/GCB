import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.nn as jnn
import optax

import pdb

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

ImageBatch = collections.namedtuple(
    'ImageBatch',
    ['observations', 'image_observations', 'actions', 'rewards', 'masks', 'next_observations', 'next_image_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, image):
        image_x = nn.Conv(features=16, kernel_size=(3, 3))(image)
        image_x = nn.relu(image_x)
        image_x = nn.avg_pool(image_x, window_shape=(2, 2), strides=(2, 2))
        image_x = nn.Conv(features=64, kernel_size=(3, 3))(image_x)
        image_x = nn.relu(image_x)
        image_x = nn.avg_pool(image_x, window_shape=(2, 2), strides=(2, 2))
        image_x = nn.Conv(features=128, kernel_size=(3, 3))(image_x)
        image_x = nn.relu(image_x)
        image_x = nn.avg_pool(image_x, window_shape=(2, 2), strides=(2, 2))
        image_x = image_x.reshape((image_x.shape[0], -1))  # flatten
        image_x = nn.Dense(features=64)(image_x)
        image_x = nn.relu(image_x)
        image_x = nn.Dense(features=10)(image_x)
        image_x = nn.log_softmax(image_x)
        return image_x

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

class InvertedResidual(nn.Module):
    """A Inverted Residual block for mobilenetv2."""
    input_channels: int
    output_channels: int
    expand_ratio: int
    stride: int

    @nn.compact
    def __call__(self, image: jnp.ndarray, training: bool = False) -> jnp.ndarray:
      
        hidden_dim = self.input_channels*self.expand_ratio

        use_res_connect = (self.stride == 1 and self.input_channels == self.output_channels)

        if self.expand_ratio != 1:
            image_x = nn.Conv(features=hidden_dim, kernel_size=(1, 1), strides =(1, 1), use_bias=False)(image)
            image_x = nn.LayerNorm()(image_x)
            image_x = jnn.relu6(image_x)
        else:
            image_x = image

        image_x = nn.Conv(features=hidden_dim, kernel_size=(3, 3), strides=(self.stride, self.stride),
                            feature_group_count=hidden_dim, use_bias=False)(image_x)
        image_x = nn.LayerNorm()(image_x)
        image_x = jnn.relu6(image_x)

        image_x = nn.Conv(features=self.output_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(image_x)
        image_x = nn.LayerNorm()(image_x)

        if use_res_connect:
            return jnp.add(image_x, image)
        else:
            return image_x 

class mobilenetv2(nn.Module):
    """A mobilenetv2. implementation based on the pytorch mobilenetv2 network"""

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        image_x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False)(image)
        image_x = nn.LayerNorm()(image_x)
        image_x = jnn.relu6(image_x)
        input_channels = 32

        for t, c, n, s in inverted_residual_setting:
            output_channels = c
            for i in range(n):
                stride = s if i == 0 else 1
                image_x = InvertedResidual(input_channels=input_channels, output_channels=output_channels,
                                            expand_ratio=t, stride=stride)(image_x)
                input_channels=output_channels
        

        image_x = nn.Conv(features=1280, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(image_x)
        image_x = nn.LayerNorm()(image_x)
        image_x = jnn.relu6(image_x)
        image_x = nn.avg_pool(image_x, (2, 3))
        return image_x

class ConcatCNN(nn.Module):
    """A Concat CNN model."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, image: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        
        #image_x = mobilenetv2()(image)

        image_x = mobilenetv2()(image)
        image_x = image_x.reshape((image_x.shape[0], -1))  # flatten
        x = jnp.concatenate([image_x, x], -1)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':

        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
