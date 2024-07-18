from typing import Dict, List

from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ConvNet(nn.Module):
    final_hidden_layers: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = jax.nn.relu(x)
        # x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        # x = jax.nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = jax.nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        T, B, _, _, _ = x.shape
        x = x.reshape((T, B, -1))  # Flatten
        x = nn.Dense(features=self.final_hidden_layers)(x)
        x = jax.nn.relu(x)

        return x


@dataclass
class Identity:
    final_hidden_layers: int = None

    def __call__(self, x):
        return x


FEATURES_EXTRACTOR_DICT = {"im_dir": ConvNet, "mission": Identity}


class KeyExtractor(nn.Module):
    final_hidden_layers: int
    keys: List
    kwargs: Dict = None
    hidden_layers: Dict = None

    @nn.compact
    def __call__(self, obs):
        outputs = []
        for key in self.keys:
            if "[" in key:
                inputs = []
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
                for subkey in list_subkeys:
                    input = obs[subkey]
                    T, B = input.shape[:2]
                    input = input.reshape((T, B, -1))
                    inputs.append(input)
                concat_inputs = jnp.concatenate(inputs, axis=-1)
                x = FEATURES_EXTRACTOR_DICT[key](self.hidden_layers.get(key, None))(
                    concat_inputs
                )
            else:
                x = FEATURES_EXTRACTOR_DICT[key](self.hidden_layers.get(key, None))(
                    obs[key]
                )
            x = nn.LayerNorm()(x)  # Layer norm
            outputs.append(x)

        flattened = jnp.concatenate(outputs, axis=-1)

        output = nn.Dense(
            self.final_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(flattened)
        output = jax.nn.relu(output)
        output = nn.Dense(
            self.final_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(output)
        output = jax.nn.relu(output)
        return output
