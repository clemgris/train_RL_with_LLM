import functools
from typing import Dict, Optional, Union, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ActorCritic(nn.Module):
    num_action: Sequence[int]
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # State feature extractor
        state_features = self.feature_extractor_class(**self.feature_extractor_kwargs)(
            x
        )
        # Actor
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_features)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.num_action, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_features)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class QNetwork(nn.Module):
    num_action: Sequence[int]
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]

    @nn.compact
    def __call__(self, x):

        # State feature extractor
        state_features = self.feature_extractor_class(**self.feature_extractor_kwargs)(
            x
        )
        x = nn.Dense(512)(state_features)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_action)(x)
        return x
