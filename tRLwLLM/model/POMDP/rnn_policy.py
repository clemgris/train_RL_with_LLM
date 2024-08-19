import functools
from typing import Dict, Optional, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1])),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=64)(carry, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(input_size):
        # Use a dummy key since the default state init fn is just zeros.
        batch_size, hidden_size = input_size
        return nn.GRUCell(features=64).initialize_carry(
            jax.random.key(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: int
    discrete: bool
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]
    num_action: int
    num_components: int = 6

    @nn.compact
    def __call__(self, rnn_state, x):
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor_class(**self.feature_extractor_kwargs)(
            obs
        )

        # RNN
        rnn_in = (state_features, dones)
        new_rnn_state, x = ScannedRNN()(rnn_state, rnn_in)

        # Actor
        x_cat = jnp.concatenate((x, state_features), axis=-1)

        x_actor = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x_cat
        )
        x_actor = jax.nn.relu(x_actor)
        # x_actor = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x_actor)
        # x_actor = jax.nn.relu(x_actor)

        # DISCRET ACTION SPACE
        if self.discrete:
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(x_actor)
            actor_mean = jax.nn.relu(actor_mean)
            actor_mean = nn.Dense(
                self.action_dim * self.num_action,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = actor_mean.reshape(
                (*actor_mean.shape[:2], self.action_dim, self.num_action)
            )

            pi = distrax.Categorical(logits=actor_mean)
            # weights = None
            # actor_std = None

        # # CONTINUOUS ACTION SPACE (GAUSSIAN MIXTURE)
        # else:
        #     weights = nn.Dense(64, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x_actor)
        #     weights = jax.nn.relu(weights)
        #     weights = nn.Dense(self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(weights)
        #     weights = weights.reshape((*weights.shape[:2], 1, self.num_components))

        #     actor_mean = nn.Dense(64, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x_actor)
        #     actor_mean = jax.nn.relu(actor_mean)
        #     actor_mean = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        #     actor_mean = actor_mean.reshape((*actor_mean.shape[:2], self.action_dim, self.num_components))

        #     actor_log_std = nn.Dense(64, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x_actor)
        #     actor_log_std = jax.nn.relu(actor_log_std)
        #     actor_log_std = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_log_std)
        #     # actor_std = nn.softplus(actor_std)

        #     actor_log_std = jnp.clip(actor_log_std, a_min=-5, a_max=2)
        #     actor_std = jnp.exp(actor_log_std)
        #     actor_std = actor_std.reshape((*actor_std.shape[:2], self.action_dim, self.num_components))

        #     pi = tfd.MixtureSameFamily(
        #             mixture_distribution=tfd.Categorical(logits=weights),
        #             components_distribution=tfd.Normal(loc=actor_mean, scale=actor_std))

        # Critic
        x_critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x_cat
        )
        x_critic = jax.nn.relu(x_critic)
        # x_critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0))(x_critic)
        # x_critic = jax.nn.relu(x_critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            x_critic
        )

        return new_rnn_state, pi, jnp.squeeze(critic, axis=-1)
