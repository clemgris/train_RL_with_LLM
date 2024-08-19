import json
import os
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import time
import flax
import flashbax as fbx

from tRLwLLM.environment import BabyAI
from tRLwLLM.utils import TransitionDQN, concatenate_dicts, concatenate_transitions

from ..feature_extractor import KeyExtractor
from ..obs_preprocessing import ExtractObs
from .policy import QNetwork


def extand(x):
    if isinstance(x, jnp.ndarray):
        return x[jnp.newaxis, ...]
    else:
        return

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


class make_train_dqn:
    def __init__(self, config):
        self.config = config

        # DEVICE
        self.devices = jax.devices()
        print(f"Available devices: {self.devices}")

        # RANDOM SEED
        self.key = jax.random.PRNGKey(self.config["key"])

        # NUMBER OF GRADIENT UPDATES
        self.config["num_updates"] = int(
            self.config["total_timesteps"]
            // self.config["num_envs"]
        )

        # ENVIRONMENT
        self.env = BabyAI(self.config)

        self.config["rf_height"] = self.env._env.agent_view_size
        self.config["rf_width"] = self.env._env.agent_view_size
        self.config["height"] = self.env._env.height
        self.config["width"] = self.env._env.width

    def train(
        self,
    ):

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=self.config["buffer_size"],
            min_length=self.config["buffer_batch_size"],
            sample_batch_size=self.config["buffer_batch_size"],
            add_sequences=False,
            add_batch_size=self.config["num_envs"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # INIT NETWORK AND OPTIMIZER
        self.extractor = ExtractObs(self.config)
        self.feature_extractor = KeyExtractor
        self.feature_extractor_kwargs = self.config["feature_extractor_kwargs"]

        network = QNetwork(
            feature_extractor_class=self.feature_extractor,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            num_action=self.env._env.action_space.n,
        )
        init_x = self.extractor.init_x(self.config["num_envs"])
        network_params = network.init(self.key, init_x[0])

        # Count number of parameters
        flat_params, _ = jax.tree_util.tree_flatten(network_params)
        network_size = sum(p.size for p in flat_params)
        print(f"Number of parameters: {network_size}")

        # INIT BUFFER
        obs, _ = self.env.reset()
        last_obsv = self.extractor(
            obs, None, jnp.ones((self.config["num_envs"]), dtype=bool)
        )
        action = jnp.zeros((self.config["num_envs"], 1), dtype='int32').squeeze(-1)
        obs, reward, done, info = self.env.step(action)
        obsv = self.extractor(obs, last_obsv, done)
        init_timestep = TransitionDQN(done, action, reward, obsv, info)
        buffer_state = buffer.init(init_timestep)
        breakpoint()

        # RESET ENV
        obs, _ = self.env.reset()
        obsv = self.extractor(
            obs, None, jnp.ones((self.config["num_envs"]), dtype=bool)
        )

        def linear_schedule(count):
            frac = 1.0 - (count / self.config["num_updates"])
            return self.config["lr"] * frac

        lr = (
            linear_schedule
            if self.config.get("lr_decay_linear", False)
            else self.config["lr"]
        )
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_map(
                lambda x: jnp.copy(x), network_params
            ),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (self.config["epsilon_finish"] - self.config["epsilon_start"])
                    / self.config["epsilon_anneal_time"]
                )
                * t
                + self.config["epsilon_start"],
                self.config["epsilon_finish"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a,
                    shape=greedy_actions.shape,
                    minval=0,
                    maxval=q_vals.shape[-1],
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # DQN LOSS
        def _loss_fn(params, learn_batch, target):
            q_vals = network.apply(
                params, learn_batch.first.obs
            )  # (batch_size, num_actions)
            chosen_action_qvals = jnp.take_along_axis(
                q_vals,
                jnp.expand_dims(learn_batch.first.action, axis=-1),
                axis=-1,
            ).squeeze(axis=-1)
            return jnp.mean((chosen_action_qvals - target) ** 2)

        # NETWORKS UPDATE
        def _learn_phase(train_state, rng):

            learn_batch = buffer.sample(buffer_state, rng).experience

            q_next_target = network.apply(
                train_state.target_network_params, learn_batch.second.obs
            )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            target = (
                learn_batch.first.reward
                + (1 - learn_batch.first.done) * self.config["gamma"] * q_next_target
            )

            loss, grads = jax.value_and_grad(_loss_fn)(
                train_state.params, learn_batch, target
            )
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            return train_state, loss

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, last_obsv, rng = runner_state

            # STEP THE ENV
            rng, rng_sample_action = jax.random.split(rng)
            q_vals = network.apply(
                train_state.params, jax.tree_map(extand, last_obsv)
            )
            action = eps_greedy_exploration(
                rng_sample_action, q_vals, train_state.timesteps
            ).squeeze(0)  # explore with epsilon greedy_exploration

            obs, reward, done, info = self.env.step(action)
            obsv = self.extractor(obs, last_obsv, done)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + self.config["num_envs"]
            )  # update timesteps count

            # UPDATE BUFFER
            timestep = TransitionDQN(
                done,
                action,
                reward,
                last_obsv,
                info,
            )
            breakpoint()
            buffer_state = buffer.add(buffer_state, timestep)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > self.config["learning_starts"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % self.config["training_interval"] == 0
                )  # training interval
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (
                    train_state,
                    jnp.array(0.0),
                ),  # do nothing
                train_state,
                _rng,
            )

            # UPDATE TARGET NETWORK
            train_state = jax.lax.cond(
                train_state.timesteps % self.config["target_update_interval"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        self.config["tau"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            runner_state = (train_state, buffer_state, obsv, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(self.key)
        runner_state = (train_state, buffer_state, obsv, _rng)

        metrics = []
        for update in range(self.config["num_updates"]):
            time_start = time.time()
            runner_state, train_metric = _update_step(runner_state, None)

            metrics.append(jax.tree_map(lambda x: jnp.mean(x), train_metric))

            train_message = f"Update | {update}/{self.config["num_updates"]} | Train | "
            for key, value in train_metric.items():
                train_message += f" {key} | {jnp.array(value).mean():.6f} | "

            train_message += f" Time | {(time.time() - time_start):0.4f} s"

            print(train_message)

            if (update % self.config["freq_save"] == 0) or (
                update == self.config["num_updates"] - 1
            ):
                past_log_metric = os.path.join(
                    self.config["log_folder"],
                    f'training_metrics_{update - self.config["freq_save"]}.pkl',
                )
                past_log_params = os.path.join(
                    self.config["log_folder"],
                    f'params_{update - self.config["freq_save"]}.pkl',
                )

                if os.path.exists(past_log_metric):
                    os.remove(past_log_metric)

                if os.path.exists(past_log_params):
                    os.remove(past_log_params)

                # Checkpoint
                with open(
                    os.path.join(
                        self.config["log_folder"], f"training_metrics_{update}.pkl"
                    ),
                    "wb",
                ) as pkl_file:
                    pickle.dump(concatenate_dicts(metrics), pkl_file)

                # Save model weights
                with open(
                    os.path.join(self.config["log_folder"], f"params_{update}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(runner_state[0].params, f)

            with open(
                os.path.join(self.config["log_folder"], "args.json"), "w"
            ) as json_file:
                json.dump(self.config, json_file, indent=4)

            # Stop environment
            self.env._env.stop()

            return {
            "runner_state": runner_state,
            "metric": concatenate_dicts(metrics),
            "self.config": self.config,
            }
