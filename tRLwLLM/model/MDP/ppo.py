import json
import os
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import time

from tRLwLLM.environment import BabyAI
from tRLwLLM.utils import TransitionRL, concatenate_dicts, concatenate_transitions

from ..feature_extractor import KeyExtractor
from ..obs_preprocessing import ExtractObs
from .policy import ActorCriticRNN


def make_train_rl(config):
    def __init__(self, config):
        self.config = config

        # DEVICE
        self.devices = jax.devices()
        print(f"Available devices: {self.devices}")

        # RANDOM SEED
        self.key = jax.random.PRNGKey(self.config["key"])

        # NUMBER OF GRADIENT UPDATES
        self.config["num_updates"] = int(
            self.config["total_time_steps"]
            // self.config["num_steps"]
            // self.config["num_envs"]
        )
        self.config["MINIBATCH_SIZE"] = (
            self.config["num_envs"]
            * self.config["num_steps"]
            // self.config["num_minibatchs"]
        )

        # ENVIRONMENT
        self.env = BabyAI(self.config)

        self.config["height"] = self.env._env.height
        self.config["width"] = self.env._env.width

    # LEARNING RATE SCHEDULER
    def linear_schedule(self, count):
        frac = (
            1.0
            - (count // (self.config["num_minibatchs"] * self.config["update_epochs"]))
            / self.config["num_updates"]
        )
        return self.config["learning_rate"] * frac

    def train(
        self,
    ):
        # INIT NETWORK
        self.extractor = ExtractObs(self.config)
        self.feature_extractor = KeyExtractor
        self.feature_extractor_kwargs = self.config["feature_extractor_kwargs"]

        network = ActorCriticRNN(
            action_dim=1,
            discrete=True,
            feature_extractor_class=self.feature_extractor,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            num_action=self.env._env.action_space.n,
            num_components=None,
        )

        init_x = self.extractor.init_x(self.config["num_envs"])

        network_params = network.init(self.key, init_x)

        # Count number of parameters
        flat_params, _ = jax.tree_util.tree_flatten(network_params)
        network_size = sum(p.size for p in flat_params)
        print(f"Number of parameters: {network_size}")

        if self.config["aneal_learning_rate"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["learning_rate"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        obs, _ = self.env.reset()
        obsv = self.extractor(
            obs, None, jnp.ones((self.config["num_envs"]), dtype=bool)
        )

        # COLLECT TRAJECTORIES
        def _env_step(runner_state):
            train_state, last_obsv, last_done, _, rng = runner_state

            # SELECT ACTION
            pi, value = network.apply(train_state.params, last_obsv)

            rng, rng_sample_action = jax.random.split(rng)
            action = pi.sample(seed=rng_sample_action)
            log_prob = pi.log_prob(action).squeeze(-1)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            obs, reward, done, info = self.env.step(action)
            obsv = self.extractor(obs, last_obsv, done)

            transition = TransitionRL(
                last_done, action, value, reward, log_prob, last_obsv, info
            )
            runner_state = (train_state, obsv, done, None, rng)
            return runner_state, transition

        # CALCULATE ADVANTAGE
        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = (
                    reward + self.config["gamma"] * next_value * (1 - next_done) - value
                )
                gae = (
                    delta
                    + self.config["gamma"]
                    * self.config["gae_lambda"]
                    * (1 - next_done)
                    * gae
                )
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val, last_done),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # PPO LOSS
        def _loss_fn(params, rnn_state, traj_batch, gae, targets):
            # RERUN NETWORK
            pi, value = network.apply(traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action).squeeze(-1)

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -self.config["clip_eps"], self.config["clip_eps"]
            )
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # CALCULATE EXPLAINED VARIANCE
            residuals = traj_batch.reward - value
            var_residuals = jnp.var(residuals, axis=0)
            var_returns = jnp.var(traj_batch.reward, axis=0)

            explained_variance = jnp.where(
                var_returns > 0, 1.0 - var_residuals / var_returns, -jnp.inf
            ).mean()

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - self.config["clip_eps"],
                    1.0 + self.config["clip_eps"],
                )
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            reward_sum = traj_batch.reward.sum(axis=0)
            num_reward = (traj_batch.reward > 0).sum(axis=0)
            done_sum = traj_batch.done.sum(axis=0)
            mean_cum_reward = jnp.where(done_sum != 0, reward_sum / done_sum, 0).mean()
            mean_success_rate = jnp.where(
                done_sum != 0, num_reward / done_sum, 0
            ).mean()

            total_loss = (
                loss_actor
                + self.config["vf_coef"] * value_loss
                - self.config["ent_coef"] * entropy
            )
            return total_loss, (
                value_loss,
                loss_actor,
                entropy,
                mean_cum_reward,
                mean_success_rate,
                explained_variance,
            )

        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, aux), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (total_loss, aux)

        # UPDATE NETWORK
        def _update_single(update_state, unused):
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            # Batching and Shuffling
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # Mini-batch Updates
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES (CPU)
            traj_batch_list = []
            for _ in range(self.config["num_steps"]):
                runner_state, transition = _env_step(runner_state)
                traj_batch_list.append(transition)
            traj_batch = concatenate_transitions(traj_batch_list)

            # CALCULATE ADVANTAGE
            train_state, last_obsv, last_done, _, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obsv)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, (total_loss, aux) = jax.lax.scan(
                _update_single, update_state, None, config["update_epochs"]
            )

            (
                value_loss,
                loss_actor,
                entropy,
                cum_reward,
                success_rate,
                explained_variance,
            ) = aux
            metric = metric = {
                "loss": [total_loss],
                "value_loss": [value_loss],
                "actor_loss": [loss_actor],
                "cum_reward": [cum_reward],
                "success_rate": [success_rate],
                "entropy": [entropy],
                "explained_variance": [explained_variance],
            }

            train_state = update_state[0]
            rng = update_state[-1]

            runner_state = (train_state, last_obsv, last_done, None, rng)
            return runner_state, metric

        _, _rng = jax.random.split(self.key)
        runner_state = (
            train_state,
            obsv,
            jnp.zeros((self.config["num_envs"]), dtype=bool),
            None,
            _rng,
        )

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
            "config": self.config,
        }
