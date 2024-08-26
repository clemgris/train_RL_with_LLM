import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import time

from tRLwLLM.environment import BabyAI
from tRLwLLM.utils import (
    TransitionBC,
    TransitionPPO,
    concatenate_dicts,
    concatenate_transitions,
)

from ..feature_extractor import KeyExtractor
from ..obs_preprocessing import ExtractObs
from .rnn_policy import ActorCriticRNN, ScannedRNN


def extand(x):
    if isinstance(x, jnp.ndarray):
        return x[jnp.newaxis, ...]
    else:
        return


class make_train_rnn_bc:
    def __init__(self, config, eval_config):
        self.config = config
        self.eval_config = eval_config

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
        self.eval_env = BabyAI(self.eval_config)

        self.config["rf_height"] = self.env._env.agent_view_size
        self.config["rf_width"] = self.env._env.agent_view_size
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

        feature_extractor_shape = self.config["feature_extractor_kwargs"][
            "final_hidden_layers"
        ]
        init_x = self.extractor.init_x(self.config["num_envs"])
        init_rnn_state_train = ScannedRNN.initialize_carry(
            (self.config["num_envs"], feature_extractor_shape)
        )
        init_rnn_state_eval = ScannedRNN.initialize_carry(
            (self.eval_config["num_envs"], feature_extractor_shape)
        )

        network_params = network.init(self.key, init_rnn_state_train, init_x)

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

        # COLLECT TRAJECTORIES FROM DEMONSTRATION (EXPERT POLICY)
        def _env_step(runner_state):
            train_state, last_obsv, last_done, rng = runner_state

            # GET EXPERT ACTION
            expert_action = self.env.get_expert_action()

            # UPDATE ENV
            obs, _, done, info = self.env.step(expert_action)
            obsv = self.extractor(obs, last_obsv, done)

            transition = TransitionBC(last_done, expert_action, last_obsv, info)
            runner_state = (
                train_state,
                obsv,
                done,
                rng,
            )
            return runner_state, transition

        # COLLECT TRAJECTORIES FROM IMITATOR
        def _env_step_eval(runner_state):
            train_state, last_obsv, last_done, rnn_state_eval, rng = runner_state

            # SELECT ACTION
            ac_in = (jax.tree_map(extand, last_obsv), last_done[np.newaxis, :])
            rnn_state_eval, pi, value = network.apply(
                train_state.params, rnn_state_eval, ac_in
            )

            rng, rng_sample_action = jax.random.split(rng)
            action = pi.sample(seed=rng_sample_action)
            log_prob = pi.log_prob(action).squeeze(-1)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            obs, reward, done, info = self.eval_env.step(action)
            obsv = self.extractor(obs, last_obsv, done)

            transition = TransitionPPO(
                last_done, action, value, reward, log_prob, last_obsv, info
            )
            runner_state = (
                train_state,
                obsv,
                done,
                rnn_state_eval,
                rng,
            )
            return runner_state, transition

        # BC LOSS
        def _loss_fn(params, init_rnn_state, traj_batch):
            _, action_dist, value = network.apply(
                params, init_rnn_state[0], (traj_batch.obs, traj_batch.done)
            )
            log_prob = action_dist.log_prob(traj_batch.expert_action)

            total_loss = -log_prob.mean()

            return total_loss

        # UPDATE NETWORK ON MINIBATCH
        def _update_minbatch(train_state, batch_info):
            init_rnn_state, traj_batch = batch_info

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            total_loss, grads = grad_fn(train_state.params, init_rnn_state, traj_batch)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        # UPDATE NETWORK ON ONE EPOCH
        def _update_single(update_state, unused):
            (
                train_state,
                init_rnn_state,
                traj_batch,
                rng,
            ) = update_state

            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, self.config["num_envs"])
            batch = (init_rnn_state, traj_batch)

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], self.config["num_minibatchs"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )

            update_state = (
                train_state,
                init_rnn_state,
                traj_batch,
                rng,
            )
            return update_state, total_loss

        # TRAIN LOOP
        def _update_step(cary, unused):
            train_state, rng = cary

            # RESET ENV
            obs, _ = self.env.reset()
            obsv = self.extractor(
                obs, None, jnp.ones((self.config["num_envs"]), dtype=bool)
            )

            rng, rng_train_rs = jax.random.split(rng)
            runner_state = (
                train_state,
                obsv,
                jnp.zeros((self.config["num_envs"]), dtype=bool),
                rng_train_rs,
            )

            # COLLECT TRAJECTORIES FROM EXPERT (CPU)
            traj_batch_list = []
            for _ in range(self.config["num_steps"]):
                runner_state, transition = _env_step(runner_state)
                traj_batch_list.append(transition)
            traj_batch = concatenate_transitions(traj_batch_list)

            update_state = (
                train_state,
                init_rnn_state_train[None, :],
                traj_batch,
                rng,
            )

            # UPDATE NETWORK ON COLLECTED TRAJECTORIES (GPU)
            update_state, total_loss = jax.lax.scan(
                _update_single, update_state, None, self.config["update_epochs"]
            )

            train_state = update_state[0]
            cary = train_state, rng

            # EVALUATE

            # RESET EVALUATION ENV
            eval_obs, _ = self.eval_env.reset()
            eval_obsv = self.extractor(
                eval_obs, None, jnp.ones((self.eval_config["num_envs"]), dtype=bool)
            )
            rng, rng_eval_rs = jax.random.split(rng)
            eval_runner_state = (
                train_state,
                eval_obsv,
                jnp.zeros((self.eval_config["num_envs"]), dtype=bool),
                init_rnn_state_eval,
                rng_eval_rs,
            )

            # COLLECT TRAJECTORIES FROM IMITATOR (CPU)
            traj_batch_eval_list = []
            for _ in range(self.eval_config["num_steps"]):
                eval_runner_state, transition = _env_step_eval(eval_runner_state)
                traj_batch_eval_list.append(transition)
            traj_batch_eval = concatenate_transitions(traj_batch_eval_list)

            # COMPUTE METRICS
            reward_sum = traj_batch_eval.reward.sum(axis=0)
            num_reward = (traj_batch_eval.reward > 0).sum(axis=0)
            done_sum = traj_batch_eval.done.sum(axis=0)
            cum_reward = jnp.where(done_sum != 0, reward_sum / done_sum, 0).mean()
            success_rate = jnp.where(done_sum != 0, num_reward / done_sum, 0).mean()

            metric = metric = {
                "loss": [total_loss],
                "cum_reward": [cum_reward],
                "success_rate": [success_rate],
            }

            return cary, metric

        _, rng = jax.random.split(self.key)
        cary = (train_state, rng)

        metrics = []
        for update in range(self.config["num_updates"]):
            time_start = time.time()
            cary, train_metric = _update_step(cary, None)

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
                    pickle.dump(train_state.params, f)

            with open(
                os.path.join(self.config["log_folder"], "args.json"), "w"
            ) as json_file:
                json.dump(self.config, json_file, indent=4)

        # Stop environment
        self.env._env.stop()

        return {
            "train_state": train_state,
            "metric": concatenate_dicts(metrics),
            "config": self.config,
        }
