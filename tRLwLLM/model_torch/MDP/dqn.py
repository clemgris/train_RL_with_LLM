import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import pickle

from tRLwLLM.environment import BabyAI
from tRLwLLM.utils import ReplayMemory, concatenate_dicts
from .policy import QNetwork
from ..obs_preprocessing import ExtractObs


class make_train_dqn:
    def __init__(self, config: Dict, eval_config: Dict) -> None:
        self.config = config
        self.eval_config = eval_config

        # INIT TRAINING PARAMETERS
        self.steps_done = 0
        self.num_updates = 0

        # DEVICE
        self.device = self.config["device"]

        # INIT ENV
        self.env = BabyAI(self.config)
        self.eval_env = BabyAI(self.eval_config)
        self.n_actions = self.env._env.action_space.n

    def train(self):

        extractor = ExtractObs(self.config)

        #  INIT NETWORK
        policy_net = QNetwork(n_actions=self.n_actions,
                              **self.config["feature_extractor_kwargs"]).to(self.device)
        target_net = QNetwork(n_actions=self.n_actions,
                              **self.config["feature_extractor_kwargs"]).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(
            policy_net.parameters(), lr=self.config["lr"], amsgrad=True
        )
        memory = ReplayMemory(size=self.config["buffer_size"])

        def select_action(state):
            sample = np.array([random.random() for _ in range(self.config["num_envs"])])
            eps_threshold = self.config["eps_end"] + (
                self.config["eps_start"] - self.config["eps_end"]
            ) * math.exp(-1.0 * self.steps_done / self.config["eps_decay"])
            self.steps_done += 1
            # Greedy actions
            with torch.no_grad():
                greedy_actions = policy_net(state).max(1).indices.unsqueeze(-1)
            # Random actions
            random_actions = torch.tensor(
                self.env.sample(), device=self.device, dtype=torch.long
            )
            # Eps-greedy actions
            chosen_actions = random_actions
            chosen_actions[sample > eps_threshold] = greedy_actions[
                sample > eps_threshold
            ]
            return chosen_actions

        def optimize_model():

            # ENOUGH EXPERIENCES IN THE BUFFER tO  CREATE AT LEAST ONE BATCH
            if len(memory) < self.config["batch_size"]:
                return

            # UPDATE NETWORK
            batch = memory.sample(self.config["batch_size"])

            not_done_batch = ~batch.done

            non_final_next_states = {k: v[not_done_batch] for k,v in batch.next_state.items()}

            state_action_values = policy_net(batch.state).gather(
                1, batch.action
            )

            next_state_values = torch.zeros(
                self.config["batch_size"], device=self.device
            )
            with torch.no_grad():
                next_state_values[not_done_batch] = (
                    target_net(non_final_next_states).max(1).values
                )

            expected_state_action_values = (
                next_state_values * self.config["gamma"]
            ) + batch.reward

            # DQN LOSS (Hubert loss)
            criterion = nn.SmoothL1Loss()
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

            # UPDATE QNETWORK
            optimizer.zero_grad()
            loss.backward()

            self.num_updates += 1

            # GRADIENT CLIPPING
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), self.config["grad_clip_value"])
            optimizer.step()

            return loss

        # TRAINING LOOP

        # RESET ENV
        obs, _ = self.env.reset()
        done = np.zeros(self.config["num_envs"])

        obsv = extractor(obs, None, done)
        obsv = {k: torch.from_numpy(v).to(dtype=torch.float32, device=self.device) for k,v in obsv.items()}

        training_metrics = []
        eval_metrics = []

        for step in range(1, self.config["timesteps"]+1):
            # EVAL POLICY
            if step % self.config["freq_eval"] == 0:
                eval_obs, _ = self.eval_env.reset()
                eval_done = np.zeros(self.eval_config["num_envs"])

                eval_obsv = extractor(eval_obs, None, eval_done)
                eval_obsv = {k: torch.from_numpy(v).to(dtype=torch.float32, device=self.device) for k,v in eval_obsv.items()}

                t_eval = 0

                eval_rewards = []
                eval_dones = []
                while t_eval < self.eval_env._env.max_steps + 1:
                    t_eval += 1
                    with torch.no_grad():
                        eval_action = policy_net(eval_obsv).max(1).indices
                    eval_next_obs, eval_reward, eval_done, _ = self.eval_env.step(
                        eval_action.to("cpu")
                    )

                    eval_next_obsv = extractor(eval_next_obs, eval_next_obs, eval_done)
                    eval_next_obsv = {k: torch.from_numpy(v).to(dtype=torch.float32, device=self.device) for k,v in eval_next_obsv.items()}

                    eval_obsv = eval_next_obsv

                    eval_rewards.append(eval_reward)
                    eval_dones.append(eval_done)

                all_eval_rewards = np.concatenate(eval_rewards).sum()
                all_eval_dones = np.concatenate(eval_dones).sum()
                num_positive_reward = (np.concatenate(eval_rewards) > 0).sum()

                cum_reward = all_eval_rewards / all_eval_dones
                success_rate = num_positive_reward / all_eval_dones
                eval_metric = {
                    "timestep": step,
                    "reward": cum_reward,
                    "success_rate": success_rate,
                    "num_envs": self.eval_config["num_envs"]
                }
                eval_metrics.append(eval_metric)

                eval_message = f"Step | {step}/{self.config["timesteps"]} | Eval | "
                for key, value in eval_metric.items():
                    eval_message += f" {key} | {value:.4f} | "
                print(eval_message)

            action = select_action(obsv)

            next_obs, reward, done, _ = self.env.step(action.to("cpu"))
            reward = torch.tensor(reward, device=self.device)
            done = torch.tensor(done, dtype=torch.bool, device=self.device)

            next_obsv = extractor(next_obs, obsv, done)
            next_obsv = {k: torch.from_numpy(v).to(dtype=torch.float32, device=self.device) for k,v in next_obsv.items()}

            # STORE TRANSITION IN REPLAY BUFFER
            memory.push(obsv, action, next_obsv, reward, done)

            # MOVE TO NEXT STATE
            obsv = next_obsv

            # UPDATE QNETWORK
            training_loss = optimize_model()
            # optimize_model()

            # UPDATE TARGET NETWORK
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.config[
                    "tau"
                ] + target_net_state_dict[key] * (1 - self.config["tau"])
            target_net.load_state_dict(target_net_state_dict)

            train_metric = {"timetseps": step, "updates": self.num_updates, "loss": training_loss.cpu().detach().numpy()}
            training_metrics.append(train_metric)

            if step % self.config["freq_display_training_metrics"] == 0:
                train_message = f"Step | {step}/{self.config["timesteps"]} | Train | "
                for key, value in train_metric.items():
                    train_message += f" {key} | {value:.4f} | "
                print(train_message)

            # SAVE
            if step % self.config["freq_save"] == 0:
                past_train_metric = os.path.join(
                    self.config["log_folder"],
                    f'training_metrics_{step - self.config["freq_save"]}.pkl',
                )
                past_eval_metric = os.path.join(
                    self.config["log_folder"],
                    f'eval_metrics_{step - self.config["freq_save"]}.pkl',
                )
                past_log_params = os.path.join(
                    self.config["log_folder"],
                    f'params_{step - self.config["freq_save"]}.pkl',
                )

                if os.path.exists(past_train_metric):
                    os.remove(past_train_metric)

                if os.path.exists(past_eval_metric):
                    os.remove(past_eval_metric)

                if os.path.exists(past_log_params):
                    os.remove(past_log_params)

                # Checkpoint
                with open(
                    os.path.join(
                        self.config["log_folder"], f"training_metrics_{step}.pkl"
                    ),
                    "wb",
                ) as pkl_file:
                    pickle.dump(concatenate_dicts(training_metrics), pkl_file)

                with open(
                    os.path.join(
                        self.config["log_folder"], f"eval_metrics_{step}.pkl"
                    ),
                    "wb",
                ) as pkl_file:
                    pickle.dump(concatenate_dicts(eval_metrics), pkl_file)

                # Save model weights
                with open(
                    os.path.join(self.config["log_folder"], f"params_{step}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(policy_net.parameters, f)

            with open(
                os.path.join(self.config["log_folder"], "args.json"), "w"
            ) as json_file:
                json.dump(self.config, json_file, indent=4)

        # Stop environment
        self.env._env.stop()

        return {
            "runner_state": policy_net.parameters,
            "training_metric": concatenate_dicts(training_metrics),
            "eval_metric": concatenate_dicts(eval_metrics),
            "config": self.config,
        }


