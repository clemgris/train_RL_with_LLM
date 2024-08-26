import logging
import math
from copy import deepcopy
from multiprocessing import Pipe, Process
from typing import List

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from .babyai_bot import BabyAIBot


def multi_worker(conn, envs, experts):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask)
        if cmd == "step":
            ret = []
            for env, expert, a, stopped in zip(envs, experts, data[0], data[1]):
                if not stopped:
                    obs, reward, done, truncated, info = env.step(a)

                    done = done or truncated
                    if done:
                        obs, info = env.reset()
                        expert.reset(env)

                    # Add agent position in the grid
                    obs["agent_pos"] = env.unwrapped.agent_pos
                    # Add full observation of the grid
                    grid = env.unwrapped.grid
                    vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
                    full_im = grid.encode(vis_mask)
                    obs["full_image"] = full_im
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        elif cmd == "expert_action":
            ret = []
            for expert in experts:
                expert_action = expert.replan(
                    None
                )  # WARING: Assume previous action taken by expert
                ret.append(expert_action)
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env, expert in zip(envs, experts):
                obs, info = env.reset()
                # Add agent position in the grid
                obs["agent_pos"] = env.unwrapped.agent_pos
                # Add full observation of the grid
                grid = env.unwrapped.grid
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
                full_im = grid.encode(vis_mask)
                obs["full_image"] = full_im
                expert.reset(env)
                ret.append((obs, info))
            conn.send(ret)
        # sample uniformally action
        elif cmd == "sample":
            ret = []
            for env in envs:
                ret.append(env.action_space.sample())
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            ret = envs[0].render()
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """Parallel environment that holds a list of environments and can
    evaluate a low-level policy.
    """

    def __init__(
        self,
        envs: List[gym.Env],  # List of environments
        experts: List[BabyAIBot],
        num_cores: int = 8,
    ):
        assert len(envs) >= 1, "No environment provided"
        self.envs = envs
        self.experts = experts
        self.num_envs = len(self.envs)
        assert self.num_envs == len(self.experts)
        self.spec = deepcopy(self.envs[0].unwrapped.spec)
        self.spec_id = f"ParallelShapedEnv<{self.spec.id}>"
        self.env_name = self.envs[0].unwrapped.spec.id
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.num_cores = num_cores
        self.max_steps = self.envs[0].unwrapped.max_steps
        self.height = self.envs[0].unwrapped.grid.height
        self.width = self.envs[0].unwrapped.grid.width
        self.agent_view_size = self.envs[0].unwrapped.agent_view_size

        self.envs_per_proc = math.ceil(self.num_envs / self.num_cores)

        # Setup arrays to hold current observation and timestep
        # for each environment
        self.obss = []
        self.ts = np.array([0 for _ in range(self.num_envs)])

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.start_processes()

    def __len__(self):
        return self.num_envs

    def __str__(self):
        self.locals[0].send(("__str__", None))
        return f"<ParallelShapedEnv<{self.locals[0].recv()}>>"

    def stop(self):
        for p in self.processes:
            p.terminate()

    def gen_obs(self):
        return self.obss

    def render(self, mode="rgb_array", highlight=False):
        """Render a single environment"""
        self.locals[0].send(("render_one", (mode, highlight)))
        return self.locals[0].recv()

    def start_processes(self):
        """Spin up the num_envs/envs_per_proc number of processes"""
        logger.info(f"spinning up {self.num_envs} processes")
        for i in range(0, self.num_envs, self.envs_per_proc):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(
                target=multi_worker,
                args=(
                    remote,
                    self.envs[i : i + self.envs_per_proc],
                    self.experts[i : i + self.envs_per_proc],
                ),
            )
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)
        logger.info("done spinning up processes")

    def request_reset_envs(self):
        """Request all processes to reset their envs"""
        logger.info("requesting resets")
        for local in self.locals:
            local.send(("reset", None))
        self.obss = []
        logger.info("requested resets")

        infos = []
        for local in self.locals:
            res = local.recv()

            for j in range(len(res)):
                infos.append(res[j][1])
                if res[j][0] is not None:
                    self.obss += [res[j][0]]
            # self.obss += local.recv()
        logger.info("completed resets")
        return infos

    def reset(self):
        """Reset all environments"""
        infos = self.request_reset_envs()
        return [obs for obs in self.obss], infos

    def request_step(self, actions, stop_mask):
        """Request processes to step corresponding to (primitive) actions
        unless stop mask indicates otherwise"""
        for i in range(0, self.num_envs, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                (
                    "step",
                    [
                        actions[i : i + self.envs_per_proc],
                        stop_mask[i : i + self.envs_per_proc],
                    ],
                )
            )
        results = []
        for i in range(0, self.num_envs, self.envs_per_proc):
            res = self.locals[i // self.envs_per_proc].recv()
            for j in range(len(res)):
                results.append(res[j])
                if results[-1][0] is not None:
                    self.obss[i + j] = results[-1][0]
        return zip(*results)

    def step(self, actions):
        """Complete a step and evaluate low-level policy / termination
        classifier as needed depending on reward shaping scheme.

        Returns:  obs: list of environment observations,
                  reward: np.array of extrinsic rewards,
                  done: np.array of booleans,
                  info: depends on self.reward_shaping. Output can be used
                        to shape the reward.
        """
        # Make sure input is numpy array:
        actions = np.array(actions)
        actions_to_take = actions.copy()

        # Make a step in the environment
        stop_mask = np.array([False for _ in range(self.num_envs)])
        obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
        reward = np.array(reward)
        done_mask = np.array(done)

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info

    def sample(self):
        for local in self.locals:
            local.send(("sample", None))
        sample_actions = []
        for local in self.locals:
            res = local.recv()
            sample_actions.append(res)
        return np.concatenate(sample_actions)[..., None]

    def request_expert_actions(self):
        """Request processes to return expert action"""
        logger.info("requesting resets")
        for local in self.locals:
            local.send(("expert_action", None))
        expert_actions = []
        for local in self.locals:
            res = local.recv()
            expert_actions.append(res)
        return expert_actions

    def get_expert_action(self):
        actions = np.concatenate(self.request_expert_actions())[..., None]
        return actions
