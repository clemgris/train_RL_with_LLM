import gymnasium as gym
from tRLwLLM.utils import concatenate_dicts

from .base_env import BaseEnv
from .vec_env import ParallelEnv
from .babyai_bot import BabyAIBot


class BabyAI(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        envs = []
        experts = []
        self.config = config

        for i in range(self.config["num_envs"]):
            env = gym.make(self.config["env_name"], render_mode="rgb_array")
            env.reset(seed=100 * self.config["key"] + i)
            envs.append(env)
            experts.append(BabyAIBot(env))

        self._env = ParallelEnv(envs, experts)

    def __prepare_infos(self, infos):
        return concatenate_dicts(infos)

    def __generate_obs(self, obs):
        return concatenate_dicts(obs)

    def reset(self):
        obs, infos = self._env.reset()
        return self.__generate_obs(obs), self.__prepare_infos(infos)

    def step(self, actions):
        obs, rews, dones, infos = self._env.step(actions)
        return self.__generate_obs(obs), rews, dones, self.__prepare_infos(infos)

    def get_expert_action(self):
        return self._env.get_expert_action()
