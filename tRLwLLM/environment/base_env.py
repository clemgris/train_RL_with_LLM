class BaseEnv:
    def __init__(self, config):
        self.n_parallel = config["num_envs"]

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()