from datetime import datetime
import jax
import os

import sys

sys.path.append(".")

from tRLwLLM.model import make_train_dqn

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if __name__ == "__main__":

    config = {
        "adam_eps": 1e-5,
        "buffer_batch_size": 1024,
        "buffer_size": 204800,  # Must be a multipliar of num_envs
        "env_name": "MiniGrid-Empty-5x5-v0",  # "BabyAI-GoToLocalS8N7-v0",
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"full_im_pos_dir": 64},  # , "mission": None},
            "keys": ["full_im_pos_dir"],  # , "mission"],
            "kwargs": {},
        },
        "entity": "",
        "epsilon_anneal_time": 25e5,
        "epsilon_finish": 0.01,
        "epsilon_start": 1.0,
        "freq_eval": 10,
        "freq_save": 10,
        "gamma": 0.99,
        "key": 42,
        "learning_starts": 204800,  # At least the buffer size
        "lr": 2.5e-4,
        "lr_linear_decay": False,
        "num_envs": 4096,  # 1024,
        "num_eval_steps": 128,
        "num_seeds": 1,
        "seed": 0,
        "target_update_interval": 500,
        "tau": 1.0,
        "total_timesteps": 5e6,
        "training_interval": 1024,  # Must be a multiple of num_envs
    }

    eval_config = {
        "env_name": config["env_name"],
        "feature_extractor_kwargs": config["feature_extractor_kwargs"],
        "key": 123,
        "num_envs": 1024,
    }

    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_dqn/{date_string}"
    os.makedirs(log_folder, exist_ok="True")

    config["log_folder"] = log_folder

    training = make_train_dqn(config, eval_config)

    # with jax.disable_jit():  # DEBUG
    training_dict = training.train()
