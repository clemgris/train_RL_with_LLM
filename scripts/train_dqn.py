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
        "buffer_batch_size": 128,
        "buffer_size": 10000,
        "env_name": "BabyAI-GoToLocalS8N7-v0",
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"full_im_pos_dir": 64, "mission": None},
            "keys": ["full_im_pos_dir", "mission"],
            "kwargs": {},
        },
        "entity": "",
        "epsilon_anneal_time": 25e4,
        "epsilon_finish": 0.05,
        "epsilon_start": 1.0,
        "freq_save": 1,
        "gamma": 0.99,
        "key": 42,
        "learning_starts": 10000,
        "lr": 2.5e-4,
        "lr_linear_decay": False,
        "num_envs": 10,
        "num_seeds": 1,
        "seed": 0,
        "target_update_interval": 500,
        "tau": 1.0,
        "total_timesteps": 5e5,
        "training_interval": 10,
    }

    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_dqn/{date_string}"
    os.makedirs(log_folder, exist_ok="True")

    config["log_folder"] = log_folder

    training = make_train_dqn(config)

    # with jax.disable_jit():  # DEBUG
    training_dict = training.train()
