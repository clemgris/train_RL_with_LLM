from datetime import datetime
import jax
import os

import sys

sys.path.append(".")

from tRLwLLM.model import make_train_bc

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if __name__ == "__main__":
    config = {
        "DEBUG": True,
        "aneal_learning_rate": True,
        "env_name": "BabyAI-GoToLocalS8N7-v0",
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"im_dir": 64, "mission": None},
            "keys": ["im_dir", "mission"],
            "kwargs": {},
        },
        "freq_save": 1,
        "key": 42,
        "learning_rate": 2.5e-4,
        "max_grad_norm": 0.5,
        "num_envs": 32,
        "num_minibatchs": 4,  # Must divide num_envs
        "num_steps": 128,
        "total_time_steps": 5e8,
        "update_epochs": 1,
    }

    eval_config = {
        "DEBUG": True,
        "aneal_learning_rate": True,
        "env_name": "BabyAI-GoToLocalS8N7-v0",
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"im_dir": 64, "mission": None},
            "keys": ["im_dir", "mission"],
            "kwargs": {},
        },
        "freq_save": 1,
        "key": 123,
        "learning_rate": 2.5e-4,
        "num_envs": 8,
        "num_steps": 128,
    }

    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_rl/{date_string}"
    os.makedirs(log_folder, exist_ok="True")

    config["log_folder"] = log_folder

    training = make_train_bc(config, eval_config)

    # with jax.disable_jit():  # DEBUG
    training_dict = training.train()
