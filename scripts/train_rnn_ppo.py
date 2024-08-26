from datetime import datetime
import jax
import os

import sys

sys.path.append(".")

from tRLwLLM.model_jax import make_train_rnn_rl

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if __name__ == "__main__":
    config = {
        "DEBUG": True,
        "aneal_learning_rate": True,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "env_name": "BabyAI-GoToLocalS8N7-v0",
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"im_dir": 64, "mission": None},
            "keys": ["im_dir", "mission"],
            "kwargs": {},
        },
        "freq_save": 1,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "key": 42,
        "learning_rate": 2.5e-4,
        "max_grad_norm": 0.5,
        "num_envs": 32,
        "num_minibatchs": 4,  # Must divide num_envs
        "num_steps": 128,
        "total_time_steps": 5e8,
        "update_epochs": 1,
        "vf_coef": 0.5,
    }

    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_rnn_ppo/{date_string}"
    os.makedirs(log_folder, exist_ok="True")

    config["log_folder"] = log_folder

    training = make_train_rnn_rl(config)

    # with jax.disable_jit():  # DEBUG
    training_dict = training.train()
