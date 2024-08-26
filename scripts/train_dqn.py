from datetime import datetime
import torch
import os

import sys

sys.path.append(".")

from tRLwLLM.model_torch import make_train_dqn

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    config = {
        "env_name": "MiniGrid-Empty-5x5-v0",
        "key": 42,
        "num_envs": 128,
        "batch_size": 128,
        "gamma": 0.99,
        "eps_start": 0.9,
        "eps_end": 0.05,
        "eps_decay": 1000,
        "tau": 0.005,
        "lr": 1e-4,
        "buffer_size": 10000,
        "device": "cuda",
        "timesteps": int(1e5),
        "grad_clip_value": 100,
        "freq_eval": 100,
        "freq_save": 100,
        "freq_display_training_metrics": 100,
        "feature_extractor_kwargs": {
            "final_hidden_layers": 64,
            "hidden_layers": {"full_im_pos_dir": 64},  # , "mission": None},
            "keys": ["full_im_pos_dir"],  # "mission"],
            "kwargs": {},
        },
    }

    eval_config = {
        "env_name": "MiniGrid-Empty-5x5-v0",
        "key": 123,
        "num_envs": 1,
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Available torch device: ", device)

    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_dqn/{date_string}"
    os.makedirs(log_folder, exist_ok="True")

    config["log_folder"] = log_folder

    training = make_train_dqn(config, eval_config)

    training_dict = training.train()
