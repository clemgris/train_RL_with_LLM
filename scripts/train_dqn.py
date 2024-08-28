from datetime import datetime
import torch
import os

import sys

sys.path.append(".")

from tRLwLLM.model_torch import make_train_dqn

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    config = {
        "batch_size": 128,
        "buffer_size": 100000,
        "device": "cuda",
        "env_name": "BabyAI-GoToLocalS8N7-v0",
        "eps_decay": int(1e9),  # = timesteps
        "eps_end": 0.05,
        "eps_start": 0.9,
        "feature_extractor_kwargs": {
            "final_hidden_layers": 512,
            "hidden_layers": {"full_im_pos_dir": 512, "mission": 512},
            "keys": ["full_im_pos_dir", "mission"],
            "kwargs": {
                "full_im_pos_dir": {
                    "conv_layers": [32, 64, 128],
                    "height": 8,
                    "width": 8,
                },
                "mission": {},
            },
        },
        "freq_display_training_metrics": 100,
        "freq_eval": 100,
        "freq_save": 100,
        "gamma": 0.99,
        "grad_clip_value": 100,
        "key": 42,
        "lr": 1e-4,
        "num_envs": 128,
        "tau": 0.005,
        "timesteps": int(1e9),
    }

    eval_config = {
        "env_name": config["env_name"],
        "key": 123,
        "num_envs": 32,
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
