import jax
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from rnn_ppo import make_train

if __name__ == "__main__":
    config = {
        "DEBUG": True,
        "aneal_learning_rate": True,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "env_name": 'MiniGrid-Empty-5x5-v0',
        "extractor": 'ExtractObs',
        "feature_extractor": 'KeyExtractor',
        "feature_extractor_kwargs": {
            'final_hidden_layers': 64,
            'hidden_layers': {'im_dir': 64},
            'keys': ['im_dir'],
            'kwargs': {}
        },
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "key": 42,
        "learning_rate": 2.5e-4,
        "max_grad_norm": 0.5,
        "num_envs": 4,
        "num_minibatchs": 1, # Must divide num_envs
        "num_steps": 16,
        "total_time_steps": 5e6,
        "update_epochs": 1,
        "vf_coef": 0.5
    }

    training = make_train(config)

    # with jax.disable_jit(): # DEBUG
    training_dict = training.train()
