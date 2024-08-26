from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import torch
from torch2jax import t2j
from transformers import BertTokenizer, BertModel


def im_dir_extract(obs, last_obs, done):
    image = obs["image"]  # (B, H, W, C)
    _, H, W, _ = image.shape
    dir = obs["direction"]  # (B,)
    extanded_dir = jnp.tile(dir[..., None, None, None], (1, H, W, 1))

    im_dir = jnp.concatenate((image, extanded_dir), axis=-1)

    return im_dir


def full_im_pos_dir_extract(obs, last_obs, done):
    full_image = obs["full_image"]  # (B, H, W, C)
    _, H, W, _ = full_image.shape
    pos = obs["agent_pos"]  # (B, 2)
    extanded_pos = jnp.tile(pos[..., None, None, :], (1, H, W, 1))
    dir = obs["direction"]  # (B,)
    extanded_dir = jnp.tile(dir[..., None, None, None], (1, H, W, 1))

    full_im_pos_dir = jnp.concatenate((full_image, extanded_pos, extanded_dir), axis=-1)

    return full_im_pos_dir


def mission_extract(obs, last_obsv, done):
    if jnp.any(done):
        mission = list(obs["mission"])  # List of len B

        # Bert pre-trained semantic encoder
        with torch.no_grad():
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").cuda()

            tokenized = tokenizer(
                mission, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            outputs = model(**tokenized)
            # Get the embedding of the entire sentence
            mission_embedding = outputs.last_hidden_state[:, 0, ...]  # (B, 768)
            mission_embedding = t2j(mission_embedding)
    else:
        return last_obsv["mission"]
    return mission_embedding


EXTRACTOR_DICT = {
    "im_dir": im_dir_extract,
    "full_im_pos_dir": full_im_pos_dir_extract,
    "mission": mission_extract,
}


def init_dict(config, batch_size):
    return {
        "im_dir": jnp.zeros(
            (1, batch_size, config["rf_height"], config["rf_width"], 4)
        ),
        "full_im_pos_dir": jnp.zeros(
            (1, batch_size, config["height"], config["width"], 6)
        ),
        "mission": jnp.zeros((1, batch_size, 768)),
    }


@dataclass
class ExtractObs:
    config: Dict

    def __call__(self, obs, last_obsv, done) -> Dict[str, Any]:
        obs_features = {}

        for key in self.config["feature_extractor_kwargs"]["keys"]:
            if "[" in key:
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
            else:
                list_subkeys = [key]
            for subkey in list_subkeys:
                if subkey in self.config["feature_extractor_kwargs"]["kwargs"]:
                    kwargs = self.config["feature_extractor_kwargs"]["kwargs"][subkey]
                    obs_feature = EXTRACTOR_DICT[subkey](obs, last_obsv, done, **kwargs)
                else:
                    obs_feature = EXTRACTOR_DICT[subkey](obs, last_obsv, done)
                obs_features[subkey] = obs_feature
        return obs_features

    def init_x(self, batch_size=None):
        init_obs = {}

        if batch_size is None:
            batch_size = self.config["num_envs"]

        all_init_dict = init_dict(self.config, batch_size)
        for key in self.config["feature_extractor_kwargs"]["keys"]:
            if "[" in key:
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
            else:
                list_subkeys = [key]
            for subkey in list_subkeys:
                init_obs[subkey] = all_init_dict[subkey]
        return (init_obs, jnp.zeros((1, batch_size), dtype=bool))
