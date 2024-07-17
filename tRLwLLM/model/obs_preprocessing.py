from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import torch
from transformers import BertTokenizer, BertModel


def im_dir_extract(obs):
    image = obs["image"]  # (B, H, W, C)
    _, H, W, _ = image.shape
    dir = obs["direction"]  # (B,)
    extanded_dir = jnp.tile(dir[..., None, None, None], (1, H, W, 1))

    im_dir = jnp.concatenate((image, extanded_dir), axis=-1)

    return im_dir


class SetenceEncoder:
    def __call__(self, x):
        with torch.no_grad():
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").cuda()

            tokenized = tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            outputs = model(**tokenized)
            mission_embedding = outputs.last_hidden_state[
                :, 0, ...
            ]  # Get the embedding of the entire sentence
        return mission_embedding


def mission_extract(obs):
    mission = list(obs["image"])  # List of len B
    embedded_mission = SetenceEncoder(mission)  # (B, 768)
    return embedded_mission


EXTRACTOR_DICT = {"im_dir": im_dir_extract, "mission": mission_extract}


def init_dict(config, batch_size):
    return {
        "im_dir": jnp.zeros((1, batch_size, config["height"], config["width"], 4)),
        "mission": jnp.zeros((1, batch_size, 768)),
    }


@dataclass
class ExtractObs:
    config: Dict

    def __call__(self, obs) -> Dict[str, Any]:
        obs_features = {}

        for key in self.config["feature_extractor_kwargs"]["keys"]:
            if "[" in key:
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
            else:
                list_subkeys = [key]
            for subkey in list_subkeys:
                if subkey in self.config["feature_extractor_kwargs"]["kwargs"]:
                    kwargs = self.config["feature_extractor_kwargs"]["kwargs"][subkey]
                    obs_feature = EXTRACTOR_DICT[subkey](obs, **kwargs)
                else:
                    obs_feature = EXTRACTOR_DICT[subkey](obs)
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
