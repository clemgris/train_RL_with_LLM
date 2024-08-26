from typing import Dict, List

import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs_preprocessing import init_channels


class ConvNet(nn.Module):
    def __init__(self, final_hidden_layers: int, in_channels: int):
        self.final_hidden_layers = final_hidden_layers
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        # self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.final_hidden_layers)

    def forward(self, x: torch.Tensor):
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class Identity:
    final_hidden_layers: int = None

    def __call__(self, x):
        return x


FEATURES_EXTRACTOR_DICT = {
    "im": ConvNet,
    "im_dir": ConvNet,
    "full_im_pos_dir": ConvNet,
    "mission": Identity,
}


class KeyExtractor(nn.Module):
    def __init__(
        self,
        final_hidden_layers: int,
        keys: List,
        kwargs: Dict = None,
        hidden_layers: Dict = None,
    ):
        self.final_hidden_layers = final_hidden_layers
        super(KeyExtractor, self).__init__()

        self.final_hidden_layers = final_hidden_layers
        self.keys = keys
        self.kwargs = kwargs
        self.hidden_layers = hidden_layers

        self.in_channels = init_channels()

        self.feature_extractors = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        self.feature_extractors_out_channels = 0
        for key in self.keys:
            feature_extractor = FEATURES_EXTRACTOR_DICT[key](
                final_hidden_layers=self.hidden_layers.get(key, None),
                in_channels=self.in_channels[key],
            )
            self.feature_extractors[key] = feature_extractor
            self.layer_norms[key] = nn.LayerNorm(feature_extractor.final_hidden_layers)
            self.feature_extractors_out_channels += (
                feature_extractor.final_hidden_layers
            )

        self.final_fc1 = nn.Linear(
            self.feature_extractors_out_channels, self.final_hidden_layers
        )
        self.final_fc2 = nn.Linear(self.final_hidden_layers, self.final_hidden_layers)

    def forward(self, obs):
        outputs = []
        for key in self.keys:
            if "[" in key:
                inputs = []
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
                for subkey in list_subkeys:
                    input = obs[subkey]
                    input = torch.flatten(input, start_dim=1)
                    inputs.append(input)
                concat_inputs = torch.cat(inputs, axis=-1)
                x = self.feature_extractors[key](concat_inputs)
            else:
                x = self.feature_extractors[key](obs[key])  # DEBUG
            x = self.layer_norms[key](x)  # Layer norm
            outputs.append(x)

        flattened = torch.cat(outputs, axis=-1)

        output = self.final_fc1(flattened)
        output = F.relu(output)
        output = self.final_fc2(output)
        output = F.relu(output)
        return output
