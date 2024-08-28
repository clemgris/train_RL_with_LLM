from typing import Dict, List

import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs_preprocessing import init_channels


class ConvNet(nn.Module):
    def __init__(
        self,
        conv_layers: List[int],
        final_hidden_layers: int,
        in_channels: int,
        height: int,
        width: int,
    ):
        super(ConvNet, self).__init__()
        self.final_hidden_layers = final_hidden_layers

        final_height = height
        final_width = width

        # Create convolutional layers dynamically
        layers = []
        for i, out_channels in enumerate(conv_layers):
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3)
            layers.append(conv_layer)
            in_channels = out_channels
            final_height -= 2
            final_width -= 2

        self.conv_layers = nn.ModuleList(layers)

        # Fully connected layers
        self.dense1 = nn.Linear(
            conv_layers[-1] * final_height * final_width, self.final_hidden_layers
        )
        self.dense2 = nn.Linear(self.final_hidden_layers, self.final_hidden_layers)

    def forward(self, x: torch.Tensor):

        x = torch.permute(x, (0, 3, 1, 2))
        for conv in self.conv_layers:
            x = F.relu(conv(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


class MLP(nn.Module):

    def __init__(self, final_hidden_layers: int, in_channels: int) -> None:
        super(MLP, self).__init__()

        self.final_hidden_layers = final_hidden_layers
        self.in_channels = in_channels

        self.dense1 = nn.Linear(in_channels, self.final_hidden_layers)
        self.dense2 = nn.Linear(self.final_hidden_layers, self.final_hidden_layers)

    def __call__(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


FEATURES_EXTRACTOR_DICT = {
    "im": ConvNet,
    "im_dir": ConvNet,
    "full_im_pos_dir": ConvNet,
    "mission": MLP,
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
                **self.kwargs[key]
            )
            self.feature_extractors[key] = feature_extractor
            self.layer_norms[key] = nn.LayerNorm(feature_extractor.final_hidden_layers)
            self.feature_extractors_out_channels += (
                feature_extractor.final_hidden_layers
            )

        self.final_fc1 = nn.Linear(
            self.feature_extractors_out_channels, self.final_hidden_layers
        )
        self.final_dense2 = nn.Linear(
            self.final_hidden_layers, self.final_hidden_layers
        )

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
        output = self.final_dense2(output)
        output = F.relu(output)
        return output
