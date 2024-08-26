import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from ..feature_extractor import KeyExtractor


class QNetwork(nn.Module):
    def __init__(
        self,
        n_actions: int,
        final_hidden_layers: int,
        keys: List,
        kwargs: Dict,
        hidden_layers: Dict,
    ):
        super(QNetwork, self).__init__()
        self.feature_extractor = KeyExtractor(
            final_hidden_layers, keys, kwargs, hidden_layers
        )

        self.dense = nn.Linear(
            self.feature_extractor.final_hidden_layers, final_hidden_layers
        )
        self.final_layer = nn.Linear(final_hidden_layers, n_actions)

    def forward(self, x):
        # Extract state features
        x = F.relu(self.feature_extractor(x))

        x = F.relu(self.dense(x))
        x = self.final_layer(x)
        return x
