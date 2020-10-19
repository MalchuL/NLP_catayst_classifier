from typing import Dict  # isort:skip
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils
from catalyst.contrib.models import SequentialNet
from catalyst.contrib.models.cv import ResnetEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class NLPClassifierModel(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels,
        hidden_size=128, # Best on MLP
        num_layers=1
    ):
        super().__init__()
        self.body_net = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.title_net = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.tags_net = nn.Linear(input_channels, hidden_size)
        self.map_classes = nn.Linear(hidden_size * 3,  num_classes)
        self.act = nn.Softmax(dim=1)


    def pack_items(self, items):
        lengths = [item.shape[0] for item in items]
        #print(lengths)
        padded_items = pad_sequence(items)
        #print(padded_items.size(), type(padded_items))
        items = pack_padded_sequence(padded_items, lengths, enforce_sorted=False)
        return items

    def forward(self, body, title, tags):
        #print(type(body), type(title), type(tags))
        _, (body_output,_) = self.body_net(self.pack_items(body))
        _, (title_output, _) = self.body_net(self.pack_items(title))
        tags_output = self.tags_net(tags)
        #print(body_output.shape, title_output.shape, tags_output.shape)
        concated = torch.cat([body_output[0], title_output[0], tags_output], dim=1)

        mapped = self.map_classes(concated)

        classes = self.act(mapped)
        return classes

