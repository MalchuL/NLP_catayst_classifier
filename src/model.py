from typing import Dict  # isort:skip
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils
from catalyst.contrib.models import SequentialNet
from catalyst.contrib.models.cv import ResnetEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

NUM_GROUPS = 8

class NLPClassifierModel(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels,
        feat_size=256,
        hidden_size=128, # Best on MLP
        num_layers=2,
        clip=1e-4
    ):
        super().__init__()

        self.clip = clip

        self.body_inner_map = nn.Sequential(nn.Conv1d(input_channels, feat_size, 1),
                                            nn.GroupNorm(NUM_GROUPS, feat_size), nn.ReLU(inplace=True))
        self.body_net = nn.LSTM(input_size=feat_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=False, bidirectional=True)

        self.title_inner_map = nn.Sequential(nn.Conv1d(input_channels, feat_size, 1),
                                             nn.GroupNorm(NUM_GROUPS, feat_size), nn.ReLU(inplace=True))
        self.title_net = nn.LSTM(input_size=feat_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.tags_net = nn.Linear(input_channels, hidden_size)
        self.hidden = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.GroupNorm(NUM_GROUPS, hidden_size),
                                    nn.ReLU(inplace=True))

        self.classify = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

        self.act = nn.Softmax(dim=1)


    def pack_items(self, items, map_nn=None):
        lengths = [item.shape[0] for item in items]
        padded_items = pad_sequence(items)
        if map_nn is not None:
            padded_items = map_nn(padded_items.permute(1,2,0)).permute(2,0,1)
        items = pack_padded_sequence(padded_items, lengths, enforce_sorted=False)
        return items

    def forward(self, body, title, tags):
        #print(type(body), type(title), type(tags))
        tags_output = self.tags_net(tags)
        _, (title_output, _) = self.title_net(self.pack_items(title, self.title_inner_map))
        _, (body_output,_) = self.body_net(self.pack_items(body, self.body_inner_map))


        #print(body_output.shape, title_output.shape, tags_output.shape)
        concated = torch.cat([body_output[0], title_output[0], tags_output], dim=1)

        hidden = self.hidden(concated)
        mapped = self.classify(hidden)

        classes = self.act(mapped)
        classes = torch.clamp(classes, self.clip, 1 - self.clip)
        return classes

