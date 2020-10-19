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
        hidden_size=128, # Best on MLP
        num_layers=1,
        bidirectional=False,
        clip=0.0
    ):
        super().__init__()

        self.clip = clip


        self.body_title_net = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=False, bidirectional=bidirectional)  # Two different networks don't works
        self.tags_net = nn.Linear(input_channels, hidden_size)


        self.title_lstm_output_to_hidden = nn.Linear(hidden_size * num_layers * (2 if bidirectional else 1), hidden_size)


        self.tags_to_title = nn.Linear(hidden_size, input_channels)
        self.title_to_body = nn.Linear(hidden_size, input_channels)

        self.hidden = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.GroupNorm(NUM_GROUPS, hidden_size),
                                    nn.ReLU(inplace=True))

        self.classify = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

        self.act = nn.Softmax(dim=1)


    def pack_items(self, items, add_elem=None):
        if add_elem is None:
            lengths = [item.shape[0] for item in items]
        else:
            lengths = [item.shape[0] + 1 for item in items]
        padded_items = pad_sequence(items)
        if add_elem is not None:
            B,C = add_elem.shape
            padded_items = torch.cat([add_elem.view(1,B,-1), padded_items], dim=0)
        items = pack_padded_sequence(padded_items, lengths, enforce_sorted=False)
        return items

    def forward(self, body, title, tags):
        #print(type(body), type(title), type(tags))
        B, C = tags.shape
        tags_output = self.tags_net(tags)

        tags_to_title = self.tags_to_title(tags_output)
        _, (title_output, title_cell) = self.body_title_net(self.pack_items(title, tags_to_title))
        assert title_cell.shape[1] == B

        title_cell = self.title_lstm_output_to_hidden(title_cell.transpose(1,0).reshape(B, -1))
        title_to_body =  self.title_to_body(title_cell)
        _, (body_output,_) = self.body_title_net(self.pack_items(body, title_to_body))
        assert body_output.shape[1] == B

        #print(body_output.shape, title_output.shape, tags_output.shape)
        concated = torch.cat([body_output[-1], title_output[-1], tags_output], dim=1)

        hidden = self.hidden(concated)
        mapped = self.classify(hidden)

        classes = self.act(mapped)
        classes = torch.clamp(classes, self.clip, 1 - self.clip)
        return classes

