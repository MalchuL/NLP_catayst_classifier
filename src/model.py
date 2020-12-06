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


        lstm_channels = hidden_size * num_layers * (2 if bidirectional else 1)



        self.hidden = nn.Sequential(nn.Linear(lstm_channels * (2 + 3), hidden_size), nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(inplace=True))

        self.classify = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

        self.act = nn.Softmax(dim=1)

    @staticmethod
    def pack_items(items):
        lengths = [item.shape[0] for item in items]

        padded_items = pad_sequence(items)
        items = pack_padded_sequence(padded_items, lengths, enforce_sorted=False)
        return items

    def process_name_i(self, name_i):
        _, (title_output, title_cell) = self.body_title_net(self.pack_items(name_i))
        return title_output


    def merge_outputs(self, *outputs_list):
        stacked_list = torch.stack(outputs_list, dim=0)
        mean = stacked_list.mean(0)
        max = torch.max(stacked_list, dim=0)[0]

        if len(outputs_list) == 2:
            diff_abs = (outputs_list[0] - outputs_list[1]).abs()
        else:
            raise

        output = torch.cat([max,mean,diff_abs,*outputs_list], dim=-1)
        return output


    def forward(self, name_1, name_2):

       # B,T,C = name_1.shape
        name_1_output = self.process_name_i(name_1)
        name_2_output = self.process_name_i(name_2)
        #assert name_1_output.shape[1] == B

        merged = self.merge_outputs(name_1_output,name_2_output)

        #(num_layers * num_directions, batch, hidden_size):
        merged = merged.permute(1,0,2).reshape(merged.shape[1],-1)


        hidden = self.hidden(merged)

        mapped = self.classify(hidden)

        classes = self.act(mapped)
        classes = torch.clamp(classes, self.clip, 1 - self.clip)
        return classes

