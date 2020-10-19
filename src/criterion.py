import torch
import torch.nn as nn
import numpy as np

class EmbeddingsNormLoss(nn.Module):
    def forward(self, embeddings, *args):
        return torch.mean(torch.norm(embeddings, dim=1))



class WeightCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean'):
        super().__init__(torch.from_numpy(np.array(weight).astype(np.float32)), size_average, ignore_index, reduce, reduction)

