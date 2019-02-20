import torch
from torch import nn

class EncNet(nn.Module):
    def __init__(self, args, aux=True, se_loss=True, lateral=False, **kwargs):
        super().__init__()
        