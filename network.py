import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FullyConnected(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_dim:int=1024, num_layers:int=6, last_relu:bool=False, BN:bool=False):
        super(FullyConnected, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.last_relu = last_relu
        self.BN = BN

        assert self.num_layers >= 1, "Number of layers must be at least 1."
        layers = []
        cur_dim = in_dim
        for i in range(self.num_layers):
            next_dim = self.hidden_dim if i < self.num_layers - 1 else self.out_dim
            layers.append(nn.Linear(cur_dim, next_dim))
            if i < self.num_layers - 1:
                if BN:
                    layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.ReLU())
            cur_dim = next_dim
        if last_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)


class Cascade_Net(nn.Module):
    def __init__(self, hidden_dim:int=512):
        super(Cascade_Net, self).__init__()
        self.shared_encoder = FullyConnected(4, 6, hidden_dim, 7, last_relu=False, BN=True)
        self.physical_params = FullyConnected(6, 6, hidden_dim, 3, last_relu=False, BN=True)

    def forward(self, x, **kwargs):
        x = self.shared_encoder(x)
        circuit_params = x
        physical_params = self.physical_params(x)
        return circuit_params, physical_params
    


def get_network(name:str):
    if name.startswith("Cascade"):
        hidden_dim = int(name.split("_")[1])
        return Cascade_Net(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Invalid network name: {name}")